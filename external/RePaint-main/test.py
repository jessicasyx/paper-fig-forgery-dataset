# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
RePaint 图像修复测试脚本

功能说明：
    使用预训练的扩散模型（Diffusion Model）进行图像修复任务。
    可选使用分类器引导（Classifier Guidance）来提高生成质量。
    
主要流程：
    1. 加载预训练的扩散模型和分类器
    2. 读取待修复图像和掩码
    3. 通过扩散过程生成修复后的图像
    4. 保存生成结果（修复图像、原图、掩码等）

使用方法：
    cd E:\GraduationProject\paper-fig-forgery-dataset\external\RePaint-main
    python test.py --conf_path confs/test_inet256_thick.yml
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util

# Linux 系统兼容性处理（Windows 系统会自动跳过）
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,  # ImageNet 类别数量（1000）
    model_and_diffusion_defaults,  # 模型和扩散过程的默认参数
    classifier_defaults,  # 分类器的默认参数
    create_model_and_diffusion,  # 创建扩散模型
    create_classifier,  # 创建分类器
    select_args,  # 从配置中选择参数
)  # noqa: E402

def toU8(sample):
    """
    将张量转换为 uint8 格式的 numpy 数组
    
    功能说明：
        将扩散模型输出的张量（值范围 [-1, 1]）转换为标准图像格式（值范围 [0, 255]）
    
    参数:
        sample: torch.Tensor, 形状为 (batch, channels, height, width)
                值范围为 [-1, 1] 的浮点数张量
    
    返回:
        numpy.ndarray: 形状为 (batch, height, width, channels)
                       值范围为 [0, 255] 的 uint8 数组
                       如果输入为 None，则返回 None
    
    转换步骤：
        1. 将 [-1, 1] 线性映射到 [0, 255]
        2. 裁剪到有效范围并转换为 uint8 类型
        3. 调整维度顺序：NCHW -> NHWC（适配 OpenCV/PIL 格式）
        4. 转换为 numpy 数组
    """
    if sample is None:
        return sample

    # 步骤1: 将 [-1, 1] 映射到 [0, 255]
    # 公式: output = (input + 1) * 127.5
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    
    # 步骤2: 调整维度顺序 (batch, channels, height, width) -> (batch, height, width, channels)
    sample = sample.permute(0, 2, 3, 1)
    
    # 步骤3: 确保内存连续性（提高后续操作效率）
    sample = sample.contiguous()
    
    # 步骤4: 转换为 numpy 数组并移到 CPU
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf):
    """
    主函数：执行图像修复/生成流程
    
    参数:
        conf: conf_mgt.Default_Conf 配置对象
              包含所有模型、数据和采样参数
    
    主要流程:
        1. 初始化设备和模型
        2. 加载预训练权重
        3. 可选：加载分类器用于引导
        4. 批量处理图像
        5. 保存生成结果
    """

    print("Start", conf['name'])

    # ==================== 步骤1: 获取计算设备 ====================
    # 选择 GPU
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #本地用0 a40自己查nvtop

    # ==================== 步骤2: 创建并加载扩散模型 ====================
    # 根据配置创建 UNet 模型和扩散过程对象
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    
    # 加载预训练权重（从 .pt 文件）
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    
    # 将模型移动到计算设备（GPU/CPU）
    model.to(device)
    
    # 如果配置启用，转换为半精度浮点数（FP16）以加速计算
    if conf.use_fp16:
        model.convert_to_fp16()
    
    # 设置为评估模式（禁用 dropout 等训练特性）
    model.eval()

    # 是否显示进度条
    show_progress = conf.show_progress

    # ==================== 步骤3: 可选加载分类器用于引导采样 ====================
    # 分类器引导（Classifier Guidance）可以提高生成图像的质量和真实性
    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        
        # 创建分类器模型
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        
        # 加载分类器预训练权重
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        # 将分类器移动到计算设备
        classifier.to(device)
        
        # 如果配置启用，转换为半精度
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        
        # 设置为评估模式
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            """
            分类器引导函数：计算梯度以引导生成过程朝向特定类别
            
            原理：
                在扩散过程的每一步，使用分类器计算当前图像属于目标类别的概率，
                并计算梯度，引导生成过程朝向更符合目标类别的方向。
            
            参数:
                x: torch.Tensor, 当前时间步的噪声图像
                t: torch.Tensor, 当前时间步
                y: torch.Tensor, 目标类别标签
                gt: torch.Tensor, 真实图像（可选）
            
            返回:
                torch.Tensor: 梯度，用于引导采样过程
            """
            assert y is not None
            
            # 启用梯度计算
            with th.enable_grad():
                # 分离张量并启用梯度追踪
                x_in = x.detach().requires_grad_(True)
                
                # 使用分类器预测类别
                logits = classifier(x_in, t)
                
                # 计算对数概率
                log_probs = F.log_softmax(logits, dim=-1)
                
                # 选择目标类别的对数概率
                selected = log_probs[range(len(logits)), y.view(-1)]
                
                # 计算梯度并乘以引导强度
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        # 不使用分类器引导
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        """
        模型前向传播包装函数
        
        功能：
            根据配置决定是否使用类别条件，调用扩散模型进行前向传播
        
        参数:
            x: torch.Tensor, 当前噪声图像，形状 (batch, channels, height, width)
            t: torch.Tensor, 时间步，形状 (batch,)
            y: torch.Tensor, 类别标签，形状 (batch,)
            gt: torch.Tensor, 真实图像（用于图像修复任务）
        
        返回:
            torch.Tensor: 模型预测的噪声或图像
        """
        assert y is not None
        # 如果 class_cond=True，传递类别标签；否则传递 None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    # ==================== 步骤4: 准备数据加载器 ====================
    dset = 'eval'  # 使用评估数据集

    # 获取评估配置名称（从配置文件的 data.eval 部分）
    eval_name = conf.get_default_eval_name()

    # 获取数据加载器（包含待修复图像、掩码等）
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    # ==================== 步骤5: 批量处理图像 ====================
    for batch in iter(dl):

        # 将批次中的所有张量移动到计算设备（GPU/CPU）
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        # 准备模型输入参数
        model_kwargs = {}

        # GT (Ground Truth): 真实图像，形状 (batch, 3, height, width)
        model_kwargs["gt"] = batch['GT']

        # gt_keep_mask: 保留区域的掩码
        # 值为 1 的区域表示保留（不需要修复）
        # 值为 0 的区域表示需要修复（挖洞区域）
        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        # 获取批次大小
        batch_size = model_kwargs["gt"].shape[0]

        # ==================== 设置类别标签 ====================
        # 强制设置 y = 0（固定类别）
        classes = th.zeros(batch_size, dtype=th.long, device=device)
        model_kwargs["y"] = classes

        # ==================== 选择采样方法 ====================
        # DDPM: 标准扩散模型采样（慢但质量高）
        # DDIM: 加速采样方法（快但可能质量略低）
        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )


        # ==================== 执行采样生成 ====================
        result = sample_fn(
            model_fn,  # 模型前向传播函数
            (batch_size, 3, conf.image_size, conf.image_size),  # 输出形状
            clip_denoised=conf.clip_denoised,  # 是否裁剪去噪结果到 [-1, 1]
            model_kwargs=model_kwargs,  # 模型参数（GT、掩码、类别等）
            cond_fn=cond_fn,  # 分类器引导函数（可选）
            device=device,  # 计算设备
            progress=show_progress,  # 是否显示进度条
            return_all=True,  # 返回所有中间结果
            conf=conf  # 配置对象
        )
        
        # ==================== 步骤6: 转换结果为图像格式 ====================
        # srs (Sample/Super-Resolution): 生成的修复图像
        # 这是模型最终输出的结果，即修复后的完整图像
        srs = toU8(result['sample'])
        
        # gts (Ground Truth): 原始真实图像
        # 用于对比和评估修复质量
        gts = toU8(result['gt'])
        
        # lrs (Low-Resolution/Corrupted): 损坏的输入图像
        # 计算方式：保留区域使用原图，挖洞区域填充为白色（-1 对应白色）
        # 公式: lrs = gt * mask + (-1) * (1 - mask)
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        # gt_keep_masks: 掩码可视化图像
        # 将掩码从 [0, 1] 映射到 [-1, 1] 以便可视化
        # 黑色区域表示需要修复，白色区域表示保留
        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        # ==================== 步骤7: 保存结果图像 ====================
        conf.eval_imswrite(
            srs=srs,  # 生成的修复图像 -> 保存到 data/fake/repaint/
            gts=gts,  # 原始真实图像 -> 保存到 data/real/
            lrs=lrs,  # 挖洞损坏图像 -> 保存到 data/corrupted/
            gt_keep_masks=gt_keep_masks,  # 掩码图像 -> 保存到 data/mask/
            img_names=batch['GT_name'],  # 图像文件名列表
            dset=dset,  # 数据集类型（eval）
            name=eval_name,  # 评估配置名称
            verify_same=False  # 不验证文件是否相同（允许覆盖）
        )

    print("sampling complete")


if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None,
                        help='配置文件路径，例如: confs/test_inet256_thick.yml')
    args = vars(parser.parse_args())

    # ==================== 加载配置并运行 ====================
    # 创建默认配置对象
    conf_arg = conf_mgt.conf_base.Default_Conf()
    
    # 从 YAML 文件读取配置并更新
    conf_arg.update(yamlread(args.get('conf_path')))
    
    # 执行主函数
    main(conf_arg)
