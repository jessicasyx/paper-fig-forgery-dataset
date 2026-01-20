"""
批量图像分割工具 - SAM Batch Processing
使用 Segment Anything Model (SAM) 批量生成图像掩码
"""

import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time
from typing import Tuple, Optional, Dict, List


def get_default_sam_config() -> Dict:
    """
    获取默认的 SAM 配置参数
    
    返回:
        config: 配置字典
    """
    return {
        'model_type': 'vit_b',  # 可选: "vit_b", "vit_l", "vit_h"
        'max_image_size': 800,  # 图像最大边长（像素）
        'resize_enabled': True,  # 是否启用图像缩放
        'invert_mask': True,  # True=掩码区域黑色(0)，背景白色(255)
        'sam_params': {
            'points_per_side': 16,  # 采样点数量
            'pred_iou_thresh': 0.88,  # IoU阈值
            'stability_score_thresh': 0.95,  # 稳定性分数阈值
            'crop_n_layers': 0,  # 裁剪层数（0=不裁剪）
            'min_mask_region_area': 500  # 最小掩码区域面积
        }
    }


def initialize_sam_model(checkpoint_path: str, model_type: str = 'vit_b', 
                        sam_params: Optional[Dict] = None) -> Tuple[SamAutomaticMaskGenerator, str]:
    """
    初始化 SAM 模型和掩码生成器
    
    参数:
        checkpoint_path: 模型检查点路径
        model_type: 模型类型 ("vit_b", "vit_l", "vit_h")
        sam_params: SAM 参数配置字典
    
    返回:
        (mask_generator, device): 掩码生成器和使用的设备
    """
    # 确定使用的设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"正在加载 SAM 模型...")
    print(f"  - 使用设备：{device}")
    
    if device == "cuda:0":
        print(f"  - GPU名称：{torch.cuda.get_device_name(0)}")
        print(f"  - GPU显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载模型
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        print(f"  ✅ 模型加载成功")
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{e}")
    
    # 使用默认参数或自定义参数
    if sam_params is None:
        sam_params = get_default_sam_config()['sam_params']
    
    # 创建掩码生成器
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        **sam_params
    )
    
    return mask_generator, device


def select_best_mask(masks: List[Dict], image_area: int) -> Optional[Dict]:
    """
    智能选择最佳掩码
    
    选择策略（按优先级）：
    1. 优先选择面积占比在 10%-30% 之间的掩码（最优）
    2. 如果没有，选择面积占比在 5%-40% 之间的掩码
    3. 如果还没有，选择面积第二大的掩码
    4. 如果只有一个掩码，选择面积最大的
    
    参数:
        masks: SAM 生成的掩码列表
        image_area: 图像总面积（像素数）
    
    返回:
        best_mask: 最佳掩码，如果没有则返回 None
    """
    if len(masks) == 0:
        return None
    
    # 计算每个掩码的面积占比
    for mask in masks:
        mask['area_ratio'] = mask['area'] / image_area
    
    # 策略1: 优先选择面积占比在 10%-30% 之间的掩码
    optimal_masks = [m for m in masks if 0.10 <= m['area_ratio'] <= 0.30]
    if optimal_masks:
        # 选择最接近 20% 的掩码
        best_mask = min(optimal_masks, key=lambda m: abs(m['area_ratio'] - 0.20))
        return best_mask
    
    # 策略2: 选择面积占比在 5%-40% 之间的掩码
    acceptable_masks = [m for m in masks if 0.05 <= m['area_ratio'] <= 0.40]
    if acceptable_masks:
        # 选择最接近 20% 的掩码
        best_mask = min(acceptable_masks, key=lambda m: abs(m['area_ratio'] - 0.20))
        return best_mask
    
    # 策略3: 选择面积第二大的掩码
    if len(masks) >= 2:
        sorted_masks = sorted(masks, key=lambda m: m['area'], reverse=True)
        return sorted_masks[1]
    
    # 策略4: 只有一个掩码，选择面积最大的
    return max(masks, key=lambda m: m['area'])


def process_single_image(image_path: str, mask_generator: SamAutomaticMaskGenerator,
                        output_path: str, device: str = "cuda:0",
                        max_image_size: int = 800, resize_enabled: bool = True,
                        invert_mask: bool = True) -> Tuple[bool, str]:
    """
    处理单张图片，生成掩码
    
    参数:
        image_path: 输入图片路径
        mask_generator: SAM 掩码生成器
        output_path: 输出掩码路径
        device: 使用的设备 ("cuda:0" 或 "cpu")
        max_image_size: 图像最大边长（像素）
        resize_enabled: 是否启用图像缩放
        invert_mask: 是否反转掩码（True=掩码黑色，背景白色）
    
    返回:
        (success, message): 成功标志和消息
    """
    try:
        # 读取图片（支持中文路径）
        with open(image_path, 'rb') as f:
            image_data = np.frombuffer(f.read(), np.uint8)
            image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            # 无法读取图片，生成全黑掩码
            roi_mask = np.zeros((100, 100), dtype=np.uint8)
            if not invert_mask:
                roi_mask = 255 - roi_mask
            success, encoded_image = cv2.imencode('.png', roi_mask)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_image.tobytes())
            return True, "无法读取图片，生成全黑掩码"
        
        # 获取原始尺寸
        original_height, original_width = image_bgr.shape[:2]
        original_area = original_height * original_width
        
        # 如果启用缩放且图像过大，则缩小图像
        if resize_enabled:
            max_dim = max(original_height, original_width)
            if max_dim > max_image_size:
                scale = max_image_size / max_dim
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                image_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 生成掩码
        masks = mask_generator.generate(image_rgb)
        
        # 如果未生成任何掩码，生成全黑掩码
        if len(masks) == 0:
            roi_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            if not invert_mask:
                roi_mask = 255 - roi_mask
            success, encoded_image = cv2.imencode('.png', roi_mask)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_image.tobytes())
            return True, "未生成掩码，生成全黑掩码"
        
        # 智能选择最佳掩码
        current_area = image_rgb.shape[0] * image_rgb.shape[1]
        best_mask = select_best_mask(masks, current_area)
        
        if best_mask is None:
            # 如果选择失败，生成全黑掩码
            roi_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            if not invert_mask:
                roi_mask = 255 - roi_mask
            success, encoded_image = cv2.imencode('.png', roi_mask)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_image.tobytes())
            return True, "掩码选择失败，生成全黑掩码"
        
        # 提取选中的掩码
        roi_mask = (best_mask["segmentation"].astype(np.uint8) * 255)
        area_ratio = best_mask.get('area_ratio', 0) * 100
        
        # 如果图像被缩放了，将掩码恢复到原始尺寸
        if resize_enabled and (roi_mask.shape[0] != original_height or roi_mask.shape[1] != original_width):
            roi_mask = cv2.resize(roi_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        
        # 根据配置反转掩码（掩码区域变黑色，背景变白色）
        if invert_mask:
            roi_mask = 255 - roi_mask  # 反转：0变255，255变0
        
        # 保存掩码（支持中文路径）
        success, encoded_image = cv2.imencode('.png', roi_mask)
        if success:
            with open(output_path, 'wb') as f:
                f.write(encoded_image.tobytes())
            return True, f"成功 (共{len(masks)}个区域, 选中面积占比{area_ratio:.1f}%)"
        else:
            # 保存失败，生成全黑掩码
            roi_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            if not invert_mask:
                roi_mask = 255 - roi_mask
            success, encoded_image = cv2.imencode('.png', roi_mask)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_image.tobytes())
            return True, "保存失败，生成全黑掩码"
            
    except Exception as e:
        # 发生异常，生成全黑掩码
        try:
            roi_mask = np.zeros((original_height if 'original_height' in locals() else 100, 
                               original_width if 'original_width' in locals() else 100), dtype=np.uint8)
            if not invert_mask:
                roi_mask = 255 - roi_mask
            success, encoded_image = cv2.imencode('.png', roi_mask)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_image.tobytes())
            return True, f"处理异常({str(e)})，生成全黑掩码"
        except:
            return False, str(e)
    finally:
        # 清理GPU显存（如果使用GPU）
        if device == "cuda:0":
            try:
                torch.cuda.empty_cache()  # 清空缓存
                torch.cuda.synchronize()  # 同步GPU操作
            except RuntimeError:
                pass  # 忽略清理失败


def get_image_files(input_dir: str, extensions: List[str] = None) -> List[Path]:
    """
    获取目录中的所有图片文件
    
    参数:
        input_dir: 输入目录
        extensions: 图片扩展名列表，默认为 ['.png']
    
    返回:
        image_files: 图片文件路径列表
    """
    if extensions is None:
        extensions = ['.png']
    
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    return list(set(image_files))  # 去重


def filter_unprocessed_files(image_files: List[Path], output_dir: str) -> Tuple[List[Path], int]:
    """
    过滤出未处理的图片文件
    
    参数:
        image_files: 所有图片文件列表
        output_dir: 输出目录
    
    返回:
        (unprocessed_files, skipped_count): 未处理的文件列表和跳过的数量
    """
    unprocessed_files = []
    for img_file in image_files:
        output_filename = img_file.stem + ".png"
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
            unprocessed_files.append(img_file)
    
    skipped_count = len(image_files) - len(unprocessed_files)
    return unprocessed_files, skipped_count


def save_processing_log(stats: Dict, config: Dict, log_path: str) -> None:
    """
    保存处理日志
    
    参数:
        stats: 统计信息字典
        config: 配置信息字典
        log_path: 日志文件路径
    """
    total = stats['success'] + stats['failed']
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("批量图像分割处理日志\n")
        f.write("="*70 + "\n\n")
        f.write(f"处理时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入目录：{config.get('input_dir', 'N/A')}\n")
        f.write(f"输出目录：{config.get('output_dir', 'N/A')}\n")
        f.write(f"模型类型：{config.get('model_type', 'N/A')}\n")
        f.write(f"掩码模式：{'反转(掩码黑/背景白)' if config.get('invert_mask', True) else '正常(掩码白/背景黑)'}\n")
        f.write(f"使用设备：{config.get('device', 'N/A')}\n\n")
        f.write(f"处理结果：\n")
        f.write(f"  - 成功：{stats['success']} 张\n")
        f.write(f"  - 失败：{stats['failed']} 张\n")
        if total > 0:
            f.write(f"  - 成功率：{stats['success']/total*100:.1f}%\n\n")
        f.write(f"时间统计：\n")
        f.write(f"  - 总耗时：{stats['total_time']:.2f} 秒\n")
        if total > 0:
            f.write(f"  - 平均每张：{stats['total_time']/total:.2f} 秒\n\n")
        
        if stats['failed'] > 0:
            f.write("失败的文件：\n")
            for filename, reason in stats['failed_files']:
                f.write(f"  - {filename}: {reason}\n")


def main():
    """
    主函数：命令行运行入口
    """
    # 设置标准输出编码为 UTF-8
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    # 获取项目根目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/02_mask/
    SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)  # scripts/
    PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)  # paper-fig-forgery-dataset/
    
    # 配置参数
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "real")  # 输入图片目录
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "mask")  # 输出掩码目录
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")  # 日志目录
    
    # 获取默认配置
    config = get_default_sam_config()
    MODEL_TYPE = config['model_type']
    MAX_IMAGE_SIZE = config['max_image_size']
    RESIZE_ENABLED = config['resize_enabled']
    INVERT_MASK = config['invert_mask']
    SAM_PARAMS = config['sam_params']
    
    # 配置参数：可选 B/H/L 三种 SAM 模型及权重
    SAM_CHECKPOINTS = {
        "vit_b": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_b_01ec64.pth"),
        "vit_h": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_h_4b8939.pth"),
        "vit_l": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_l_0b3195.pth"),
    }
    CKPT_PATH = SAM_CHECKPOINTS[MODEL_TYPE]
    
    # 是否跳过已处理的图片
    SKIP_EXISTING = True

    print("="*70)
    print("批量图像分割工具 - SAM Batch Processing")
    print("="*70)
    print(f"\n配置信息：")
    print(f"  - 输入目录：{INPUT_DIR}")
    print(f"  - 输出目录：{OUTPUT_DIR}")
    print(f"  - 模型类型：{MODEL_TYPE}")
    print(f"  - 图像缩放：{'启用' if RESIZE_ENABLED else '禁用'} (最大边长: {MAX_IMAGE_SIZE}px)")
    print(f"  - 掩码模式：{'反转(掩码黑/背景白)' if INVERT_MASK else '正常(掩码白/背景黑)'}")
    print(f"  - 跳过已处理：{SKIP_EXISTING}")

    # 创建输出目录和日志目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 获取所有图片文件
    image_files = get_image_files(INPUT_DIR, extensions=['.png'])
    
    print(f"\n找到 {len(image_files)} 张图片")

    if len(image_files) == 0:
        print(f"❌ 错误：在 {INPUT_DIR} 中没有找到图片文件")
        sys.exit(1)

    # 检查已处理的图片
    if SKIP_EXISTING:
        image_files, skipped_count = filter_unprocessed_files(image_files, OUTPUT_DIR)
        
        if skipped_count > 0:
            print(f"  - 跳过已处理：{skipped_count} 张")
            print(f"  - 待处理：{len(image_files)} 张")

    if len(image_files) == 0:
        print("\n✅ 所有图片已处理完成！")
        sys.exit(0)

    # 初始化 SAM 模型
    try:
        mask_generator, device = initialize_sam_model(CKPT_PATH, MODEL_TYPE, SAM_PARAMS)
    except Exception as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    # 统计信息
    stats = {
        'success': 0,
        'failed': 0,
        'total_time': 0,
        'failed_files': []
    }

    # 开始批量处理
    print("\n" + "="*70)
    print("开始批量处理")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        # 使用 tqdm 显示进度条，循环处理每张图片
        for img_file in tqdm(image_files, desc="处理进度", unit="张"):
            img_start_time = time.time()
            
            # 构建输出路径
            output_filename = img_file.stem + ".png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # 处理单张图片
            success, message = process_single_image(
                str(img_file), 
                mask_generator, 
                output_path,
                device=device,
                max_image_size=MAX_IMAGE_SIZE,
                resize_enabled=RESIZE_ENABLED,
                invert_mask=INVERT_MASK
            )
            
            img_time = time.time() - img_start_time
            stats['total_time'] += img_time
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
                stats['failed_files'].append((img_file.name, message))
                tqdm.write(f"❌ {img_file.name}: {message}")
                    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断处理")
    except Exception as e:
        print(f"\n\n❌ 处理过程中发生错误: {e}")
    finally:
        total_time = time.time() - start_time

    # 输出统计结果
    print("\n" + "="*70)
    print("处理完成 - 统计报告")
    print("="*70)

    total = stats['success'] + stats['failed']
    if total > 0:
        print(f"\n处理结果：")
        print(f"  ✅ 成功：{stats['success']} 张")
        print(f"  ❌ 失败：{stats['failed']} 张")
        print(f"  📊 成功率：{stats['success']/total*100:.1f}%")

        print(f"\n时间统计：")
        print(f"  - 总耗时：{total_time:.2f} 秒")
        print(f"  - 平均每张：{stats['total_time']/total:.2f} 秒")

        if stats['failed'] > 0:
            print(f"\n失败的文件：")
            for filename, reason in stats['failed_files']:
                print(f"  - {filename}: {reason}")

        print(f"\n输出目录：{OUTPUT_DIR}")
        print("="*70)

        # 生成处理日志
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_filename = f"batch_processing_mask{timestamp}.log"
        log_path = os.path.join(LOG_DIR, log_filename)
        
        # 保存配置信息用于日志
        log_config = {
            'input_dir': INPUT_DIR,
            'output_dir': OUTPUT_DIR,
            'model_type': MODEL_TYPE,
            'invert_mask': INVERT_MASK,
            'device': device
        }
        
        save_processing_log(stats, log_config, log_path)
        print(f"\n日志已保存到：{log_path}")


if __name__ == "__main__":
    main()

