"""
批量生成挖洞图像（LRS）工具
将真实图像根据mask生成corrupted图像（挖洞图像）
"""

import os
import cv2
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import sys


def generate_single_lrs(real_image: np.ndarray, mask_image: np.ndarray, 
                        fill_color: List[int] = [255, 255, 255]) -> np.ndarray:
    """
    生成单张挖洞图像（LRS）
    
    参数:
        real_image: 真实图像（BGR格式）
        mask_image: mask图像（灰度图，0表示需要修补的区域）
        fill_color: 填充颜色，默认白色[255, 255, 255]
    
    返回:
        lrs_image: 生成的挖洞图像
    """
    # 创建 lrs 图像（复制真实图像）
    lrs_image = real_image.copy()
    
    # 将 mask 为 0 的区域（需要修补的区域）设为指定颜色
    lrs_image[mask_image == 0] = fill_color
    
    return lrs_image


def read_image_with_chinese_path(image_path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    读取图像（支持中文路径）
    
    参数:
        image_path: 图像路径
        flags: OpenCV读取标志（IMREAD_COLOR或IMREAD_GRAYSCALE）
    
    返回:
        image: 读取的图像，失败返回None
    """
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags)
        return image
    except Exception:
        return None


def save_image_with_chinese_path(image: np.ndarray, output_path: str, 
                                 extension: str = '.png') -> bool:
    """
    保存图像（支持中文路径）
    
    参数:
        image: 要保存的图像
        output_path: 输出路径
        extension: 文件扩展名，默认'.png'
    
    返回:
        success: 是否保存成功
    """
    try:
        cv2.imencode(extension, image)[1].tofile(output_path)
        return True
    except Exception:
        return False


def update_statistics(stats: Dict, success: bool, error_msg: str, 
                     image_name: str) -> None:
    """
    更新统计信息
    
    参数:
        stats: 统计信息字典
        success: 是否成功
        error_msg: 错误信息
        image_name: 图像文件名
    """
    if success:
        stats['success'] += 1
    else:
        stats['failed'] += 1
        stats['failed_files'].append((image_name, error_msg))
        
        # 更新错误类型统计
        if "未找到对应的mask文件" in error_msg:
            stats['no_mask'] += 1
        elif "无法读取" in error_msg:
            stats['read_error'] += 1
        elif "尺寸不匹配" in error_msg:
            stats['size_mismatch'] += 1


def create_statistics_dict() -> Dict:
    """
    创建统计信息字典
    
    返回:
        stats: 初始化的统计信息字典
    """
    return {
        'success': 0,
        'failed': 0,
        'no_mask': 0,
        'size_mismatch': 0,
        'read_error': 0,
        'failed_files': [],
        'total_time': 0
    }


def generate_lrs_from_paths(real_image_path: str, mask_image_path: str, 
                           output_image_path: str,
                           fill_color: List[int] = [255, 255, 255]) -> Tuple[bool, str]:
    """
    从指定路径读取图像并生成LRS图像（核心处理函数）
    
    参数:
        real_image_path: 真实图像路径
        mask_image_path: mask图像路径
        output_image_path: 输出图像路径
        fill_color: 填充颜色，默认白色[255, 255, 255]
    
    返回:
        (success, error_message): 成功标志和错误信息
    """
    try:
        # 检查是否存在对应的 mask 图像
        if not os.path.exists(mask_image_path):
            return False, "未找到对应的mask文件"
        
        # 读取真实图像和 mask 图像
        real_image = read_image_with_chinese_path(real_image_path, cv2.IMREAD_COLOR)
        mask_image = read_image_with_chinese_path(mask_image_path, cv2.IMREAD_GRAYSCALE)
        
        # 检查图像是否成功读取
        if real_image is None:
            return False, "无法读取real图像"
        if mask_image is None:
            return False, "无法读取mask图像"
        
        # 确保真实图像和 mask 图像的尺寸一致
        if real_image.shape[:2] != mask_image.shape[:2]:
            return False, f"尺寸不匹配 real:{real_image.shape[:2]} mask:{mask_image.shape[:2]}"
        
        # 生成 lrs 图像
        lrs_image = generate_single_lrs(real_image, mask_image, fill_color)
        
        # 保存生成的 lrs 图像
        success = save_image_with_chinese_path(lrs_image, output_image_path)
        
        if not success:
            return False, "保存图像失败"
        
        return True, ""
        
    except Exception as e:
        return False, str(e)


def get_image_list(real_dir: str, output_dir: str, skip_existing: bool = True) -> List[str]:
    """
    获取需要处理的图像列表
    
    参数:
        real_dir: 真实图像目录
        output_dir: 输出目录
        skip_existing: 是否跳过已处理的图片
    
    返回:
        image_list: 需要处理的图像文件名列表
    """
    # 获取所有图片文件名
    all_images = os.listdir(real_dir)
    
    # 检查已处理的图片
    if skip_existing:
        unprocessed_files = []
        for image_name in all_images:
            output_path = os.path.join(output_dir, image_name)
            if not os.path.exists(output_path):
                unprocessed_files.append(image_name)
        return unprocessed_files
    
    return all_images


def print_statistics(stats: Dict, output_dir: str):
    """
    打印统计信息
    
    参数:
        stats: 统计信息字典
        output_dir: 输出目录
    """
    print("\n" + "="*70)
    print("处理完成 - 统计报告")
    print("="*70)
    
    total = stats['success'] + stats['failed']
    
    print(f"\n处理结果：")
    print(f"  ✅ 成功：{stats['success']} 张")
    print(f"  ❌ 失败：{stats['failed']} 张")
    if stats['failed'] > 0:
        print(f"     - 未找到mask：{stats['no_mask']} 张")
        print(f"     - 读取错误：{stats['read_error']} 张")
        print(f"     - 尺寸不匹配：{stats['size_mismatch']} 张")
    if total > 0:
        print(f"  📊 成功率：{stats['success']/total*100:.1f}%")
    
    print(f"\n时间统计：")
    print(f"  - 总耗时：{stats['total_time']:.2f} 秒")
    if total > 0:
        print(f"  - 平均每张：{stats['total_time']/total:.2f} 秒")
    
    if stats['failed'] > 0:
        print(f"\n失败的文件：")
        for filename, reason in stats['failed_files'][:10]:  # 只显示前10个
            print(f"  - {filename}: {reason}")
        if len(stats['failed_files']) > 10:
            print(f"  ... 还有 {len(stats['failed_files']) - 10} 个失败文件")
    
    print(f"\n输出目录：{output_dir}")
    print("="*70)


def save_log(stats: Dict, real_dir: str, mask_dir: str, output_dir: str, log_dir: str):
    """
    保存处理日志
    
    参数:
        stats: 统计信息字典
        real_dir: 真实图像目录
        mask_dir: mask图像目录
        output_dir: 输出目录
        log_dir: 日志目录
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_filename = f"batch_processing_lrs_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    total = stats['success'] + stats['failed']
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("批量生成挖洞图像处理日志\n")
        f.write("="*70 + "\n\n")
        f.write(f"处理时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入目录（real）：{real_dir}\n")
        f.write(f"输入目录（mask）：{mask_dir}\n")
        f.write(f"输出目录（corrupted）：{output_dir}\n\n")
        f.write(f"处理结果：\n")
        f.write(f"  - 成功：{stats['success']} 张\n")
        f.write(f"  - 失败：{stats['failed']} 张\n")
        if stats['failed'] > 0:
            f.write(f"     - 未找到mask：{stats['no_mask']} 张\n")
            f.write(f"     - 读取错误：{stats['read_error']} 张\n")
            f.write(f"     - 尺寸不匹配：{stats['size_mismatch']} 张\n")
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
    
    print(f"\n日志已保存到：{log_path}")


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
    real_dir = os.path.join(PROJECT_ROOT, "data", "real")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "mask")
    output_dir = os.path.join(PROJECT_ROOT, "data", "corrupted")
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    
    # 是否跳过已处理的图片
    SKIP_EXISTING = True
    fill_color = [255, 255, 255]  # 白色填充
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 打印配置信息
    print("="*70)
    print("批量生成挖洞图像工具 - LRS Batch Generation")
    print("="*70)
    print(f"\n配置信息：")
    print(f"  - 输入目录（real）：{real_dir}")
    print(f"  - 输入目录（mask）：{mask_dir}")
    print(f"  - 输出目录（corrupted）：{output_dir}")
    print(f"  - 跳过已处理：{SKIP_EXISTING}")
    
    # 获取需要处理的图像列表
    all_images = os.listdir(real_dir)
    print(f"\n找到 {len(all_images)} 张图片")
    
    image_list = get_image_list(real_dir, output_dir, skip_existing=SKIP_EXISTING)
    
    skipped_count = len(all_images) - len(image_list)
    if skipped_count > 0:
        print(f"  - 跳过已处理：{skipped_count} 张")
        print(f"  - 待处理：{len(image_list)} 张")
    
    if len(image_list) == 0:
        print("\n✅ 所有图片已处理完成！")
        return
    
    # 初始化统计信息
    stats = create_statistics_dict()
    
    print("\n" + "="*70)
    print("开始批量处理")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # 循环处理每张图像
    for image_name in tqdm(image_list, desc="处理进度", unit="张"):
        # 构建路径
        real_image_path = os.path.join(real_dir, image_name)
        mask_image_path = os.path.join(mask_dir, image_name)
        output_image_path = os.path.join(output_dir, image_name)
        
        # 处理单张图像
        success, error_msg = generate_lrs_from_paths(
            real_image_path, mask_image_path, output_image_path, fill_color
        )
        
        # 更新统计信息
        update_statistics(stats, success, error_msg, image_name)
        
        if not success:
            tqdm.write(f"❌ {image_name}: {error_msg}")
    
    # 记录总耗时
    stats['total_time'] = time.time() - start_time
    
    # 打印统计结果
    print_statistics(stats, output_dir)
    
    # 保存日志
    save_log(stats, real_dir, mask_dir, output_dir, log_dir)


if __name__ == "__main__":
    main()
