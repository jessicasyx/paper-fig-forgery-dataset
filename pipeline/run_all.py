from scripts.mask.gen_batch_lrs import (
    generate_lrs_from_paths, 
    get_image_list,
    create_statistics_dict,
    update_statistics
)

# 获取图像列表
image_list = get_image_list(real_dir, output_dir, skip_existing=True)

# 初始化统计
stats = create_statistics_dict()

# 外部循环处理
for image_name in image_list:
    real_path = os.path.join(real_dir, image_name)
    mask_path = os.path.join(mask_dir, image_name)
    output_path = os.path.join(output_dir, image_name)
    
    # 调用核心处理函数
    success, error_msg = generate_lrs_from_paths(
        real_path, mask_path, output_path, fill_color=[255, 255, 255]
    )
    
    # 更新统计
    update_statistics(stats, success, error_msg, image_name)