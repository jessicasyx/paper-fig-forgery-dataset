# pipeline/steps/step_sample_qa.py
"""
Step: Fake QA 评估（灾难级筛选）
输入：real_path, mask_path, fake_path
输出：QA评分结果（dict）
"""

import os
from typing import Tuple, Dict, Any

from pipeline.utils.qa import eval_fake_qa


def run_step_sample_qa(
    real_path: str,
    mask_path: str,
    fake_path: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    对fake图像进行质量评估（灾难级筛选）
    
    绝大多数样本：只打分、只记录，不淘汰
    只有灾难级：淘汰（否则污染数据集）
    
    返回：
        (ok, qa_result)
        ok: 是否成功执行评估
        qa_result: 包含评分结果的字典
    """
    try:
        # 检查文件存在性
        if not os.path.exists(fake_path) or os.path.getsize(fake_path) == 0:
            return False, {
                "hard_fail": True,
                "error": "fake_not_found_or_empty",
                "reason": "fake file not found or empty",
                "score": 0
            }
        
        if not os.path.exists(real_path) or os.path.getsize(real_path) == 0:
            return False, {
                "hard_fail": True,
                "error": "real_not_found_or_empty",
                "reason": "real file not found or empty",
                "score": 0
            }
        
        if not os.path.exists(mask_path) or os.path.getsize(mask_path) == 0:
            return False, {
                "hard_fail": True,
                "error": "mask_not_found_or_empty",
                "reason": "mask file not found or empty",
                "score": 0
            }
        
        # 调用核心评估函数
        qa_result = eval_fake_qa(
            real_path=real_path,
            fake_path=fake_path,
            mask_path=mask_path
        )
        
        return True, qa_result
        
    except Exception as e:
        return False, {
            "hard_fail": True,
            "error": f"exception:{repr(e)}",
            "reason": "evaluation exception",
            "score": 0
        }
