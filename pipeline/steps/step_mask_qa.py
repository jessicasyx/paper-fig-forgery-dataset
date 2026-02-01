# pipeline/steps/step_mask_qa.py
"""
Step: Mask QA 评估（硬筛选）
输入：mask_path
输出：QA评分结果（dict）
"""

import os
from typing import Tuple, Dict, Any

from pipeline.utils.qa import eval_mask_qa


def run_step_mask_qa(
    mask_path: str,
    real_shape: tuple = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    对单个mask进行QA评估（硬筛选）
    
    返回：
        (ok, qa_result)
        ok: 是否成功执行评估（不代表是否通过QA）
        qa_result: 包含评分结果的字典
    """
    try:
        if not os.path.exists(mask_path):
            return False, {
                "hard_fail": True,
                "error": "mask_not_found",
                "reason": "mask file not found",
                "tier": "FAIL"
            }
        
        if os.path.getsize(mask_path) == 0:
            return False, {
                "hard_fail": True,
                "error": "mask_empty",
                "reason": "mask file is empty",
                "tier": "FAIL"
            }
        
        # 调用核心评估函数
        qa_result = eval_mask_qa(
            mask_path=mask_path,
            real_shape=real_shape
        )
        
        return True, qa_result
        
    except Exception as e:
        return False, {
            "hard_fail": True,
            "error": f"exception:{repr(e)}",
            "reason": "evaluation exception",
            "tier": "FAIL"
        }
