import math
import numpy as np
from core.entities import Job, Task, TaskType

class ReworkPolicy:
    def __init__(self, rework_load_factor: float, weight_dist_func: callable, max_rework_cycles: int = 5, decay: float = 0.7, task_type_mix: float = 1.0):
        self.rework_load_factor = rework_load_factor
        self.weight_dist_func = weight_dist_func
        self.max_rework_cycles = max_rework_cycles
        self.decay = decay
        self.task_type_mix = task_type_mix

    def apply_rework(self, job: Job, now: float) -> int:
        """
        差し戻しを適用し、生成された新規タスク（小実験）の数を返す。
        """
        if job.rework_count > self.max_rework_cycles:
            # 無限ループ防止のため、これ以上増殖させない
            return 0
        
        # 重みを決定（Step 2: 重み＝増殖を表現する器）
        # 指数減衰を伴うサンプリング
        current_weight = self.weight_dist_func() * (self.decay ** (job.rework_count - 1))
        job.rework_weight = current_weight
        
        # 小実験の増殖（rework_load_factor * rework_weight）
        n_new_tasks = math.ceil(self.rework_load_factor * current_weight)
        
        for i in range(n_new_tasks):
            # mix に基づいてタイプを決定（簡易的に実装、本来はrngが必要だが一旦固定または後で修正）
            # ここでは単純に SMALL_EXP をメインとする
            t_type = TaskType.SMALL_EXP
            
            new_task = Task(
                task_id=f"{job.job_id}_rework_{job.rework_count}_{i}",
                task_type=t_type,
                duration_days=0.0, 
                generated_by="REWORK",
                created_at=now
            )
            job.tasks.append(new_task)
            
        return n_new_tasks
