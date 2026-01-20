import math
import numpy as np
from core.entities import Job, Task, TaskType

class ReworkPolicy:
    def __init__(self, rework_load_factor: float, weight_dist_func: callable, max_rework_cycles: int = 5, decay: float = 0.7):
        self.rework_load_factor = rework_load_factor
        self.weight_dist_func = weight_dist_func
        self.max_rework_cycles = max_rework_cycles
        self.decay = decay

    def apply_rework(self, job: Job, now: float) -> int:
        """
        差し戻しを適用し、生成された新規タスク（小実験）の数を返す。
        """
        if job.rework_count > self.max_rework_cycles:
            # 無限ループ防止のため、これ以上増殖させない（または終了させる）
            return 0
        
        # 重みを決定
        job.rework_weight = self.weight_dist_func() * (self.decay ** (job.rework_count - 1))
        
        # 小実験の増殖
        n_new_tasks = math.ceil(self.rework_load_factor * job.rework_weight)
        
        for i in range(n_new_tasks):
            new_task = Task(
                task_id=f"{job.job_id}_rework_{job.rework_count}_{i}",
                task_type=TaskType.SMALL_EXP,
                duration_days=0.0, # あとでWorkGateなどでサンプリングされる前提、またはここで決める
                generated_by="REWORK",
                created_at=now
            )
            job.tasks.append(new_task)
            
        return n_new_tasks
