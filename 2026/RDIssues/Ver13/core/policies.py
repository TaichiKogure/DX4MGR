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

    def apply_rework(self, job: Job, now: float) -> dict:
        """
        Step 5: 差し戻し“重み”を「増殖」に変換するルール
        差し戻しを適用し、生成された新規タスク数と再投入数を返す。
        """
        # Step 5.2 ⚠️引っかかり：無限ループ防止のため、上限（max_rework_cycles）を適用
        if job.rework_count > self.max_rework_cycles:
            return {"n_new_tasks": 0, "n_reinject": 0}
        
        # Step 5.1 基本ルール: 重みを決定し、指数減衰（decay）を伴うサンプリング
        current_weight = self.weight_dist_func() * (self.decay ** (job.rework_count - 1))
        job.rework_weight = current_weight
        
        # 重みに応じて追加の Task(SMALL_EXP) を生成（増殖）
        n_new_tasks = math.ceil(self.rework_load_factor * current_weight)
        try:
            mix = float(self.task_type_mix)
        except (TypeError, ValueError):
            mix = 1.0
        if math.isnan(mix):
            mix = 1.0
        mix = max(0.0, min(mix, 1.0))
        n_reinject = int(round(n_new_tasks * mix))
        n_reinject = max(0, min(n_reinject, n_new_tasks))
        
        for i in range(n_new_tasks):
            t_type = TaskType.SMALL_EXP
            
            new_task = Task(
                task_id=f"{job.job_id}_rework_{job.rework_count}_{i}",
                task_type=t_type,
                duration_days=0.0, 
                generated_by="REWORK",
                created_at=now
            )
            job.tasks.append(new_task)
            
        return {"n_new_tasks": n_new_tasks, "n_reinject": n_reinject}
