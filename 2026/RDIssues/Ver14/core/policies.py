import math
import re
import numpy as np
from core.entities import Job, Task, TaskType

class ReworkPolicy:
    def __init__(
        self,
        rework_load_factor: float,
        weight_dist_func: callable,
        max_rework_cycles: int = 5,
        decay: float = 0.7,
        task_type_mix: float = 1.0,
        reinject_ratio: float = None,
        reinject_mode: str = "all",
    ):
        self.rework_load_factor = rework_load_factor
        self.weight_dist_func = weight_dist_func
        self.max_rework_cycles = max_rework_cycles
        self.decay = decay
        self.task_type_mix_raw = task_type_mix
        mix_weights, legacy_ratio = self._parse_task_type_mix(task_type_mix)
        self.task_type_mix = mix_weights
        if reinject_ratio is None:
            reinject_ratio = legacy_ratio if legacy_ratio is not None else 1.0
        self.reinject_ratio = self._clip_ratio(reinject_ratio, default=1.0)
        mode = reinject_mode if isinstance(reinject_mode, str) else None
        self.reinject_mode = (mode or "all").strip().lower()
        if self.reinject_mode not in ("all", "ratio"):
            self.reinject_mode = "all"

    @staticmethod
    def _coerce_float(value):
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(val):
            return None
        return val

    @classmethod
    def _clip_ratio(cls, value, default: float = 1.0) -> float:
        val = cls._coerce_float(value)
        if val is None:
            return float(default)
        return max(0.0, min(1.0, val))

    @staticmethod
    def _normalize_mix(values):
        cleaned = [max(0.0, float(v)) for v in values]
        total = sum(cleaned)
        if total <= 0.0:
            return (1.0, 0.0, 0.0)
        return tuple(v / total for v in cleaned)

    @classmethod
    def _parse_task_type_mix(cls, mix):
        if mix is None:
            return (1.0, 0.0, 0.0), None

        if isinstance(mix, (int, float, np.floating)):
            val = cls._coerce_float(mix)
            if val is None:
                return (1.0, 0.0, 0.0), None
            return (1.0, 0.0, 0.0), val

        if isinstance(mix, str):
            s = mix.strip()
            if not s:
                return (1.0, 0.0, 0.0), None
            if any(sep in s for sep in ("|", "/", ":", ";")):
                parts = [p.strip() for p in re.split(r"[|/:;]+", s) if p.strip()]
                if len(parts) == 3:
                    vals = [cls._coerce_float(p) for p in parts]
                    if all(v is not None for v in vals):
                        return cls._normalize_mix(vals), None
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if len(parts) == 3:
                    vals = [cls._coerce_float(p) for p in parts]
                    if all(v is not None for v in vals):
                        return cls._normalize_mix(vals), None
            val = cls._coerce_float(s)
            if val is not None:
                return (1.0, 0.0, 0.0), val
            return (1.0, 0.0, 0.0), None

        if isinstance(mix, dict):
            val_small = None
            val_mid = None
            val_fin = None
            if TaskType.SMALL_EXP in mix:
                val_small = mix.get(TaskType.SMALL_EXP)
            if TaskType.MID_EXP in mix:
                val_mid = mix.get(TaskType.MID_EXP)
            if TaskType.FIN_EXP in mix:
                val_fin = mix.get(TaskType.FIN_EXP)
            for k, v in mix.items():
                if not isinstance(k, str):
                    continue
                key = k.strip().lower()
                if key in ("small", "small_exp", "smallexp"):
                    val_small = v
                elif key in ("mid", "mid_exp", "midexp"):
                    val_mid = v
                elif key in ("fin", "fin_exp", "finexp"):
                    val_fin = v
            if val_small is not None or val_mid is not None or val_fin is not None:
                vals = [
                    cls._coerce_float(val_small) or 0.0,
                    cls._coerce_float(val_mid) or 0.0,
                    cls._coerce_float(val_fin) or 0.0,
                ]
                return cls._normalize_mix(vals), None

        if isinstance(mix, (list, tuple, np.ndarray)) and len(mix) >= 3:
            vals = [
                cls._coerce_float(mix[0]) or 0.0,
                cls._coerce_float(mix[1]) or 0.0,
                cls._coerce_float(mix[2]) or 0.0,
            ]
            return cls._normalize_mix(vals), None

        return (1.0, 0.0, 0.0), None

    @staticmethod
    def _allocate_counts(total: int, weights, eligible=None):
        """Allocates counts based on weights and eligibility"""
        if total <= 0:
            return [0, 0, 0]
        raw = [total * w for w in weights]
        base = [int(math.floor(v)) for v in raw]
        remainder = total - sum(base)
        frac = [r - b for r, b in zip(raw, base)]
        indices = list(range(len(base)))
        if eligible is not None:
            indices = [i for i in indices if eligible[i]]
        indices.sort(key=lambda i: frac[i], reverse=True)
        for i in indices:
            if remainder <= 0:
                break
            base[i] += 1
            remainder -= 1
        return base

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
        
        # 重みに応じて追加の Task を生成（増殖）
        n_new_tasks = math.ceil(self.rework_load_factor * current_weight)
        if self.reinject_mode == "all":
            n_reinject = n_new_tasks
        else:
            n_reinject = int(round(n_new_tasks * self.reinject_ratio))
            n_reinject = max(0, min(n_reinject, n_new_tasks))

        counts = self._allocate_counts(n_new_tasks, self.task_type_mix)
        task_counts = {
            TaskType.SMALL_EXP: counts[0],
            TaskType.MID_EXP: counts[1],
            TaskType.FIN_EXP: counts[2],
        }

        reinject_counts = {TaskType.SMALL_EXP: 0, TaskType.MID_EXP: 0, TaskType.FIN_EXP: 0}
        # Splits reinjected tasks respecting task type counts
        if n_reinject > 0 and n_new_tasks > 0:
            eligible = [c > 0 for c in counts]
            reinject_split = self._allocate_counts(n_reinject, self._normalize_mix(counts), eligible=eligible)
            for task_type, count in zip(
                (TaskType.SMALL_EXP, TaskType.MID_EXP, TaskType.FIN_EXP),
                reinject_split,
            ):
                reinject_counts[task_type] = min(count, task_counts.get(task_type, 0))

        for task_type, count in task_counts.items():
            for i in range(count):
                new_task = Task(
                    task_id=f"{job.job_id}_rework_{job.rework_count}_{task_type.name}_{i}",
                    task_type=task_type,
                    duration_days=0.0,
                    generated_by="REWORK",
                    created_at=now
                )
                job.tasks.append(new_task)

        return {
            "n_new_tasks": n_new_tasks,
            "n_reinject": n_reinject,
            "task_counts": task_counts,
            "reinject_counts": reinject_counts,
            "reinject_mode": self.reinject_mode,
            "reinject_ratio": self.reinject_ratio,
        }
