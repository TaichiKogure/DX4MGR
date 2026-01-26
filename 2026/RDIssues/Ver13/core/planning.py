from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import bisect
import math

@dataclass
class DRCalendar:
    # times are "days since t0" で engine.now と同じ単位に揃える
    schedule_by_gate: Dict[str, List[float]]  # {"DR1":[...], "DR2":[...], ...}

    def next_after(self, gate_id: str, now: float) -> Optional[float]:
        times = self.schedule_by_gate.get(gate_id, [])
        i = bisect.bisect_right(times, now)
        return times[i] if i < len(times) else None

@dataclass
class WorkPackage:
    wp_id: str
    job_id: str
    gate_target: str          # "DR1"/"DR2" etc (どのDRのための準備か)
    kind: str                 # "experiment" / "analysis" / "doc" / "risk_burn"
    effort_hours: float       # これが「割り当て工数」
    due_time: float           # 次DR日時（days）
    created_at: float
    meta: Dict[str, Any] = None

@dataclass
class LatentRisk:
    # 0..1 っぽいスケールで持つ（雑に始められる）
    uncertainty: float = 1.0      # U：未検証在庫（高いほどヤバい）
    latent_risk: float = 1.0      # R：潜伏地雷の量
    evidence: float = 0.0         # 根拠量（増えるほど uncertainty を減らす）
    scale_factor: float = 1.0     # DR2で上がる想定（例 3.0）

    def apply_work(self, wp_kind: str, effort_hours: float):
        """
        Schedulerが割り当てた工数に応じて状態を更新。
        ここは最初は単純でOK（逓減も後で入れられる）
        """
        # 効き方（仮）：kind別の係数
        k = {"experiment": 0.030, "analysis": 0.020, "doc": 0.010, "risk_burn": 0.040}.get(wp_kind, 0.015)
        gain = 1.0 - math.exp(-k * max(effort_hours, 0.0))
        self.evidence += gain
        # evidenceが増えると uncertainty が減る
        self.uncertainty *= (1.0 - 0.5 * gain)
        # 潜伏リスクも少し減る（ただしdocはあまり減らない…みたいな思想も入れられる）
        risk_k = {"experiment": 0.035, "analysis": 0.020, "doc": 0.005, "risk_burn": 0.045}.get(wp_kind, 0.015)
        self.latent_risk *= (1.0 - 0.6 * (1.0 - math.exp(-risk_k * effort_hours)))

        # 下限クリップ
        self.uncertainty = max(self.uncertainty, 0.02)
        self.latent_risk = max(self.latent_risk, 0.02)

    def gate_modifiers(self, gate_id: str) -> Dict[str, float]:
        """
        MeetingGateに渡す補正値を返す。
        - DR1: 資料完成でも通りやすい（uncertaintyの罰は小さめ）
        - DR2: scaleでlatentが顕在化（爆発）
        """
        if gate_id == "DR1":
            q_mult = 1.0 - 0.25 * self.uncertainty
            cond_boost = 0.20 * self.uncertainty     # 条件付き（差戻し）が増えやすい
            nogo_boost = 0.05 * self.uncertainty
            return {"quality_mult": q_mult, "conditional_add": cond_boost, "nogo_add": nogo_boost}

        if gate_id == "DR2":
            # “顕在化ハザード”：latent_risk * scale が大きいほどヤバい
            hazard = self.latent_risk * self.scale_factor
            explosion = 1.0 - math.exp(-1.2 * hazard)   # 0..1
            q_mult = 1.0 - 0.55 * self.uncertainty - 0.40 * explosion
            cond_boost = 0.35 * explosion
            nogo_boost = 0.20 * explosion
            return {"quality_mult": q_mult, "conditional_add": cond_boost, "nogo_add": nogo_boost}

        if gate_id == "DR3":
            # ここは製品化判定＝厳しい、など
            q_mult = 1.0 - 0.35 * self.uncertainty
            return {"quality_mult": q_mult, "conditional_add": 0.10 * self.uncertainty, "nogo_add": 0.10 * self.uncertainty}

        return {"quality_mult": 1.0, "conditional_add": 0.0, "nogo_add": 0.0}

@dataclass
class Scheduler:
    engine: Any
    calendar: DRCalendar

    # ざっくりの人員能力（後で部署/スキルに拡張できる）
    engineer_pool_size: int = 10
    hours_per_day_per_engineer: float = 6.0

    tick_days: float = 1.0  # 1日ごとに計画更新（AnyLogicの定期イベントっぽい）

    def start(self, t0: float = 0.0):
        self.engine.schedule_event(t0, "SCHED_TICK", {"scheduler": self}, priority=7)

    def on_tick(self, now: float):
        # 次回tick
        self.engine.schedule_event(now + self.tick_days, "SCHED_TICK", {"scheduler": self}, priority=7)

        # 全ジョブを眺めて、DR締切までの残りを見て工数配分
        active_jobs = self._collect_active_jobs()

        total_hours_today = self.engineer_pool_size * self.hours_per_day_per_engineer

        # 例：最も近い締切順（EDF）
        plan_items = []
        for job in active_jobs:
            gate = self._target_gate_for(job)  # 例：jobの現在地からDR1/DR2/DR3を返す
            due = self.calendar.next_after(gate, now)
            if due is None:
                continue
            slack = max(due - now, 0.0)
            urgency = 1.0 / max(slack, 0.25)   # slack小さいほど緊急
            plan_items.append((urgency, job, gate, due))

        plan_items.sort(key=lambda x: x[0], reverse=True)

        # 今日は各ジョブに割り当てる工数（超シンプル割当）
        for urgency, job, gate, due in plan_items:
            if total_hours_today <= 0:
                break

            # 例：riskが高いほど「risk_burn」へ寄せる（DR2爆発対策）
            lr = getattr(job, "latent", None)
            if lr is None:
                continue

            kind = "experiment"
            if gate == "DR2" and lr.latent_risk > 0.5:
                kind = "risk_burn"
            elif lr.uncertainty > 0.6:
                kind = "analysis"
            else:
                kind = "doc"

            # ざっくり：緊急度に応じて今日の割当を決める
            alloc = min(total_hours_today, 2.0 + 6.0 * urgency)  # 2h〜多め
            total_hours_today -= alloc

            wp = WorkPackage(
                wp_id=f"WP_{job.job_id}_{gate}_{int(now*100)}",
                job_id=job.job_id,
                gate_target=gate,
                kind=kind,
                effort_hours=alloc,
                due_time=due,
                created_at=now,
                meta={"urgency": urgency}
            )
            # jobに計画を積む（計画ログ）
            if not hasattr(job, "work_packages"):
                job.work_packages = []
            job.work_packages.append(wp)

            # ここが肝：割当工数で潜伏リスク状態を更新＝品質が決まる
            lr.apply_work(kind, alloc)

    def _collect_active_jobs(self) -> List[Any]:
        # 最小実装：全ノードqueue + 稼働中ジョブを拾う
        jobs = []
        seen = set()
        # SimulationEngineは nodes 属性を持っている想定
        for node in self.engine.nodes.values():
            # nodeがqueueを持っているか確認
            if hasattr(node, "queue"):
                for job in node.queue:
                    if job.job_id not in seen:
                        jobs.append(job)
                        seen.add(job.job_id)
            # 稼働中ジョブ (WorkGateのbusy_jobs)
            if hasattr(node, "busy_jobs"):
                for job in getattr(node, "busy_jobs", []):
                    if job.job_id not in seen:
                        jobs.append(job)
                        seen.add(job.job_id)
        return jobs

    def _target_gate_for(self, job: Any) -> str:
        # 最小実装：job履歴や現在ノードから推定
        # ここは後で「案件がどのDR狙いか」をJobに持たせると綺麗
        # とりあえず現在のノード名から推測するか、デフォルトを返す
        if job.current_node:
            if "DR1" in job.current_node: return "DR1"
            if "DR2" in job.current_node: return "DR2"
            if "DR3" in job.current_node: return "DR3"
        
        # 履歴から判断（もしあれば）
        visited = {h['node_id'] for h in job.history}
        if "DR2" in visited: return "DR3"
        if "DR1" in visited: return "DR2"
        
        return "DR1"
