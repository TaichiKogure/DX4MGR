# DX4MGR Ver13 計算数式ガイド

## 1. フロー構成 (DR3回)
```mermaid
flowchart LR
    A[案件の到着] --> B[SMALL_EXP 小実験]
    B --> C[BUNDLE_SMALL まとめ]
    C --> D[DR1 第1定期判定]
    D -->|GO| E[MID_EXP 中実験]
    D -->|CONDITIONAL/NOGO| B

    E --> F[BUNDLE_MID まとめ]
    F --> G[DR2 試作ライン検証]
    G -->|GO| H[FIN_EXP 最終実験]
    G -->|CONDITIONAL/NOGO| E

    H --> I[BUNDLE_FIN まとめ]
    I --> J[DR3 商品化判定]
    J -->|GO| K[ミッション完了]
    J -->|CONDITIONAL/NOGO| H
```

- 差し戻し先は「直前の実験工程」。
- BUNDLEは「一定件数がたまったら束ねる」処理です。

## 2. 離散イベントシミュレーションの仕組み
Ver13はイベント駆動型のシミュレーションです。
主なイベント:
- ARRIVAL: ジョブがゲートに到着
- PROCESS_READY: すぐ処理可能な場合に作業開始
- WORK_COMPLETE: 作業完了
- MEETING_START: DR会議のタイミング
- SCHED_TICK: Schedulerの定期イベント

## 3. 到着プロセス (Poisson)
到着間隔は指数分布に従います。

- 到着率: `arrival_rate` [件/日]
- 到着間隔: Δt ~ Exp(1 / arrival_rate)

例: arrival_rate=0.5 -> 平均2日に1件。

## 4. 作業時間 (WorkGate)
作業時間は指数分布 + 摩擦係数で伸びます。

- 基本時間: T_base ~ Exp(mean = duration)
- 摩擦係数: M_friction
- 実効時間: T_eff = T_base * M_friction

摩擦モデル:
- linear: M = 1 + friction_alpha * (n_servers - 1)
- pairs:  M = 1 + friction_alpha * n_servers * (n_servers - 1) / 2
- その他: M = 1

注意: n_serversを増やすと速くなる一方、摩擦で逆効果になる場合があります。

## 5. バンドル処理 (BundleGate)
- キュー長が bundle_size に達すると処理が発火。
- bundle_size は定数 (分布ではなく固定値)。
- バンドル後は1つのJobとしてDRに進む。

Ver13追加: BundleJobはLatentRiskを合成して保持します。
- uncertainty / latent_risk / scale_factor は max
- evidence は平均

## 6. DRカレンダー (DRCalendar)
- 各DRの固定開催時刻を `schedule_by_gate` に持つ
- 次回時刻: next_after(gate_id, now)
- Ver13では `dr*_period` から定期カレンダーを生成

## 7. Scheduler と WorkPackage
Schedulerは日次で「締切までの残り」を見て工数を配分し、WorkPackage(計画)を生成します。

- 1日の総工数 = engineer_pool_size * hours_per_day_per_engineer
- 緊急度: urgency = 1 / max(slack, 0.25)
- 割当工数: alloc = min(total_hours_today, 2 + 6 * urgency)

WorkPackage生成後に LatentRisk を更新します。

## 8. LatentRisk の更新
工数割当に応じて evidence / uncertainty / latent_risk が更新されます。

- gain = 1 - exp(-k * effort_hours)
- evidence += gain
- uncertainty *= (1 - 0.5 * gain)
- latent_risk *= (1 - 0.6 * (1 - exp(-risk_k * effort_hours)))
- uncertainty / latent_risk は下限 0.02 でクリップ

## 9. DR会議の容量と品質 (MeetingGate)
### 9.1 容量
承認者の構成から実効容量を計算します。

- capacity = Σ(人数_i * capacity_i)

承認者タイプ:
- Senior: capacity=7, quality=0.76
- Coordinator: capacity=3, quality=0.70
- New: capacity=1, quality=0.40

### 9.2 品質 (GOの確率)
容量を重みとした加重平均で品質を算出します。

- quality = Σ(quality_i * capacity_i) / capacity
- `dr_quality` が指定されている場合はこの値を上書きします。
- 品質⇔速度トレードオフ: `dr_quality` が指定されている場合、DR容量に補正を掛けます。
  - speed_mult = clamp(1 + alpha * (0.8 - dr_quality), 0.5, 1.5)
  - capacity = round(capacity * speed_mult)
  - alpha = dr_quality_speed_alpha (既定1.0)

LatentRisk補正:
- q = quality * quality_mult - nogo_add
- cond_ratio = conditional_prob_ratio + conditional_add

### 9.3 判定ロジック
乱数 u ~ Uniform(0,1)
- GO: u < q
- CONDITIONAL: q <= u < q + (1 - q) * cond_ratio
- NOGO: それ以外

判定後の遷移は decision_latency_days の遅延を持ちます。

## 10. 差し戻し (Rework) の数式
CONDITIONAL時のみ差し戻しを発生させます。

- rework_count を +1
- 重み: w = Beta(a, b) * decay^(rework_count - 1)
  - a = rework_beta_a
  - b = rework_beta_b
- 生成タスク数: n_new = ceil(rework_load_factor * w)
- タイプ配分: (p_small, p_mid, p_fin) = normalize(rework_task_type_mix)
- 生成内訳: (n_small, n_mid, n_fin) = allocate(n_new, p_small, p_mid, p_fin)
- 再投入方式:
  - rework_reinject_mode = all: n_reinject = n_new
  - rework_reinject_mode = ratio: n_reinject = round(n_new * rework_reinject_ratio)
  - 既定は rework_reinject_mode = all
- 再投入内訳: (r_small, r_mid, r_fin) = allocate(n_reinject, n_small, n_mid, n_fin)

DR2だけは `dr2_rework_multiplier` が掛かります。

注意:
- rework_count > max_rework_cycles の場合、新規タスクは追加しません。
- 再投入分は新規ジョブとして SMALL/MID/FIN へ戻り、処理負荷に反映されます。

## 11. WIP計算
- 各時点の WIP = (キュー内 + 処理中) の合計
- sampling_interval 間隔でサンプルされます (既定は1日)

## 12. 主要指標の定義
- Completed count = 完了ジョブ数
- Throughput = Completed / days
- Lead time = 完了時刻 - 作成時刻
- P50/P90/P95 = リードタイム分位点
- Avg reworks = 平均差し戻し回数
- Avg WIP = WIPの平均

## 13. 品質ゲート判定 (Quality Gates)
- Gate1: 完了ジョブ数下限 (min_completed)
- Gate2: 95% CI幅 / 平均TP < max_ci_width
- Gate3: P90待ち時間 < max_wait_p90
- Gate4: 平均差し戻し回数 < max_reworks

PASSが揃えば overall PASS となります。
