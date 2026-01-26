# DX4MGR Ver13 パラメーターガイド (簡易)

このファイルは概要のみです。詳細は `PARAMETER_GUIDE_Ver13.md` と `MATH_GUIDE_Ver13.md` を参照してください。

- `bundle_size_*`: バンドルのサイズ。小さいほど待ちが減るがDR負荷は増える
- `dr*_period`: DR会議の間隔。Ver13ではカレンダー生成に使う
- `dr*_capacity`: DR会議で処理できる件数
- `dr_quality`: DR品質(0..1)と速度トレードオフ (空/NaNなら承認者構成)
- `dr_quality_speed_alpha`: トレードオフの強さ (既定1.0)
- `n_servers_mid` / `n_servers_fin`: Mid/Finの並列数
- `n_senior` / `n_coordinator` / `n_new`: 承認者の構成 (品質と容量に影響)
- `conditional_prob_ratio` / `decision_latency_days`: DR判定の挙動
- `rework_*`: 差し戻しの強さと増殖
- `rework_task_type_mix`: 追加タスクのタイプ配分 (SMALL/MID/FIN)
- `rework_reinject_mode`: 再投入の方式 (`all` or `ratio`)
- `rework_reinject_ratio`: 追加タスクの再投入比率 (`ratio` 時のみ)
- `friction_model` / `friction_alpha`: 人数増加の摩擦
- `engineer_pool_size` / `hours_per_day_per_engineer`: 計画工数の総量 (LatentRiskに影響)

未使用/予約:
なし
