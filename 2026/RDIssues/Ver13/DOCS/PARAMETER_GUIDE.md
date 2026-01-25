# DX4MGR Ver13 パラメーターガイド (簡易)

このファイルは概要のみです。詳細は `PARAMETER_GUIDE_Ver13.md` と `MATH_GUIDE_Ver13.md` を参照してください。

- `bundle_size_*`: バンドルのサイズ。小さいほど待ちが減るがDR負荷は増える
- `dr*_period`: DR会議の間隔。Ver13ではカレンダー生成に使う
- `dr*_capacity`: DR会議で処理できる件数
- `n_servers_mid` / `n_servers_fin`: Mid/Finの並列数
- `n_senior` / `n_coordinator` / `n_new`: 承認者の構成 (品質と容量に影響)
- `conditional_prob_ratio` / `decision_latency_days`: DR判定の挙動
- `rework_*`: 差し戻しの強さと増殖
- `friction_model` / `friction_alpha`: 人数増加の摩擦
- `engineer_pool_size` / `hours_per_day_per_engineer`: 計画工数の総量 (LatentRiskに影響)

未使用/予約:
- `dr_quality` は現状未使用 (承認者構成で品質を計算)
- `rework_task_type_mix` は未使用
