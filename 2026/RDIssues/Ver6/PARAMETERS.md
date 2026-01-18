# DX4MGR Sim Ver6 パラメータ定義 (Agent + Gate Graph)

## 1. 共通設定 (Simulation)
- `days`: シミュレーション期間 (日)
- `n_trials`: モンテカルロ試行回数
- `base_seed`: 乱数シードのベース値

## 2. 流入 (Arrival)
- `arrival_rate`: 1日あたりの案件流入数 (Job/day)

## 3. 作業ゲート (WorkGate)
- `small_exp_duration`: 小実験の平均処理日数 (指数分布)
- `proto_duration`: 試作の平均処理日数 (指数分布)
- `n_servers`: 並列処理可能なサーバー数 (試作ゲートなどで制限)

## 4. バンドルゲート (BundleGate)
- `bundle_size`: 何件の試作を束ねて1つのDRパッケージにするか

## 5. 会議ゲート (MeetingGate / DR)
- `dr_period`: 会議の開催周期 (日)
- `dr_capacity`: 1回の会議で審議可能な最大件数
- `approvers`: 承認者の構成 (Senior, Coordinator, New)
- `dr_quality`: 会議の意思決定品質 (成功確率)

## 6. 差し戻し・増殖 (Rework Proliferation)
- `p_rework`: 差し戻し発生確率 (1.0 - dr_quality)
- `rework_load_factor`: 差し戻し1件あたりの小実験増殖係数
- `max_rework_cycles`: 最大ループ回数 (無限ループ防止)
- `decay`: 繰り返し差し戻し時の負荷減衰率

## 7. 指標 (Metrics)
- `Throughput`: 期間内に完了したDRパッケージ数 / 期間
- `Lead Time P90`: 案件の90パーセンタイル・リードタイム (尾の長さ)
- `CCDF`: 相補累積分布関数 (リスクの可視化)
- `Gate Wait Time`: 各ゲートでの平均滞留時間
