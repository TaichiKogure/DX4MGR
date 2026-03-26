# Plan7 パラメータ感度 DOE レポート

実行時刻: 2026-03-26 06:15:36

## 実験設定
- steps: 30
- seeds/point: 1
- 走査パラメータ: dr1_period, proto_n_servers, res_n_servers

## 中間データの要約（平均値）
- throughput: 0.0000
- lead_time_p50: 0.0000
- lead_time_p90: 0.0000
- avg_wip: 0.0000
- loss_time_per_primary: 0.0000

## 代表的な図
![plot](plots\main_effects_dr1_period.png)
![plot](plots\main_effects_proto_n_servers.png)
![plot](plots\main_effects_res_n_servers.png)
![plot](plots\heatmap_proto_n_servers_vs_dr1_period.png)

## データファイル
- runs.csv: 各試行の生データ
- summary_main_effects.csv: パラメータ毎の主効果