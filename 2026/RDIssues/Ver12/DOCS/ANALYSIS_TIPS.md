# DX4MGR Ver12 分析メモ

## クイック確認
- `quality_gate_summary.csv` でシナリオ判定を一覧
- `comparison_throughput.png` でスループット比較
- `job_gantt_...png` でジョブの待ちと処理を直感的に確認

## よくある課題
- **完了ジョブ数が少ない**: `days` を増やす / `arrival_rate` を上げる
- **DRの混雑**: `dr1_period`～`dr3_period` を短くする
- **Mid/Finが混む**: `n_servers_mid` / `n_servers_fin` を見直す
- **DR1????**: `bundle_size_small` ?????
