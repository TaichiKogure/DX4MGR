# DX4MGR Ver12 初心者ガイド

Ver12は、DRを3段に固定した流れで動くモデルです。
シンプルに書くと次の順で進みます。

**フロー概要**
- Small実験 -> DR1
- DR1 GO -> Mid実験 -> バンドル -> DR2
- DR2 GO -> Fin実験 -> バンドル -> DR3
- DR3 GO -> ミッション完了

**差し戻しルール**
- DR1 で NOGO/CONDITIONAL -> Small実験に戻る
- DR2 で NOGO/CONDITIONAL -> Mid実験に戻る
- DR3 で NOGO/CONDITIONAL -> Fin実験に戻る

## 1. シナリオを編集する
`scenarios.csv` にパラメータを書きます。
詳細は `PARAMETER_GUIDE_Ver12.md` を参照してください。

## 2. 実行
```bash
python3 2026/RDIssues/Ver12/run.py
```

## 3. 出力先
`output/` に結果が出力されます。

- **Quality Gate判定一覧(CSV)**: `quality_gate_summary.csv`
- **DR?????(CSV)**: `dr_cost_summary.csv`
- **シナリオ比較**: `comparison_throughput.png`
- **待ち時間分布**: `compare_violin.png`
- **WIP推移**: `wip_time_series.png`
- **ゲート別待ちヒートマップ**: `gate_wait_heatmap.png`
- **フロー時間内訳(平均)**: `flow_time_breakdown.png`
- **フロー時間内訳(CSV)**: `flow_time_breakdown.csv`
- **Job進行（サンプルガント）**: `job_gantt_...png`
- **ゲート別待ち時間（時間帯ヒートマップ）**: `job_wait_heatmap_...png`
- **ゲート別待ち時間分布**: `job_wait_dist_...png`

## 4. DOEは別スクリプト（任意）
探索用は `run_doe.py` を使います。

```bash
python3 2026/RDIssues/Ver12/run_doe.py
```
`output_doe/` に `doe_results_scored.csv` と `scenarios_from_doe.csv` が出力されます。
## ??????

- **bundle_size_small ?????**
  Small???1??????????????????????
  ????? Mid/Fin ???????????

