# DX4MGR Ver9: 標準フロー・WIP可視化強化モデル

## 概要
Ver9（Ver8強化版）では、シミュレーションのメインフローを `runner/adapters.py` の標準フロー（SMALL_EXP → PROTO → BUNDLE → DR）に差し替え、ゲート別の「詰まり（待ち時間・WIP）」を強力に可視化する機能を導入しました。
これにより、開発プロセスのどこでボトルネックが発生しているかを直感的に把握することが可能になりました。

## 主な変更点
1. **シミュレーション本体の刷新**: `runner/adapters.py` を利用した `simulate_standard_flow_v8()` をメインに採用。
2. **WIPサンプリングの実装**: `core/engine.py` で WIP を一定間隔でサンプリングし、ノード別の滞留を記録。
3. **メトリクス強化**: `analysis/metrics.py` にノード別平均WIPの集計ロジックを追加。
4. **強力な可視化**: `visualizer_v8.py` に「ゲート別平均待ち時間」と「ゲート別平均WIP」のヒートマップを追加。
5. **パイプライン接続**: `run_v8.py` を標準フロー版に切り替え、DOE探索からモンテカルロ実行、統計解析までを一貫して実行。

## ファイル構成
- `run_v8.py`: メイン実行スクリプト（標準フロー版）。
- `simulator_v8.py`: 標準フロー対応シミュレーションロジック。
- `analyzer_v8.py`: 統計解析・検証ゲート判定。
- `visualizer_v8.py`: ヒートマップを含む高度な可視化。
- `scenarios.csv`: 標準フロー用の解析シナリオ設定。

## 実行方法
```bash
python3 2026/RDIssues/Ver9/run_v8.py
```

## 解析結果の見方
- `v8_step1_doe_analysis.png`: パラメータ感度分析。
- `v8_step4_gate_wait_heatmap.png`: どのゲートで「待ち」が発生しているかを表示。
- `v8_step4_gate_wip_heatmap.png`: どのゲートに「仕掛品(WIP)」が溜まっているかを表示。
