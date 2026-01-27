# Ver13 スコアカード拡張 変更レポート

## 目的
Ver13のシナリオ性能スコアカードを最新システム（Scheduler/LatentRisk/DRCalendar）に合わせ、
損失分解（時間ロス・コストロス）とDRゲート突破所要時間を詳細に出力できるように拡張した。

## 定義（本レポートで使用する指標）
- **DR突破所要時間**: `ENQUEUE -> DECISION` の経過時間
- **差し戻しロス（時間）**: 再作業ジョブのみを対象に、`created_at -> 完了` の全経過時間
- **時間ロス（1件あたり）**: 
  - `待ち時間（primary） + 意思決定遅延（primary） + 差し戻しロス（rework）/ primary件数`
- **コストロス（1件あたり）**:
  - 再作業ジョブに紐づくレビューコスト合計 / primary件数

## 追加/拡張された出力
- `loss_breakdown.csv`
  - シナリオ単位の時間ロス分解（待ち/意思決定/差し戻し）、コストロスを出力
- `rework_loss_summary.csv`
  - 再作業ジョブのみの平均ロス（時間・コスト）を出力
- `rework_source_gate_summary.csv`
  - 差し戻し発生源（DR1/DR2/DR3）別のロス内訳を出力
- `dr_gate_cycle_times.csv`
  - DR1/DR2/DR3の突破所要時間（P50/P90/P95）、待ち時間、判定率を出力
- `loss_time_breakdown.png`
  - 1件あたり時間ロスの分解（待ち/意思決定/差し戻し）
- `loss_cost_breakdown.png`
  - 1件あたり差し戻しコストロス
- `dr_gate_cycle_p90_heatmap.png`
  - DR突破所要時間(P90)のヒートマップ

## スコアカード拡張
- 既存の `LT(P90) / TP / AvgWIP / Rework` に加えて、
  - **TimeLoss**, **CostLoss**
  - **DR1/DR2/DR3(P90)**
  を追加。

## 実装メモ
- MeetingGateに `DECISION` イベントとレビューコストの履歴追加。
- `analysis/metrics.py` で損失分解とDR突破時間を算出。
- `visualizer.py` にロス分解とDRヒートマップの描画関数を追加。

