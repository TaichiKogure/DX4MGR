Plan7 GUI改良（初期投入JOB数の入力対応とJSONパラメータ全面対応）レポート

作成日: 2026-03-29

## 要約
- Plan7 のGUIに「初期投入JOB数（num_initial_jobs）」等の不足パラメータを追加し、期間内完了件数（Completed Jobs）の感度をGUI操作だけで出せるようにしました。
- さらに「高度なJSON（Advanced JSON）」編集を各シナリオ（A/B/C）に追加。JSONで指定可能だった全パラメータをGUIからも直接上書き・指定できるように拡張しました（GUI値よりJSONが優先）。

## 変更点（VerUP: GUI v2.3 相当）
1. 新規GUI入力項目の追加（テーブル行）
   - sampling_interval（サンプリング間隔）
   - dr2_period / dr3_period（DR2/DR3 会議周期）
   - num_initial_jobs（初期投入JOB数）
   - start_node_id（開始ノードID）
   - job_arrival_low / job_arrival_high（初期到着分布の一様分布Low/High）
2. Advanced JSON（高度なJSON）エディタを各シナリオに追加
   - 任意のパラメータをJSONで上書き可能。
   - GUIテーブルで未提供/複雑な構造のパラメータも編集可能（例: approvers設定、departments/handoffs配列、rework_policy、flow 等）。
   - JSON構文チェック（バリデーション）ボタンを実装。
3. JSONの保存/読込の拡張
   - 「設定を保存（JSON）」で、マージ後パラメータに加えて `advanced_json`（生文字列）も埋め込んで保存。
   - 「設定を読込（JSON）」で `advanced_json` をGUIに復元し、テーブル既知キーは自動反映。

## 対応コード
- 2026/DataAnalysis/Plan7/gui_sim.py
  - PARAM_DEFINITIONS に新規パラメータを追加
  - 高度なJSONエディタ（_open_json_editor）を実装し、行末に編集ボタンを配置
  - get_parsed_params で新規キーの型変換・job_arrival_dist の自動生成・Advanced JSONの安全マージ（JSON優先）を追加
  - save_params_json / load_params_json を拡張し、advanced_json の保存/復元に対応

## 使い方（GUI）
1. 初期投入JOB数の設定
   - 左ペイン「初期投入JOB数 / Initial Jobs」で任意の件数を指定。
   - 例えば A/B/C で 3 / 10 / 30 と設定し、同条件で `steps` を短くすると `Completed Jobs` の感度がGUI比較で明瞭になります。
2. 初期到着分布の調整
   - `job_arrival_low` と `job_arrival_high` を設定すると、初期JOBの到着タイミングを一様分布でばらします。
3. さらなる詳細調整（Advanced JSON）
   - 「高度なJSON / Advanced JSON」行の「編集…（A/B/C）」ボタンから、当該シナリオのJSONを編集。
   - 構文チェックでエラーがなければOKで確定。GUIテーブルの値に上書きマージされ、同一キーはJSON側が優先。

### Advanced JSON で指定可能な主な項目（例）
- approvers設定（DR1/DR2/DR3）
  - `dr1_approvers`, `dr2_approvers`, `dr3_approvers`
- 部署/ハンドオフ（詳細）
  - `departments`（id/name/calendar/cost_factor/sla 等）
  - `handoffs`（from/to, q_if, info_loss_lambda, transfer_time_dist 等）
- リワーク/会議/フロー
  - `rework_policy`（rework_load_factor/max_rework_cycles/decay）
  - `cross_meetings`（departments/interval/threshold/logic）
  - `flow`（ノード列の完全上書き: work/meeting/thresholds/period_days/next_node_id 等）
- その他
  - `job_arrival_dist`（任意の分布に差し替え: exponential/triangular/constant…）
  - `teams`, `tech_items`（詳細なスキーマでの指定）

## 互換性と注意
- 既存のJSON設定ファイルは従来どおり読込可能です。今回の拡張で `advanced_json` キーが追加保存されますが、これを無視しても従来処理に影響はありません。
- Advanced JSON は GUIテーブル値よりも優先されます。重複キーにご注意ください。
- 不正なJSON（配列トップ等）を保存しようとするとエラー表示の上、反映されません。

## 簡易検証
- 単体起動: `python 2026/DataAnalysis/Plan7/gui_sim.py`
  - A/B/C それぞれで `num_initial_jobs=3/10/30`、`steps=120`、`dr1_period=14` 等を設定し実行。
  - KPI/メトリクス比較で `Completed Jobs` と `throughput/lead_time` の差異がGUI上で確認できます。
  - Advanced JSON に `{"dr2_period": 21, "dr3_period": 42}` を入力して反映確認。

## 今後の拡張候補
- Advanced JSON のプレビュー要約（差分表示）
- approvers/departments/handoffs のGUIフォーム化（配列エディタ）
- `flow` のGUI可視化・編集（ドラフト）

---
本レポートは、Plan7の「GUIから全変数指定可能」への改善に関する実装変更点と利用手順をまとめたものです。詳細は `gui_sim.py` の該当実装をご参照ください。
