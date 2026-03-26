# Plan7 改善プラン Ver2（GUI + 計算ロジック）

最終更新: 2026-03-25

---

## 1. 目的とスコープ
- 目的: Plan7 を「可視化の表現力」と「現実の組織に近い計算ロジック」の両面で強化し、意思決定の質と説明力を高める。
- スコープ:
  - GUI: 3D可視化、時間推移の動画生成、対話的再生（スクラブ・速度変更）、比較シナリオの一括エクスポート。
  - 計算ロジック: 複数部署（Department）を跨いだ連携・ハンドオフ・会議・リソース共有を具体的に表現できる拡張。
- 非スコープ（Ver2では扱わない）: Web配信/Dash化、SaaS連携、実データ自動同化（Digital Twin 拡張は現状維持）。

---

## 2. 背景と現状課題（要約）
- 現状GUIは2D中心で、時系列のダイナミクス（混雑の波、WIPの立ち上がり/崩れ、技術成熟の拡散）を直感的に把握しづらい。
- 計算ロジックはチーム/ゲート/会議でR&Dの基本構造は表現できるが、「複数部署の連携」や「部署境界での情報劣化・調整遅延・同期の必要性」が十分にパラメタライズされていない。

---

## 3. 改善方針（Ver2 概要）
- GUI面:
  - 3D表現を導入（例: 時間×工程×WIPの3Dサーフェス、混雑のヒートボリューム、ネットワーク立体図）。
  - シミュレーション推移を動画として自動生成（mp4/webm/gif）。比較シナリオの動画を並列生成可能にする。
  - GUIに「再生コントロール（Play/Pause/スクラブ/速度）」を追加し、実行後に時系列を追体験できるようにする。
- 計算ロジック面:
  - Department 概念を導入し、Team と Gate/Meeting を部署に紐付け。
  - 複数部署を跨ぐ WorkItem（AND/OR 並列・直列・合流）と Handoff（情報移送）をモデル化。
  - 部署間会議（Cross-Department Meeting）や SLA/カレンダー差異、知識移転ロス、再作業の波及をパラメタ化。

---

## 4. GUI強化の詳細設計
### 4.1 可視化コンポーネント候補
- 3D可視化ライブラリ候補:
  - Plotly（推奨）: 3Dサーフェス/散布/ネットワークが容易。インタラクティブ（回転/ズーム）が標準。HTML保存に強い。
  - Matplotlib mplot3d: 既存資産との親和性が高く、動画生成も容易。インタラクションは限定的。
  - PyQtGraph(OpenGL): 高速・軽量。ネイティブGUI一体化に有利だが、実装コストや依存が増える。
- 方針: レポート出力はまず Matplotlib で静的/動画を安定化 → 余力があれば Plotly でHTML/交互操作を追加。

### 4.2 3Dグラフ案（例）
- 3D WIPサーフェス: 横軸=時間、奥行=工程/部署、縦軸=WIP量（Z）。混雑の山谷を直感表示。
- 3D リードタイム分解: 時間×部署で平均滞留をサーフェス化（待ち/作業/会議で色分け）。
- 3D 技術成熟の拡散: 技術の成熟度/不確実性の分布変化を3D散布/ボリュームで可視化。
- 立体ネットワーク図: 部署—部署のハンドオフ網を3Dで描画（辺の太さ=流量、色=欠陥率）。

### 4.3 動画生成・書き出し
- 方式:
  - Matplotlib.animation.FuncAnimation + ffmpeg で mp4 を生成。
  - 軽量な共有用に gif/webm も選択可能（フレーム間引き設定）。
- 出力先・命名規則:
  - ディレクトリ: `reports/viz/videos/`
  - 例: `Scenario_A_wip_3d_t{start}-{end}_fps{fps}.mp4`
- GUI操作:
  - 「詳細レポート出力」ダイアログに「動画も生成（3D/2D）」チェックを追加（デフォルトOFF）。
  - 生成進捗をプログレスバーで表示。失敗時はログを出して他シナリオの生成は続行。

### 4.4 再生コントロール（GUI）
- 実装イメージ（`gui_sim.py`）:
  - 新タブ「リプレイ」を追加。直近の `wip_history`, `util_history`, `tech_snapshots` を読み込み、時刻スライダーでフレームを切替。
  - 再生速度（0.25x, 0.5x, 1x, 2x, 4x）、一時停止、先頭/末尾ジャンプ、スクラブ中は一時的に解像度を落としてスムーズに。
- バッチ/比較モード:
  - リストから対象シナリオを切替。並列再生はVer2では非対応（性能確保のため）。

### 4.5 パフォーマンス/依存
- 大規模フレームのレンダリングは重い。対策:
  - フレーム間引き（例: 60s→5s刻み）、メッシュ解像度の自動スケーリング、GPU非依存な描画。
  - 長尺はmp4優先、gifは最大30秒/10fpsなどの上限をガード。
- 依存関係:
  - `matplotlib`, `ffmpeg`（外部バイナリ）, （任意で）`plotly`。
  - リポジトリには大容量動画をコミットしない運用（`reports/`出力のみ）。

---

## 5. 計算ロジック強化（複数部署連携）
### 5.1 データモデル拡張
- 追加エンティティ:
  - `Department`: 部署ID, 名称, 稼働カレンダー, コスト係数, SLA（平均応答/最大待機）
  - `Team`: 所属Department, スキル/能力, 人員, 稼働率上限
  - `Handoff`: fromDept→toDept, インターフェース品質q_if, 情報損失率λ, 待機/転送時間分布
  - `CrossDeptMeeting`: 参加部署集合, 開催間隔, 閾値, 意思決定ロジック（GO/COND/NO_GO）
  - `WorkItem`: 必要部署シーケンス/並列要件（AND/OR）, 各工程の責任部署, リワーク規則

### 5.2 イベント/ロジック拡張（`core/engine.py`）
- 新イベント種別（例）:
  - `HANDOFF_START` / `HANDOFF_COMPLETE`: 部署間の引継ぎ。転送遅延/情報損失を適用。
  - `SYNC_REQUIRE` / `SYNC_MEETING`: 複数部署の同期ポイント。必要数が揃った時点で開始。
  - `MULTI_RESOURCE_ALLOC` / `MULTI_RESOURCE_RELEASE`: 複部署/複チーム同時アロケーション。
- WorkItem進行:
  - 並列工程（AND）: すべて完了後に合流（マージコスト適用）。
  - 代替工程（OR）: いずれか一方の成功で通過。選択は期待値/混雑/スキルで動的決定可能。
  - Handoff の情報劣化が一定以上で再作業（Rework）確率を上昇。

### 5.3 メトリクス拡張（`analysis/metrics.py`）
- 追加指標:
  - 部署別リードタイム（作業/待機/判定/転送の内訳）
  - Handoff 間の欠陥率と再作業伝播率
  - 部署間同期待ち時間・頻度、会議による遅延・決定品質
  - 部署別/全体の稼働率・ボトルネック寄与度（貢献分析）
  - フロー可視化用の流量マトリクス（Sankey/ネットワーク図入力）

### 5.4 コンフィグ拡張（`core/scenario.py`）
- JSON/YAML 例（抜粋）:
```json
{
  "departments": [
    {"id": "RD", "name": "研究", "calendar": "5x8", "sla": {"p95_wait_max": 120}},
    {"id": "DV", "name": "開発", "calendar": "5x10"},
    {"id": "MF", "name": "製造", "calendar": "7x24"}
  ],
  "teams": [
    {"id": "chemA", "dept": "RD", "skill": ["chem"], "members": 4},
    {"id": "proto1", "dept": "DV", "skill": ["proto"], "members": 3},
    {"id": "lineX", "dept": "MF", "skill": ["mass"], "members": 6}
  ],
  "handoffs": [
    {"from": "RD", "to": "DV", "q_if": 0.85, "lambda_loss": 0.1, "delay_mean": 12},
    {"from": "DV", "to": "MF", "q_if": 0.9, "lambda_loss": 0.05, "delay_mean": 24}
  ],
  "cross_meetings": [
    {"id": "DR2", "participants": ["RD", "DV"], "interval": 40, "rule": "majority_go"}
  ],
  "workflows": [
    {"id": "phase2", "sequence": [
      {"op": "exp", "dept": "RD"},
      {"op": "proto", "dept": "DV"},
      {"op": "ramp", "dept": "MF"}
    ]}
  ]
}
```

### 5.5 既存互換性
- 既存シナリオは `departments` 未指定で従来どおり単一部署モードとして解釈。
- 既存メトリクスは維持しつつ、新指標は追加列/別セクションで提供。GUIでもタブ分離。

---

## 6. 実装ロードマップ（推奨）
### フェーズ1: 下回り整備（2–3週間）
- データモデル追加（Department/Handoff/CrossDeptMeeting）と既存ロジックへの最小侵襲での導入。
- 3D WIPサーフェスのCLI出力（静止画/短尺動画）試作。`reports/viz/videos/` へ保存。
- メトリクスに部署ID次元を追加（集計の疎行列対応）。
- 単体/比較テストを追加。性能回帰をチェック。

### フェーズ2: GUI統合（2–4週間）
- 「リプレイ」タブ追加、動画出力オプション、進捗表示。
- 「部署」タブ（新規）に部署別KPI/ボトルネック表示、Sankeyやネットワーク図の静的出力連携。
- 比較シナリオの一括動画生成と命名規則整備。

### フェーズ3: 高度化（以降）
- 並列工程（AND/OR）と同期ポイントの本格実装、Cross-Dept Meetingのポリシー拡張。
- Plotlyベースのインタラクティブ3D/HTML出力（必要に応じてエクスポート）。
- DOE・感度分析と部署パラメータの連動（SLA/カレンダー/人員）を自動スイープ。

---

## 7. リスクと対策
- 依存性追加・環境差: ffmpeg の存在確認を行い、無ければ動画出力を自動的にスキップ（警告表示）。
- 性能劣化: フレーム間引き・メッシュ解像度縮小・描画キャッシュで軽量化。
- 複雑化: 既存互換モードを維持し、「部署拡張ON/OFF」フラグで段階導入。

---

## 8. 成果物と出力先
- 3D静止画: `reports/viz/Scenario_X_*_3d.png`
- 動画: `reports/viz/videos/Scenario_X_*.mp4`（デフォルト10–30秒）
- 部署拡張メトリクスCSV: `reports/doe/metrics_dept.csv`（比較やDOEで集計）
- ドキュメント: 本ファイル、GUIヘルプの追補、API差分リファレンス

---

## 9. 検証方針
- 最小構成シナリオ（2部署）でのE2E: 実行→3D静止画/動画生成→メトリクスに部署軸が出ること。
- 比較モード: 3シナリオ同時実行後の一括動画出力と命名規則の検証。
- 再現性: seed固定で動画フレームの整合性を確認（チェックサム/代表フレーム差分）。

---

## 10. 参考インターフェース（擬似コード）
```python
# core/scenario.py
@dataclass
class Department:
    id: str
    name: str
    calendar: str = "5x8"
    cost_factor: float = 1.0

@dataclass
class Handoff:
    from_dept: str
    to_dept: str
    q_if: float = 0.9
    lambda_loss: float = 0.05
    delay_mean: float = 8.0

@dataclass
class ScenarioParams:
    departments: list[Department] = field(default_factory=list)
    handoffs: list[Handoff] = field(default_factory=list)
    # 既存フィールドは維持

# core/engine.py (イベント拡張)
class EventType(Enum):
    ARRIVAL = auto()
    PROCESS_READY = auto()
    WORK_COMPLETE = auto()
    MEETING_START = auto()
    HANDOFF_START = auto()
    HANDOFF_COMPLETE = auto()
    SYNC_REQUIRE = auto()
    SYNC_MEETING = auto()

# analysis/metrics.py（集計拡張）
def calculate_metrics_by_dept(results) -> dict:
    """部署別のリードタイム/待機/転送/会議/再作業率などを返す"""
    ...

# gui_sim.py（動画出力の呼び出し例）
if opts.export_video:
    from analysis.viz3d import render_wip_surface_video
    render_wip_surface_video(sim.results.wip_history, outfile)
```

---

## 11. 導入ポリシー
- まずは「静的3D + 短尺動画 + 部署IDの導入」に絞って確実に安定化。
- 既存利用者の操作手順はほぼ不変（オプションを増やすのみ）。
- ドキュメント/README/サンプル設定を同時更新してトレーサビリティを確保。

---

このVer2プランに追加要望（例: KPIの定義拡張、部署固有のカレンダー制約の細分化、Plotlyの完全対応）があればコメントください。優先度に応じてフェーズ配分を再調整します。

---

## 12. UPDATE（実装差分・最小版）
本ドキュメントVer2の方針に基づき、以下の最小実装を追加しました（2026-03-26）。

- 追加エンティティ（最小データモデルの導入）:
  - Department: `dept_id`, `name`, `calendar(未使用スタブ)`, `cost_factor`, `sla={avg_response, max_wait}`
  - Team拡張: `department`, `utilization_cap` を保持
  - Handoff: `from_dept`, `to_dept`, `q_if`, `info_loss_lambda(λ)`, `transfer_time_dist`
  - CrossDeptMeeting: `departments`, `interval_days`, `threshold`, `logic(GO/COND/NO_GO)`（スタブ保持）
  - WorkItem: `steps(AND/OR表現を許容)`, `owners`, `rework_rules`（保持のみ）

- シミュレーション統合（軽量適用）:
  - 部門SLA（`max_wait`）超過を`SLA_VIOLATION`としてKPIに加算。
  - 部門ハンドオフ時に、`q_if/λ`でプロトコル遵守率を劣化、`transfer_time_dist`で処理時間に転送遅延を付与。
  - 部門コスト（簡易）として「処理時間×部門`cost_factor`」を`dept_cost_time`に集計。
  - 既存のフロー/ゲートは温存。フラグOFF時は従来挙動。

- GUIの拡張（`gui_sim.py`）:
  - パラメータテーブルにフラグを追加: 
    - 「部署を考慮」「ハンドオフを考慮」（既定ON）、「横断会議を考慮」（既定OFF）、「リワーク規則を考慮」（ON）。
  - 入力が無くても、ON時は最小の部署・ハンドオフ設定を自動生成（R/P/A/M）。
  - 「詳細レポート出力」で以下を追加保存:
    - `*_dept_network.png`: 部署/ネットワーク簡易図（最小、既存ネットワーク3D静止画を流用）
    - `*_dept_summary.csv`: `sla_violations`, `handoff_events`, `dept_cost_time` をKPIからサマリ

- 出力先ポリシー:
  - 画像: `2026/DataAnalysis/Plan7/reports/viz/`
  - CSV: 同上
  - 動画: `.../reports/viz/videos/`（ffmpeg未導入時はgifへ自動フォールバック）

- 既知の制約/今後の拡張:
  - CrossDeptMeeting/WorkItem は現状スタブ保持（意思決定・分岐の本格実装は次フェーズ）。
  - 部署別LT分解・Sankey等の詳細レポートは次フェーズで追加。
  - カレンダー（稼働日/時間）は未適用。コスト/待機・遅延・品質劣化に限定した導入。
