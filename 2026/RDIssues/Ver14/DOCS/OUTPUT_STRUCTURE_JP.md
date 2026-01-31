# Ver14 シミュレーション出力構造ルール

`run.py` の実行により生成される出力ディレクトリの構造とルールについて説明します。

## 1. 基本的な出力構造

出力ディレクトリ（既定: `2026/RDIssues/Ver14/output`）配下は、指定されたシナリオCSVファイルの構成に基づいて自動的に整理されます。

### A. 単一のCSVファイルを指定した場合 (`--scenarios`)
```
output/
  ├── Output-(シナリオ名1)/
  │     ├── job_details_(シナリオ名1).csv
  │     ├── job_gantt_(シナリオ名1).png
  │     └── ... (個別プロット)
  ├── Output-(シナリオ名2)/
  ├── ...
  ├── comparison_throughput.png (全体比較)
  ├── scenario_scorecard.png
  └── final_analysis_report.json
```

### B. ディレクトリを指定した場合 (`--scenarios-dir`)
各CSVファイル名に基づいたディレクトリが作成され、その中でシナリオが管理されます。
```
output/
  ├── (CSVファイル名A)/
  │     ├── Output-(シナリオ名1)/
  │     ├── Output-(シナリオ名2)/
  │     ├── flow_time_breakdown.csv
  │     └── ... (CSV A内の全シナリオ比較結果)
  └── (CSVファイル名B)/
        ├── Output-(シナリオ名X)/
        └── ...
```

## 2. 出力ファイルの詳細

### 個別シナリオディレクトリ (`Output-xxx/`)
各シナリオの計算結果が格納されます。
- `job_details_xxx.csv`: 全完了ジョブのイベントログ
- `job_gantt_xxx.png`: ジョブの進行状況（ガントチャート、最大30ジョブ）
- `job_wait_heatmap_xxx.png`: 時系列・ゲート別の待ち時間ヒートマップ
- `job_wait_dist_xxx.png`: ゲート別の待ち時間分布

### CSVサマリー（親ディレクトリ）
CSV内の全シナリオを横断的に解析した結果が格納されます。
- `comparison_throughput.png`: スループットの信頼区間付き比較図
- `scenario_scorecard.png`: 性能スコアカード
- `flow_time_breakdown.png/csv`: 工程別・シナリオ別の所要時間内訳
- `final_analysis_report.json`: 詳細な統計数値を含む統合レポート

## 3. 実行オプションの挙動

- `--scenarios-dir`: 指定されたディレクトリ内の全ての `.csv` ファイルを読み込み、順次計算を実行します。
- 各CSVファイルの計算が完了するたびに、そのCSV用のサマリーレポートが生成されます。
- 同一CSV内に `Baseline` という文字列を含むシナリオがある場合、それを比較の基準（Baseline）として自動認識します。存在しない場合は1番目のシナリオを基準とします。
