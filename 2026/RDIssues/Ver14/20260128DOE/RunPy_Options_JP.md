# run.py 実行時に指定できるコマンド一覧（Ver14）

`2026/RDIssues/Ver14/run.py` の CLI オプション一覧です。

## 使用可能なオプション
- `--scenarios`
  - 説明: シナリオCSVのパスを直接指定
  - 例: `--scenarios /path/to/scenarios.csv`

- `--scenarios-dir`
  - 説明: シナリオCSVが置かれたディレクトリを指定
  - 例: `--scenarios-dir /path/to/dir`

- `--scenarios-file`
  - 説明: シナリオCSVのファイル名を指定（既定: `scenarios.csv`）
  - 例: `--scenarios-file scenarios_custom.csv`

- `--out`
  - 説明: 出力ディレクトリを指定（既定: `2026/RDIssues/Ver14/output`）
  - 例: `--out /path/to/output_dir`

## 解決ルール（優先順位）
1. `--scenarios` が指定されていれば、それを最優先で使用
2. `--scenarios-dir` が指定されていれば、`--scenarios-file`（未指定なら `scenarios.csv`）と結合
3. それ以外は `--scenarios-file` を `run.py` のディレクトリ基準で解決

## 実行例
```bash
# 絶対パスでCSV指定
python run.py --scenarios /Users/.../scenarios.csv --out /Users/.../output

# ディレクトリ＋ファイル名指定
python run.py --scenarios-dir /Users/.../configs --scenarios-file my_scenarios.csv

# ファイル名だけ指定（run.py と同じディレクトリ基準）
python run.py --scenarios-file scenarios_test.csv
```
