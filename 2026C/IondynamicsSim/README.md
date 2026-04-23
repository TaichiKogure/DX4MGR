# IondynamicsSim

リチウムイオン二次電池の充放電時におけるイオン輸送抵抗をグラフィカルに可視化・動画出力する小規模シミュレータ。

## インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# または、プロジェクトを編集可能モードでインストール（推奨）
pip install -e .
```

## 実行方法

### 1. GUIで実行（推奨・簡単）
プロジェクトルートにある `run_gui.bat` (Windows) または `run_gui.ps1` (PowerShell) を実行してください。
パラメータの入力、シミュレーションの実行、結果ファイル（CSVやグラフ）へのアクセスがGUI上で行えます。

### 2. スクリプトで実行
プロジェクトルートにある `run_default.bat` (Windows) または `run_default.ps1` (PowerShell) を実行すると、デフォルト設定でシミュレーションが走ります。

### 3. コマンドとして実行
インストール済みの場合、以下のコマンドが使えます。
```bash
# GUIの起動
iondynamics-gui

# CLIの実行
iondynamics run --config configs/default.yaml
```

### 3. python -m で実行
```bash
# PowerShell
$env:PYTHONPATH="src"
python -m iondynamics.cli run --config configs/default.yaml

# Command Prompt
set PYTHONPATH=src
python -m iondynamics.cli run --config configs/default.yaml
```

## 実行例

### シミュレーションの実行
```bash
iondynamics run --config configs/default.yaml
```

### 動画の生成
```bash
iondynamics animate --config configs/default.yaml --out outputs/animations/demo.mp4
```

### パラメータスイープ
```bash
iondynamics sweep --config configs/default.yaml --spec configs/sweep.yaml
```

### 粒子配置のプレビュー
```bash
iondynamics particles --mode random --n 500 --preview
```

## 出力先
- `outputs/runs/YYYYMMDD_HHMMSS_<slug>/`: 各実行の結果（YAML, CSV, 図, 動画）
- `outputs/figures/`: 静止画プロット
- `outputs/animations/`: 動画ファイル
