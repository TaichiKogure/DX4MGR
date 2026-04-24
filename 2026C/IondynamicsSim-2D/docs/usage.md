# IondynamicsSim 使用方法

IondynamicsSimは、リチウムイオン電池の正極内におけるイオン輸送と濃度分布をシミュレーション・分析するためのツールです。
本システムは**GUI（グラフィカルユーザーインターフェース）からの操作を基本**として設計されています。

## 1. 起動方法

プロジェクトルートにある以下のファイルのいずれかを使用して起動してください。

- **`run_gui.bat`**: Windows環境でダブルクリックして起動（推奨）。
- **`run_gui.py`**: Python環境から `python run_gui.py` で起動。

起動すると、詳細なパラメータ設定、シミュレーション実行、結果解析が可能なウィンドウが表示されます。

## 2. GUIの操作手順

### Settings タブ（設定）
- **YAMLの読み込み/保存**: 既存の設定ファイルを読み込んだり、現在のパラメータを保存したりできます。
- **パラメータ編集**: 厚み、空隙率、Cレート、物性値（伝導度、拡散係数）などを直接編集できます。
- **バリデーション**: 入力値が物理的に不正な場合、エラーメッセージが表示されます。

### Run / Compare タブ（実行・比較）
- **Run Single Simulation**: 現在の設定で単体シミュレーションを実行します。
- **Generate Animation**: 粒子の濃度変化をアニメーション（MP4）として出力します。
- **Run Comparison Report**: 指定したパラメータ（厚みなど）を変化させた複数ケースの比較レポートを自動生成します。
- **Run Sweep**: カスタムのスイープ設定（YAML）に基づいた一括計算を実行します。

### Analysis タブ（解析・結果確認）
- 実行完了後、このタブから最新のグラフやCSVデータ、レポートを直接開くことができます。
- 電圧推移、濃度プロファイル、KPI推移、抵抗内訳などがボタン一つで確認可能です。

## 3. 主要KPIと物理モデルの解説

システムの背景にある物理原理やKPIの詳細な定義については、以下の解説書を参照してください。

- **[技術解説書 (theory.md)](theory.md)**: DFNモデル、KPI算出ロジック、PyBaMMの活用方法について。

## 4. CLI（コマンドライン）による操作

自動化スクリプトやサーバー環境での実行のために、CLI機能も提供されています。

```bash
# PYTHONPATHをsrcに設定した上で実行
python -m iondynamics.cli run --config configs/default.yaml
python -m iondynamics.cli compare --config configs/default.yaml --axis thickness --values 60 80 100
```

詳細は `python -m iondynamics.cli --help` を参照してください。
