# IondynamicsSim

リチウムイオン二次電池の正極内におけるイオン輸送と濃度分布をシミュレーション・分析するためのツールです。特に「厚膜電極」における課題の可視化とKPI算出に特化しています。

## 特徴
- **厚み方向解析**: 電解液濃度の偏り（Δc_e）や勾配を定量化。
- **輸送抵抗KPI**: 厚膜化による輸送悪化を「抵抗指標」として数値化。
- **ケース比較**: 厚み・空隙率・粒径を振った比較レポートを自動生成。
- **インタラクティブGUI**: パラメータ変更から結果確認まで一貫して操作可能。
- **粒子レベル可視化**: 1D物理モデルの結果を2D粒子配置上に擬似投影し、直感的な理解を促進。

## インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# プロジェクトを編集可能モードでインストール（推奨）
pip install -e .
```

## 実行方法

### 1. GUIで実行
プロジェクトルートにある `run_gui.bat` を実行してください。
パラメータの入力、シミュレーション、比較実行、結果確認がすべてGUI上で行えます。

### 2. CLIで実行
詳細な条件指定や一括処理にはCLIが便利です。
```bash
# 厚み方向の比較実行とレポート生成
python -m iondynamics.cli compare --config configs/default.yaml --axis thickness --values 60 80 100

# 通常のシミュレーション実行
python -m iondynamics.cli run --config configs/default.yaml

# アニメーション生成
python -m iondynamics.cli animate --config configs/default.yaml
```

## ドキュメント
詳細な仕様や使い方は `docs/` フォルダを参照してください。
- [使用方法 (usage.md)](docs/usage.md)
- [制限事項と今後の展望 (limitations.md)](docs/limitations.md)

## 出力先
- `outputs/runs/YYYYMMDD_HHMMSS_<CaseName>/`: 各実行の結果（CSV, プロット, サマリ）
- 比較実行時は、同フォルダに `report.md` と比較用グラフが生成されます。
