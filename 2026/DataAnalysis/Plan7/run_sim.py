import os
import sys
import pandas as pd
from core.simulation import Simulation

def run_batch_simulation():
    print("="*60)
    print(" [WARNING] run_sim.py (CLI実行) は非推奨となりました。")
    print(" 今後は機能が統合された gui_sim.py を使用してください。")
    print("="*60)
    print("\nGUIを起動するには以下のコマンドを実行してください:")
    print("  python3 gui_sim.py")
    print("\nシミュレーションを強制実行したい場合は、このスクリプトの内容を確認してください。")
    sys.exit(0)

if __name__ == "__main__":
    run_batch_simulation()
