import sys
import os

# srcをPYTHONPATHに追加してiondynamicsパッケージをインポート可能にする
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from iondynamics.gui import main
except ImportError as e:
    print(f"Error: Could not import iondynamics. {e}")
    print(f"PYTHONPATH: {sys.path}")
    sys.exit(1)

if __name__ == "__main__":
    main()
