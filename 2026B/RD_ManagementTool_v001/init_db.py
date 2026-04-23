import sqlite3
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "rd_management.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 研究テーマテーブル
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS themes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        category TEXT,
        lead TEXT,
        members TEXT,
        priority TEXT,
        status TEXT,
        start_date TEXT,
        target_date TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # マイルストーンテーブル
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS milestones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        theme_id INTEGER,
        name TEXT NOT NULL,
        target_date TEXT,
        actual_date TEXT,
        progress INTEGER DEFAULT 0,
        delay_reason TEXT,
        next_action TEXT,
        member TEXT,
        FOREIGN KEY (theme_id) REFERENCES themes (id)
    )
    ''')

    # 課題・リスクテーブル
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS issues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        theme_id INTEGER,
        title TEXT NOT NULL,
        type TEXT, -- 課題 or リスク
        content TEXT,
        importance TEXT,
        impact TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        deadline TEXT,
        member TEXT,
        status TEXT,
        mitigation TEXT,
        remarks TEXT,
        FOREIGN KEY (theme_id) REFERENCES themes (id)
    )
    ''')

    # 実験結果・ナレッジテーブル
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        theme_id INTEGER,
        date TEXT,
        condition TEXT,
        result TEXT,
        discussion TEXT,
        materials TEXT,
        tags TEXT,
        related_issue_id INTEGER,
        related_knowledge_id INTEGER,
        FOREIGN KEY (theme_id) REFERENCES themes (id)
    )
    ''')

    # サンプルデータの投入 (既にデータがある場合はスキップ)
    cursor.execute("SELECT COUNT(*) FROM themes")
    if cursor.fetchone()[0] == 0:
        themes_data = [
            ("次世代高効率太陽電池の研究", "ペロブスカイト構造を用いた高効率化", "太陽光", "田中 太郎", "田中, 佐藤", "高", "進行中", "2026-04-01", "2027-03-31"),
            ("AIによる材料特性予測システム", "機械学習を用いた新素材探索の高速化", "デジタル", "鈴木 次郎", "鈴木, 伊藤", "中", "計画中", "2026-05-01", "2026-12-31")
        ]
        cursor.executemany("INSERT INTO themes (name, description, category, lead, members, priority, status, start_date, target_date) VALUES (?,?,?,?,?,?,?,?,?)", themes_data)
        
        milestones_data = [
            (1, "基礎データ収集", "2026-05-31", "2026-05-20", 100, "", "次のフェーズへ", "田中"),
            (1, "試作品製作", "2026-08-31", "", 20, "", "材料調達中", "田中")
        ]
        cursor.executemany("INSERT INTO milestones (theme_id, name, target_date, actual_date, progress, delay_reason, next_action, member) VALUES (?,?,?,?,?,?,?,?)", milestones_data)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()
