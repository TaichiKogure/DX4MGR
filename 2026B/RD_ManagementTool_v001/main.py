import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import os
import sys
from streamlit.web import cli as stcli

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "rd_management.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def load_data(table_name):
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def main():
    st.set_page_config(page_title="研究開発部マネジメント支援ツール", layout="wide")
    st.title("🧪 研究開発部マネジメント支援ツール (v0.0.1)")

    menu = ["ダッシュボード", "研究テーマ一覧", "進捗・マイルストーン", "課題・リスク", "実験結果・ナレッジ"]
    choice = st.sidebar.selectbox("メニュー", menu)

    if choice == "ダッシュボード":
        show_dashboard()
    elif choice == "研究テーマ一覧":
        show_themes()
    elif choice == "進捗・マイルストーン":
        show_milestones()
    elif choice == "課題・リスク":
        show_issues()
    elif choice == "実験結果・ナレッジ":
        show_experiments()

def show_dashboard():
    st.header("📊 全体ダッシュボード")
    
    themes_df = load_data("themes")
    milestones_df = load_data("milestones")
    issues_df = load_data("issues")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("進行中テーマ数", len(themes_df[themes_df['status'] == '進行中']))
    with col2:
        # 遅延の判定（簡易的に進捗100%未満で予定日を過ぎているもの）
        today = datetime.now().strftime("%Y-%m-%d")
        delayed = milestones_df[(milestones_df['progress'] < 100) & (milestones_df['target_date'] < today)]
        st.metric("遅延マイルストーン", len(delayed))
    with col3:
        st.metric("未解決課題数", len(issues_df[issues_df['status'] != '完了']))
    with col4:
        st.metric("登録済み実験数", len(load_data("experiments")))

    st.subheader("テーマ別進捗状況")
    if not themes_df.empty:
        # テーマごとの平均進捗率を計算（マイルストーンから）
        avg_progress = milestones_df.groupby('theme_id')['progress'].mean().reset_index()
        display_df = pd.merge(themes_df[['id', 'name', 'status', 'lead']], avg_progress, left_on='id', right_on='theme_id', how='left')
        display_df['progress'] = display_df['progress'].fillna(0)
        st.dataframe(display_df[['name', 'lead', 'status', 'progress']], width="stretch")

def show_themes():
    st.header("📋 研究テーマ一覧")
    themes_df = load_data("themes")
    
    # データ編集機能
    edited_df = st.data_editor(themes_df, num_rows="dynamic", width="stretch", key="themes_editor")
    
    if st.button("保存"):
        conn = get_connection()
        # 簡易的な全削除・再投入（本番ではUPDATE推奨だがMVPのため）
        edited_df.to_sql("themes", conn, if_exists="replace", index=False)
        conn.close()
        st.success("保存しました")

def show_milestones():
    st.header("📅 進捗・マイルストーン管理")
    themes_df = load_data("themes")
    milestones_df = load_data("milestones")
    
    theme_options = {row['id']: row['name'] for _, row in themes_df.iterrows()}
    
    # フィルタ
    selected_theme_id = st.selectbox("テーマで絞り込み", options=[None] + list(theme_options.keys()), format_func=lambda x: theme_options.get(x, "すべて"))
    
    display_df = milestones_df
    if selected_theme_id:
        display_df = milestones_df[milestones_df['theme_id'] == selected_theme_id]
    
    edited_df = st.data_editor(display_df, num_rows="dynamic", width="stretch", key="milestones_editor")
    
    if st.button("保存"):
        conn = get_connection()
        # 元のデータからこのテーマ分だけ置き換え、または全体置換
        # MVPなので全体置換で対応
        edited_df.to_sql("milestones", conn, if_exists="replace", index=False)
        conn.close()
        st.success("保存しました")

def show_issues():
    st.header("⚠️ 課題・リスク管理")
    issues_df = load_data("issues")
    
    edited_df = st.data_editor(issues_df, num_rows="dynamic", width="stretch", key="issues_editor")
    
    if st.button("保存"):
        conn = get_connection()
        edited_df.to_sql("issues", conn, if_exists="replace", index=False)
        conn.close()
        st.success("保存しました")

def show_experiments():
    st.header("🧪 実験結果・ナレッジ")
    exp_df = load_data("experiments")
    
    edited_df = st.data_editor(exp_df, num_rows="dynamic", width="stretch", key="exp_editor")
    
    if st.button("保存"):
        conn = get_connection()
        edited_df.to_sql("experiments", conn, if_exists="replace", index=False)
        conn.close()
        st.success("保存しました")

if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
