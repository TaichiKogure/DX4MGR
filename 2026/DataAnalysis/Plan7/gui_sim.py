import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import threading
import queue
import time
import json
try:
    import psutil
except ImportError:
    psutil = None

# Simulationクラスのインポート
try:
    from core.simulation import Simulation
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.simulation import Simulation

PARAM_DEFINITIONS = [
    # 基本設定 / Basic Settings
    {"key": "scenario_id", "label": "シナリオID / Scenario ID", "default": "Scenario_A", "type": "entry", "description": "シミュレーション実行の識別名。 / Identifier for simulation execution."},
    {"key": "steps", "label": "シミュレーション日数 / Simulation Days", "default": 500, "type": "int", "range": (100, 2000), "description": "シミュレーションを実行する総期間（日単位）。 / Total duration of the simulation (days)."},
    {"key": "strategy", "label": "実験戦略 / Strategy", "default": "strategic", "type": "combo", "values": ["strategic", "random"], "description": "strategic: 技術成熟度に基づく戦略的選択, random: ランダムな実験選択 / strategic: maturity-based choice, random: random choice"},
    {"key": "use_digital_twin", "label": "デジタルツイン / Digital Twin", "default": True, "type": "check", "description": "デジタルツインによるシミュレーション加速・予測機能の有効化。 / Enable digital twin simulation acceleration."},
    {"key": "seed", "label": "乱数シード / Random Seed", "default": 42, "type": "int", "range": (0, 10000), "description": "乱数生成の初期値。同じ値にすると結果の再現性が確保されます。 / Initial value for RNG. Ensures reproducibility."},
    
    # リソース / Resources
    {"key": "res_n_servers", "label": "Research エンジニア数 / Res Engineers", "default": 5, "type": "int", "range": (1, 20), "description": "基礎研究・探索フェーズに割り当てる人数。 / Number of engineers for research phase."},
    {"key": "proto_n_servers", "label": "Prototype エンジニア数 / Proto Engineers", "default": 3, "type": "int", "range": (1, 20), "description": "試作・検証フェーズに割り当てる人数。 / Number of engineers for prototyping phase."},
    {"key": "mass_n_servers", "label": "Mass Production 数 / Mass Production", "default": 2, "type": "int", "range": (1, 20), "description": "量産化・プロセス適合フェーズに割り当てる人数。 / Number of servers for mass production phase."},
    {"key": "wip_limit_res", "label": "Research WIP制限 / Res WIP Limit", "default": 2, "type": "int", "range": (1, 10), "description": "研究フェーズで同時に進行できるテーマ数。 / Max parallel themes in research."},
    {"key": "wip_limit_proto", "label": "Prototype WIP制限 / Proto WIP Limit", "default": 1, "type": "int", "range": (1, 10), "description": "試作フェーズで同時に進行できるテーマ数。 / Max parallel themes in prototyping."},
    
    # 技術・ゲート / Technology & Gates
    {"key": "dr1_uncert_limit", "label": "DR1 不確実性閾値 / DR1 Uncert Limit", "default": 0.4, "type": "float", "range": (0.0, 1.0), "description": "DR1通過に必要な不確実性の低減度。 / Uncertainty threshold for DR1 pass."},
    {"key": "dr2_matur_limit", "label": "DR2 成熟度閾値 / DR2 Maturity Limit", "default": 0.5, "type": "float", "range": (0.0, 1.0), "description": "DR2通過に必要な技術成熟度。 / Maturity threshold for DR2 pass."},
    {"key": "dr3_matur_limit", "label": "DR3 成熟度閾値 / DR3 Maturity Limit", "default": 0.8, "type": "float", "range": (0.0, 1.0), "description": "DR3通過に必要な技術成熟度。 / Maturity threshold for DR3 pass."},
    {"key": "decay", "label": "知識減衰率 / Knowledge Decay", "default": 0.7, "type": "float", "range": (0.0, 1.0), "description": "時間の経過やリワークによって失われる知識の割合。 / Rate of knowledge loss over time/rework."},
    {"key": "rework_load_factor", "label": "リワーク負荷係数 / Rework Load Factor", "default": 0.5, "type": "float", "range": (0.0, 1.0), "description": "不具合修正時の追加負荷。 / Additional load factor for bug fixes."},
    {"key": "max_rework_cycles", "label": "最大リワーク回数 / Max Rework Cycles", "default": 5, "type": "int", "range": (1, 10), "description": "許容される最大のリワーク回数。 / Maximum allowed rework cycles."},
    {"key": "tacitness", "label": "暗黙知度 / Tacitness", "default": 0.5, "type": "float", "range": (0.0, 1.0), "description": "言語化しにくい知識の割合。 / Ratio of tacit (undocumented) knowledge."},
    
    # 通信・DOE / Communication & DOE
    {"key": "loss_res_proto", "label": "Res->Proto 欠落率 / Information Loss", "default": 0.1, "type": "float", "range": (0.0, 0.5), "description": "情報の欠落割合。 / Ratio of information loss during handoff."},
    {"key": "dist_proto_ana", "label": "Proto->Ana 歪み率 / Info Distortion", "default": 0.05, "type": "float", "range": (0.0, 0.5), "description": "情報のノイズ・歪み割合。 / Ratio of noise/distortion in feedback."},
    {"key": "delay_ana_res", "label": "Ana->Res 遅延 / Feedback Delay", "default": 2, "type": "int", "range": (0, 10), "description": "フィードバック遅延（日）。 / Delay in days for feedback loop."},
    {"key": "dr1_period", "label": "DR1 会議周期 / DR1 Period", "default": 30, "type": "int", "range": (7, 90), "description": "DRを実施する周期。 / Frequency of DR meetings."},
    {"key": "cost_per_review", "label": "レビュー単価 / Review Cost", "default": 100, "type": "int", "range": (10, 500), "description": "1回のレビューにかかるコスト。 / Cost per single DR session."},

    # Ver2: 部署/会議/ハンドオフ/リワーク規則の考慮トグル（最小）
    {"key": "consider_departments", "label": "部署を考慮 / Consider Departments", "default": True, "type": "check", "description": "部署カレンダー/SLA/コスト係数を考慮（最小実装: SLA最大待機・コスト係数反映）。 / Consider dept SLA & cost (minimal)."},
    {"key": "consider_handoffs", "label": "ハンドオフを考慮 / Consider Handoffs", "default": True, "type": "check", "description": "部署間ハンドオフのインターフェース品質・情報損失・転送遅延を考慮（最小実装）。 / Apply handoff delay and quality."},
    {"key": "consider_cross_meetings", "label": "横断会議を考慮 / Cross-Dept Meetings", "default": False, "type": "check", "description": "部署横断会議の閾値ロジック（最小スタブ）を有効化。 / Enable cross-dept meeting stub (minimal)."},
    {"key": "consider_rework_rules", "label": "リワーク規則を考慮 / Rework Rules", "default": True, "type": "check", "description": "OR/AND結合や再作業規則（将来拡張）を保持。 / Keep rework rules (future ext)."}
]

GRAPH_EXPLANATIONS = {
    "KPI": "主要業績指標（KPI）の推移。累積利益、総コスト、成功したテーマ数を時系列で追跡。\nKey Performance Indicators: Tracking cumulative gain, total cost, and completed themes over time.",
    "技術状態 / Tech Status": "開発テーマの技術成熟度（TRL）と不確実性。DR審査閾値との関係を可視化。\nTechnology Status: Visualizing maturity (TRL) and uncertainty relative to DR thresholds.",
    "技術推移 / Tech Evolution": "各技術項目の成熟度向上プロセス。知識蓄積とリワークの影響を表示。\nTech Evolution: Shows how maturity improves over time, including rework effects.",
    "WIP/負荷 / WIP Heatmap": "各工程の仕掛品（WIP）数とエンジニア負荷のヒートマップ。ボトルネック特定に活用。\nWIP/Load: Heatmap of work-in-progress and engineer load to identify bottlenecks.",
    "LT分布 / Lead Time Dist": "開発テーマ完了までのリードタイム（LT）分布。プロセスの安定性を表示。\nLead Time Distribution: Stability and predictability of the development process.",
    "ゲート待機 / Gate Waiting": "DR審査待機の数と時間。意思決定の停滞を可視化。\nGate Waiting: Number and duration of items waiting for DR decision.",
    "リソース稼働率 / Resource Util": "エンジニアや設備の稼働率推移。リソース配分の妥当性を分析。\nResource Utilization: Tracking how busy engineers are to optimize allocation.",
    "累積コスト / Cumulative Cost": "プロジェクト進行に伴う累積費用の推移。投資対効果の確認。\nCumulative Cost: Trend of total expenses to evaluate ROI.",
    "LT内訳 / Lead Time Breakdown": "リードタイムを「実作業」「待ち時間」「判定待ち」に分解。無駄の特定。\nLead Time Breakdown: Decomposing LT into work, wait, and decision latency."
}

class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plan 7: Hybrid R&D Simulation Dashboard (v2.2)")
        self.root.geometry("1400x950")

        self.log_queue = queue.Queue()
        self.is_running = False
        
        # Execution time tracking
        self.start_times = [None, None, None]
        self.elapsed_times = [0.0, 0.0, 0.0]
        self.progress_values = [0.0, 0.0, 0.0]
        self.scenario_states = ["idle", "idle", "idle"]
        self.progress_log_next = [10.0, 10.0, 10.0]
        self.last_progress_log_at = [0.0, 0.0, 0.0]
        self.plot_refresh_job = None
        self.pending_single_plot = None
        self.pending_comparison_plot = None
        self.pending_plot_mode = None
        self.plot_refresh_interval_ms = 1200
        self.last_plot_refresh_at = 0.0
        self.plot_render_in_progress = False
        self.plot_refresh_requested = False
        
        # Plot visibility flags
        self.plot_visibility = {
            'kpi': tk.BooleanVar(value=True),
            'tech_status': tk.BooleanVar(value=True),
            'tech_history': tk.BooleanVar(value=True),
            'wip': tk.BooleanVar(value=True),
            'lead_time': tk.BooleanVar(value=True),
            'gate_waits': tk.BooleanVar(value=True),
            'resource_util': tk.BooleanVar(value=True),
            'cost_trend': tk.BooleanVar(value=True),
            'lt_breakdown': tk.BooleanVar(value=True)
        }
        
        self.comparison_data = {} # To hold partial results during comparison
        self.layout_cols = tk.IntVar(value=2) # Default to 2 for better use of space
        
        # Scenario Parameters (A, B, C)
        self.scenario_params = [{}, {}, {}]
        self.compare_mode = tk.BooleanVar(value=True)
        # シナリオ有効/無効（A/B/C）
        self.scenario_enabled = [tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)]
        # Ver2: 動画/3D出力オプション
        self.export_video_var = tk.BooleanVar(value=False)
        # Ver2+: 3D出力の種類選択とフレーム間引き
        self.viz3d_wip_var = tk.BooleanVar(value=True)
        self.viz3d_lt_var = tk.BooleanVar(value=False)
        self.viz3d_tech_var = tk.BooleanVar(value=False)
        self.viz3d_net_var = tk.BooleanVar(value=False)
        self.video_frame_step_var = tk.IntVar(value=1)
        # 3D動画の回転/視点設定
        self.video_rotate_var = tk.BooleanVar(value=True)
        self.video_elev_var = tk.IntVar(value=25)
        self.video_azim_var = tk.IntVar(value=45)
        # 3Dデザイン（各グラフごと）
        self.viz3d_style_options = [
            "Default", "Dark", "Publication", "Wireframe", "HighContrast", "Monochrome"
        ]
        self.viz3d_wip_style_var = tk.StringVar(value="Default")
        self.viz3d_lt_style_var = tk.StringVar(value="Publication")
        self.viz3d_tech_style_var = tk.StringVar(value="Default")
        self.viz3d_net_style_var = tk.StringVar(value="Default")
        
        # Configure fonts for JP/EN rendering in Matplotlib (avoid tofu squares)
        self._configure_matplotlib_fonts()

        self.setup_ui()
        self.periodic_check()

    def setup_ui(self):
        # Main layout: Left (Input), Right (Dashboard)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(left_frame, weight=1)

        right_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(right_frame, weight=2)

        # --- Left Frame: Table-style Input ---
        input_header = ttk.Frame(left_frame)
        input_header.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(input_header, text="シミュレーション・パラメータ設定", font=("", 12, "bold")).pack(side=tk.LEFT)
        
        # Scenarios Header
        header_row = ttk.Frame(left_frame)
        header_row.pack(fill=tk.X)
        ttk.Label(header_row, text="パラメータ名", width=20).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(header_row, text="Scenario A", width=12).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(header_row, text="Scenario B", width=12).grid(row=0, column=2, sticky=tk.W)
        ttk.Label(header_row, text="Scenario C", width=12).grid(row=0, column=3, sticky=tk.W)

        # シナリオ有効チェックボックス（A/B/C）
        en_row = ttk.Frame(left_frame)
        en_row.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(en_row, text="計算するシナリオ:").grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(en_row, text="A", variable=self.scenario_enabled[0]).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(en_row, text="B", variable=self.scenario_enabled[1]).grid(row=0, column=2, sticky=tk.W)
        ttk.Checkbutton(en_row, text="C", variable=self.scenario_enabled[2]).grid(row=0, column=3, sticky=tk.W)
        
        # Scrollable area for parameters
        table_container = ttk.Frame(left_frame)
        table_container.pack(fill=tk.BOTH, expand=True)
        
        canvas_params = tk.Canvas(table_container, highlightthickness=0)
        scrollbar_params = ttk.Scrollbar(table_container, orient="vertical", command=canvas_params.yview)
        self.params_frame = ttk.Frame(canvas_params)
        
        self.params_frame.bind(
            "<Configure>",
            lambda e: canvas_params.configure(scrollregion=canvas_params.bbox("all"))
        )
        canvas_params.create_window((0, 0), window=self.params_frame, anchor="nw")
        canvas_params.configure(yscrollcommand=scrollbar_params.set)
        
        canvas_params.pack(side="left", fill="both", expand=True)
        scrollbar_params.pack(side="right", fill="y")

        # Create table rows based on PARAM_DEFINITIONS
        self._setup_param_table()

        # --- Action Buttons (Bottom of Left Frame) ---
        btn_frame = ttk.Frame(left_frame, padding=10)
        btn_frame.pack(fill=tk.X)

        self.run_btn = ttk.Button(btn_frame, text="シミュレーション実行", command=self.start_simulation)
        self.run_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.batch_btn = ttk.Button(btn_frame, text="DOE一括実行 (5回)", command=self.start_batch_simulation)
        self.batch_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.viz_btn = ttk.Button(btn_frame, text="詳細レポート出力", command=self.output_viz_report)
        self.viz_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # --- JSON Import/Export Buttons ---
        io_frame = ttk.Frame(left_frame, padding=10)
        io_frame.pack(fill=tk.X)

        self.save_btn = ttk.Button(io_frame, text="設定を保存 (JSON)", command=self.save_params_json)
        self.save_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.load_btn = ttk.Button(io_frame, text="設定を読込 (JSON)", command=self.load_params_json)
        self.load_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Ver2: 詳細レポート時に動画/3Dも生成
        self.cb_export_video = ttk.Checkbutton(io_frame, text="動画も生成（3D/2D）",
                                               variable=self.export_video_var)
        self.cb_export_video.pack(side=tk.LEFT, padx=5)

        # 3D出力の詳細オプション
        three_d_opts = ttk.Frame(io_frame)
        three_d_opts.pack(side=tk.LEFT, padx=10)
        ttk.Label(three_d_opts, text="3D出力:").grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(three_d_opts, text="WIPサーフェス", variable=self.viz3d_wip_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(three_d_opts, text="LT分解", variable=self.viz3d_lt_var).grid(row=0, column=2, sticky=tk.W)
        ttk.Checkbutton(three_d_opts, text="技術拡散", variable=self.viz3d_tech_var).grid(row=0, column=3, sticky=tk.W)
        ttk.Checkbutton(three_d_opts, text="ネットワーク", variable=self.viz3d_net_var).grid(row=0, column=4, sticky=tk.W)

        # フレーム間引き
        frame_opt = ttk.Frame(io_frame)
        frame_opt.pack(side=tk.LEFT, padx=10)
        ttk.Label(frame_opt, text="フレーム間引き:").pack(side=tk.LEFT)
        self.frame_step_cmb = ttk.Combobox(frame_opt, width=4, state="readonly",
                                           values=["1", "2", "3", "5", "10"])
        self.frame_step_cmb.set(str(self.video_frame_step_var.get()))
        def _on_frame_step_change(event=None):
            try:
                val = int(self.frame_step_cmb.get())
                self.video_frame_step_var.set(max(1, val))
            except Exception:
                self.video_frame_step_var.set(1)
                self.frame_step_cmb.set("1")
        self.frame_step_cmb.bind("<<ComboboxSelected>>", _on_frame_step_change)
        self.frame_step_cmb.pack(side=tk.LEFT)

        # 回転と視点設定
        view_opt = ttk.Frame(io_frame)
        view_opt.pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(view_opt, text="回転", variable=self.video_rotate_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(view_opt, text="elev:").grid(row=0, column=1, sticky=tk.E)
        self.entry_elev = ttk.Entry(view_opt, width=4, textvariable=self.video_elev_var)
        self.entry_elev.grid(row=0, column=2, sticky=tk.W)
        ttk.Label(view_opt, text="azim:").grid(row=0, column=3, sticky=tk.E)
        self.entry_azim = ttk.Entry(view_opt, width=4, textvariable=self.video_azim_var)
        self.entry_azim.grid(row=0, column=4, sticky=tk.W)

        # デザインプリセット選択（各3Dグラフごと）
        design_opt = ttk.Frame(io_frame)
        design_opt.pack(side=tk.LEFT, padx=10)
        ttk.Label(design_opt, text="3Dデザイン:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(design_opt, text="WIP").grid(row=0, column=1, sticky=tk.E, padx=(6, 2))
        self.cbx_wip_style = ttk.Combobox(design_opt, width=12, state="readonly",
                                          values=self.viz3d_style_options,
                                          textvariable=self.viz3d_wip_style_var)
        self.cbx_wip_style.grid(row=0, column=2, sticky=tk.W)
        ttk.Label(design_opt, text="LT").grid(row=0, column=3, sticky=tk.E, padx=(8, 2))
        self.cbx_lt_style = ttk.Combobox(design_opt, width=12, state="readonly",
                                         values=self.viz3d_style_options,
                                         textvariable=self.viz3d_lt_style_var)
        self.cbx_lt_style.grid(row=0, column=4, sticky=tk.W)
        ttk.Label(design_opt, text="Tech").grid(row=0, column=5, sticky=tk.E, padx=(8, 2))
        self.cbx_tech_style = ttk.Combobox(design_opt, width=12, state="readonly",
                                           values=self.viz3d_style_options,
                                           textvariable=self.viz3d_tech_style_var)
        self.cbx_tech_style.grid(row=0, column=6, sticky=tk.W)
        ttk.Label(design_opt, text="Net").grid(row=0, column=7, sticky=tk.E, padx=(8, 2))
        self.cbx_net_style = ttk.Combobox(design_opt, width=12, state="readonly",
                                          values=self.viz3d_style_options,
                                          textvariable=self.viz3d_net_style_var)
        self.cbx_net_style.grid(row=0, column=8, sticky=tk.W)

        # --- Graph Visibility Controls ---
        vis_frame = ttk.LabelFrame(left_frame, text="グラフ表示設定 / Visualization Settings", padding=5)
        vis_frame.pack(fill=tk.X, pady=5)
        
        # Grid for checkboxes to save space
        cb_kpi = ttk.Checkbutton(vis_frame, text="KPI", variable=self.plot_visibility['kpi'], command=self._refresh_plots)
        cb_kpi.grid(row=0, column=0, sticky=tk.W)
        cb_tech_s = ttk.Checkbutton(vis_frame, text="技術状態 / Status", variable=self.plot_visibility['tech_status'], command=self._refresh_plots)
        cb_tech_s.grid(row=0, column=1, sticky=tk.W)
        cb_tech_h = ttk.Checkbutton(vis_frame, text="技術推移 / Evolution", variable=self.plot_visibility['tech_history'], command=self._refresh_plots)
        cb_tech_h.grid(row=1, column=0, sticky=tk.W)
        cb_wip = ttk.Checkbutton(vis_frame, text="WIP/負荷", variable=self.plot_visibility['wip'], command=self._refresh_plots)
        cb_wip.grid(row=1, column=1, sticky=tk.W)
        cb_lt = ttk.Checkbutton(vis_frame, text="LT分布 / LT Dist", variable=self.plot_visibility['lead_time'], command=self._refresh_plots)
        cb_lt.grid(row=2, column=0, sticky=tk.W)
        cb_gate = ttk.Checkbutton(vis_frame, text="ゲート待機 / Gates", variable=self.plot_visibility['gate_waits'], command=self._refresh_plots)
        cb_gate.grid(row=2, column=1, sticky=tk.W)
        cb_res = ttk.Checkbutton(vis_frame, text="リソース稼働率 / Util", variable=self.plot_visibility['resource_util'], command=self._refresh_plots)
        cb_res.grid(row=3, column=0, sticky=tk.W)
        cb_cost = ttk.Checkbutton(vis_frame, text="累積コスト / Cost", variable=self.plot_visibility['cost_trend'], command=self._refresh_plots)
        cb_cost.grid(row=3, column=1, sticky=tk.W)
        cb_lt_b = ttk.Checkbutton(vis_frame, text="LT内訳 / LT Break", variable=self.plot_visibility['lt_breakdown'], command=self._refresh_plots)
        cb_lt_b.grid(row=4, column=0, sticky=tk.W)

        # Layout selection
        layout_frame = ttk.Frame(vis_frame)
        layout_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(layout_frame, text="レイアウト:").pack(side=tk.LEFT)
        ttk.Radiobutton(layout_frame, text="1列", variable=self.layout_cols, value=1, command=self._refresh_plots).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(layout_frame, text="2列", variable=self.layout_cols, value=2, command=self._refresh_plots).pack(side=tk.LEFT, padx=5)

        # --- Status Monitoring Frame ---
        status_frame = ttk.LabelFrame(left_frame, text="ステータス / System Status", padding=5)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.cpu_mem_lbl = ttk.Label(status_frame, text="CPU: -% | Mem: -%")
        self.cpu_mem_lbl.pack(fill=tk.X)
        
        self.progress_bars = []
        self.progress_labels = []
        for i in range(3):
            lbl = ttk.Label(status_frame, text=f"Scenario {chr(65+i)}: 0% (0.0s)")
            lbl.pack(fill=tk.X)
            pb = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
            pb.pack(fill=tk.X, pady=(0, 5))
            self.progress_labels.append(lbl)
            self.progress_bars.append(pb)

        # --- Right Frame: Dashboard & Explanation ---
        self.right_notebook = ttk.Notebook(right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)

        # Dashboard Tab
        self.tab_dashboard = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_dashboard, text="ダッシュボード")

        # Explanation Tab
        self.tab_explanation = ttk.Frame(self.right_notebook, padding=20)
        self.right_notebook.add(self.tab_explanation, text="グラフ解説")
        self._setup_explanation_tab()

        # --- Dashboard Tab Content ---
        # Top: KPI Results Text
        self.results_text = tk.Text(self.tab_dashboard, height=12, width=60, font=("Courier", 10))
        self.results_text.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Bottom: Plot area with Scrollbar
        plot_wrapper = ttk.Frame(self.tab_dashboard)
        plot_wrapper.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(plot_wrapper)
        self.scrollbar = ttk.Scrollbar(plot_wrapper, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Mouse wheel support
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        self.fig_canvases = []

    def _setup_param_table(self):
        for i, param in enumerate(PARAM_DEFINITIONS):
            key = param['key']
            label = param['label']
            p_type = param['type']
            description = param.get('description', '')
            
            # Label with Help Icon
            lbl_frame = ttk.Frame(self.params_frame)
            lbl_frame.grid(row=i, column=0, padx=5, pady=2, sticky=tk.W)
            
            lbl = ttk.Label(lbl_frame, text=label, width=22, anchor="w")
            lbl.pack(side=tk.LEFT)
            
            help_lbl = ttk.Label(lbl_frame, text=" [?]", foreground="blue", cursor="question_arrow")
            help_lbl.pack(side=tk.LEFT)
            self._add_tooltip(help_lbl, description)
            
            # For each scenario (A, B, C)
            for s_idx in range(3):
                default_val = param['default']
                if key == "scenario_id":
                    default_val = f"Scenario_{chr(65+s_idx)}"
                
                if p_type == "entry":
                    var = tk.StringVar(value=str(default_val))
                    widget = ttk.Entry(self.params_frame, textvariable=var, width=12)
                elif p_type == "int":
                    var = tk.IntVar(value=int(default_val))
                    low, high = param['range']
                    widget = ttk.Spinbox(self.params_frame, from_=low, to=high, textvariable=var, width=10)
                elif p_type == "float":
                    var = tk.DoubleVar(value=float(default_val))
                    low, high = param['range']
                    widget = ttk.Spinbox(self.params_frame, from_=low, to=high, increment=0.1, textvariable=var, width=10)
                elif p_type == "check":
                    var = tk.BooleanVar(value=default_val)
                    widget = ttk.Checkbutton(self.params_frame, variable=var)
                elif p_type == "combo":
                    var = tk.StringVar(value=default_val)
                    widget = ttk.Combobox(self.params_frame, textvariable=var, values=param['values'], width=10)
                
                widget.grid(row=i, column=s_idx+1, padx=3, pady=2, sticky=tk.EW)
                self.scenario_params[s_idx][key] = var
                
                # Trace changes to highlight differences
                var.trace_add("write", lambda *args, k=key: self._on_param_change(k))

    def _setup_explanation_tab(self):
        # Scrollable text area for explanations
        txt_area = tk.Text(self.tab_explanation, wrap=tk.WORD, font=("Yu Gothic", 11), bg="#f9f9f9", padx=10, pady=10)
        txt_area.pack(fill=tk.BOTH, expand=True)
        
        txt_area.insert(tk.END, "■ グラフの解説と意思決定への活用 / Chart Explanations & Decision Support\n\n", "header")
        txt_area.tag_configure("header", font=("", 14, "bold"))
        txt_area.tag_configure("graph_name", font=("", 11, "bold"), foreground="#2c3e50")
        
        for name, desc in GRAPH_EXPLANATIONS.items():
            txt_area.insert(tk.END, f"▼ {name}\n", "graph_name")
            txt_area.insert(tk.END, f"{desc}\n\n")
        
        txt_area.config(state=tk.DISABLED)
        
    def _configure_matplotlib_fonts(self):
        """Ensure Matplotlib can render Japanese text properly on Windows.
        Prefer common JP fonts; gracefully fall back to defaults.
        """
        try:
            jp_candidates = [
                'Yu Gothic', 'Meiryo', 'MS Gothic',
                'Noto Sans CJK JP', 'IPAGothic', 'Hiragino Sans'
            ]
            # Prepend JP fonts to sans-serif list, keep existing ones as fallback
            existing = list(rcParams.get('font.sans-serif', []))
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.sans-serif'] = jp_candidates + existing + ['DejaVu Sans']
            # Avoid unicode minus rendering issues
            rcParams['axes.unicode_minus'] = False
        except Exception:
            # Safe no-op on environments where rcParams is locked
            pass

    def _on_param_change(self, key):
        # Optional: highlight cells that differ between scenarios
        pass

    def _on_mousewheel(self, event):
        if sys.platform == 'darwin':
            self.canvas.yview_scroll(int(-1 * (event.delta)), "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


    def _add_tooltip(self, widget, text):
        def enter(event):
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(self.tooltip, text=text, justify='left', background="#ffffe0", relief='solid', borderwidth=1, font=("Yu Gothic", 9))
            label.pack()
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def get_parsed_params(self, params_dict=None):
        if params_dict is None: params_dict = self.scenario_params[0]
        p = {}
        for k, v in params_dict.items():
            val = v.get()
            # 型変換
            if k in ['steps', 'seed', 'dr1_period']:
                p[k] = int(float(val))
            elif k in ['res_n_servers', 'proto_n_servers', 'mass_n_servers', 'wip_limit_res', 'wip_limit_proto', 'max_rework_cycles']:
                p[k] = int(float(val))
            elif isinstance(val, bool):
                p[k] = val
            elif isinstance(val, (float, int)):
                p[k] = val
            elif isinstance(val, str) and val.replace('.','',1).isdigit():
                p[k] = float(val)
            else:
                p[k] = val
        
        # 構造化が必要なパラメータの組み立て
        p['tech_items'] = [
            {"name": "Core Tech", "tacitness": p.get('tacitness', 0.5)},
            {"name": "Interface", "tacitness": 0.3}
        ]
        p['teams'] = [
            {"name": "Research", "wip_limit": p.get('wip_limit_res', 2)},
            {"name": "Prototype", "wip_limit": p.get('wip_limit_proto', 1)},
            {"name": "Analysis", "wip_limit": 2},
            {"name": "MassProduction", "wip_limit": 5}
        ]

        # Ver2: トグルON時に最小の部署/ハンドオフ/会議設定を自動生成（GUI未入力でも動くように）
        if p.get('consider_departments'):
            # 部署IDの割当: R, P, A, M
            team_order = ["Research", "Prototype", "Analysis", "MassProduction"]
            dept_ids = {"Research": "R", "Prototype": "P", "Analysis": "A", "MassProduction": "M"}
            # teamsにdeptキーを付与
            for t in p['teams']:
                t_name = t.get('name')
                if t_name in dept_ids:
                    t['dept'] = dept_ids[t_name]
            p['departments'] = [
                {"id": did, "name": name, "cost_factor": cf, "sla": {"avg_response": 0.0, "max_wait": mw}}
                for (name, did, cf, mw) in [
                    ("ResearchDept", "R", 1.0, 8.0),
                    ("ProtoDept", "P", 1.1, 10.0),
                    ("AnalysisDept", "A", 1.0, 12.0),
                    ("MassProdDept", "M", 1.3, 15.0)
                ]
            ]

        if p.get('consider_handoffs'):
            p['handoffs'] = [
                {"from": "R", "to": "P", "q_if": 0.9, "lambda": 0.1, "wait_dist": {"type": "exponential", "params": {"scale": 1.5}}},
                {"from": "P", "to": "A", "q_if": 0.92, "lambda": 0.08, "wait_dist": {"type": "exponential", "params": {"scale": 1.0}}},
                {"from": "A", "to": "R", "q_if": 0.85, "lambda": 0.12, "wait_dist": {"type": "uniform", "params": {"low": 0.5, "high": 2.0}}},
                {"from": "P", "to": "M", "q_if": 0.8, "lambda": 0.2, "wait_dist": {"type": "constant", "params": {"value": 2.0}}}
            ]

        if p.get('consider_cross_meetings'):
            p['cross_meetings'] = [
                {"departments": ["R", "P", "A"], "interval": 30.0, "threshold": 0.0, "logic": "GO"}
            ]
        p['dr_threshold'] = 5.0 # Fixed for now
        p['past_logs_file'] = 'configs/past_logs.csv'
        return p

    def start_simulation(self):
        if self.is_running: return
        
        scenarios = []  # 長さ3、無効はNone
        for i in range(3):
            if self.scenario_enabled[i].get():
                scenarios.append(self.get_parsed_params(self.scenario_params[i]))
            else:
                scenarios.append(None)
        if all(p is None for p in scenarios):
            messagebox.showwarning("Warning", "少なくとも1つのシナリオを選択してください（A/B/C）。")
            return
        
        # Initialize progress bars and labels
        for i in range(3):
            if scenarios[i] is None:
                # スキップ
                self.start_times[i] = None
                self.elapsed_times[i] = 0.0
                self.progress_values[i] = 0.0
                self.scenario_states[i] = "skipped"
                self.progress_bars[i]['value'] = 0
                self.progress_labels[i].config(text=f"Scenario {chr(65+i)}: Skipped")
            else:
                self.start_times[i] = time.perf_counter()
                self.elapsed_times[i] = 0.0
                self.progress_values[i] = 0.0
                self.scenario_states[i] = "running"
                self.progress_log_next[i] = 10.0
                self.last_progress_log_at[i] = 0.0
                self.progress_bars[i]['value'] = 0
                self.progress_labels[i].config(text=self._format_progress_label(i))
        self.cpu_mem_lbl.config(text="CPU: -% | Mem: -%")
        self.comparison_data = {}
        self.pending_comparison_plot = None
        self.pending_single_plot = None
        
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "比較シミュレーション開始（選択シナリオのみ）... / Start selected-scenarios comparison...\n")
        
        thread = threading.Thread(target=self.run_comparison_thread, args=(scenarios,))
        thread.daemon = True
        thread.start()

    def run_comparison_thread(self, scenarios):
        try:
            self.comparison_data = {}
            def run_single_sim(p, label, idx):
                start_time = time.perf_counter()
                sim = Simulation()
                sim.setup_with_params(p)
                progress_state = {
                    "last_emit_at": start_time,
                    "last_partial_sim_time": -1.0
                }
                
                def live_cb(at_time):
                    # システム情報の取得 (psutil)
                    cpu = psutil.cpu_percent() if psutil else 0.0
                    mem = psutil.virtual_memory().percent if psutil else 0.0
                    
                    now_perf = time.perf_counter()
                    elapsed = now_perf - start_time
                    progress = min(100.0, (at_time / p['steps']) * 100)

                    if progress >= 100.0 or now_perf - progress_state["last_emit_at"] >= 0.25:
                        progress_state["last_emit_at"] = now_perf
                        self.log_queue.put(("progress", (idx, progress, elapsed, cpu, mem)))
                    
                    if (at_time - progress_state["last_partial_sim_time"] >= 100) or progress >= 100.0:
                        progress_state["last_partial_sim_time"] = at_time
                        # Send partial results
                        p_kpis = sim.kpis.copy()
                        for node in sim.engine.nodes.values():
                            if hasattr(node, 'total_cost'):
                                p_kpis['dr_cost'] += node.total_cost
                        
                        p_tech_status = sim.get_tech_status()
                        p_metrics = {
                            'tech_history': list(sim.tech_history),
                            'wip_history': list(sim.engine.results["wip_history"]),
                            'gate_stats': [n.get_stats() for n in sim.engine.nodes.values() if hasattr(n, 'get_stats')]
                        }
                        self.log_queue.put(("comp_partial_result", (label, p_kpis, p, p_tech_status, p_metrics)))

                sim.gui_callback = live_cb
                kpis = sim.run(steps=p['steps'])
                tech_status = sim.get_tech_status()
                
                final_elapsed = time.perf_counter() - start_time
                final_cpu = psutil.cpu_percent() if psutil else 0.0
                final_mem = psutil.virtual_memory().percent if psutil else 0.0
                self.log_queue.put(("progress", (idx, 100.0, final_elapsed, final_cpu, final_mem)))
                self.log_queue.put(("scenario_done", (idx, final_elapsed)))
                self.log_queue.put(("log", f"{label} completed in {final_elapsed:.2f}s\n"))
                
                from analysis.metrics import calculate_metrics
                nodes_stats = [n.get_stats() for n in sim.engine.nodes.values() if hasattr(n, 'get_stats')]
                metrics = calculate_metrics(sim.engine.results["completed_jobs"], nodes_stats, sim.engine.now, sim.engine.results["wip_history"])
                metrics['tech_history'] = sim.tech_history
                metrics['wip_history'] = sim.engine.results["wip_history"]
                metrics['gate_stats'] = nodes_stats
                
                # 単一シナリオ完了時に、詳細レポート出力用に保持しておく
                # 比較実行後に「詳細レポート出力」ボタンを押しても動作するようにするため
                self.last_sim = sim
                self.last_params = p

                self.log_queue.put(("comp_partial_result", (label, kpis, p, tech_status, metrics)))

            threads = []
            labels = ["Scenario A", "Scenario B", "Scenario C"]
            for i, p in enumerate(scenarios):
                if p is None:
                    continue  # スキップ
                t = threading.Thread(target=run_single_sim, args=(p, labels[i], i))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            self.log_queue.put(("log", "Comparison simulation completed successfully.\n"))
        except Exception as e:
            import traceback
            self.log_queue.put(("error", f"{str(e)}\n{traceback.format_exc()}"))
        finally:
            self.is_running = False

    def run_simulation_thread(self, p):
        try:
            start_time = time.perf_counter()
            self.log_queue.put(("log", f"Initializing Simulation with Strategy: {p['strategy']}\n"))
            sim = Simulation()
            sim.setup_with_params(p)
            progress_state = {
                "last_emit_at": start_time,
                "last_partial_sim_time": -1.0
            }
            
            # Setup callback for real-time monitoring
            def live_update_cb(at_time):
                # システム情報の取得 (psutil)
                cpu = psutil.cpu_percent() if psutil else 0.0
                mem = psutil.virtual_memory().percent if psutil else 0.0
                
                now_perf = time.perf_counter()
                elapsed = now_perf - start_time
                progress = min(100.0, (at_time / p['steps']) * 100)
                
                if progress >= 100.0 or now_perf - progress_state["last_emit_at"] >= 0.25:
                    progress_state["last_emit_at"] = now_perf
                    self.log_queue.put(("progress", (0, progress, elapsed, cpu, mem)))

                # Send current state periodically
                if (at_time - progress_state["last_partial_sim_time"] >= 100) or progress >= 100.0:
                    progress_state["last_partial_sim_time"] = at_time
                    partial_kpis = sim.kpis.copy()
                    # MeetingGateのコストなどを集計 (Partial)
                    for node in sim.engine.nodes.values():
                        if hasattr(node, 'total_cost'):
                            partial_kpis['dr_cost'] += node.total_cost
                    
                    partial_tech_status = sim.get_tech_status()
                    partial_metrics = {
                        'tech_history': list(sim.tech_history),
                        'wip_history': list(sim.engine.results["wip_history"])
                    }
                    self.log_queue.put(("partial_result", (partial_kpis, partial_tech_status, partial_metrics)))

            sim.gui_callback = live_update_cb
            
            self.log_queue.put(("log", f"Running for {p['steps']} days...\n"))
            kpis = sim.run(steps=p['steps'])
            tech_status = sim.get_tech_status()
            
            final_elapsed = time.perf_counter() - start_time
            final_cpu = psutil.cpu_percent() if psutil else 0.0
            final_mem = psutil.virtual_memory().percent if psutil else 0.0
            self.log_queue.put(("progress", (0, 100.0, final_elapsed, final_cpu, final_mem)))
            self.log_queue.put(("scenario_done", (0, final_elapsed)))

            # Metrics calculation for visualization
            from analysis.metrics import calculate_metrics
            nodes_stats = []
            for node_id, node in sim.engine.nodes.items():
                if hasattr(node, 'get_stats'):
                    nodes_stats.append(node.get_stats())
            
            metrics = calculate_metrics(
                sim.engine.results["completed_jobs"],
                nodes_stats,
                sim.engine.now,
                sim.engine.results["wip_history"]
            )
            metrics['tech_history'] = sim.tech_history
            metrics['wip_history'] = sim.engine.results["wip_history"]
            
            self.last_sim = sim
            self.last_params = p
            
            self.log_queue.put(("result", (kpis, p, tech_status, metrics)))
            self.log_queue.put(("log", f"Simulation completed in {final_elapsed:.2f}s.\n"))
        except Exception as e:
            import traceback
            self.log_queue.put(("error", f"{str(e)}\n{traceback.format_exc()}"))
        finally:
            self.is_running = False

    def start_batch_simulation(self):
        if self.is_running: return
        self.is_running = True
        self.batch_btn.config(state=tk.DISABLED)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "DOE一括実行開始 (エンジニア数を変えて5回試行)...\n")
        
        params = self.get_parsed_params()
        thread = threading.Thread(target=self.run_batch_thread, args=(params,))
        thread.daemon = True
        thread.start()

    def run_batch_thread(self, p):
        results = []
        try:
            base_n = p['res_n_servers']
            for i in range(5):
                current_p = p.copy()
                current_p['res_n_servers'] = base_n + i
                current_p['scenario_id'] = f"DOE_ResEng_{current_p['res_n_servers']}"
                
                self.log_queue.put(("log", f"Batch {i+1}/5: Res Engineers = {current_p['res_n_servers']}\n"))
                sim = Simulation()
                sim.setup_with_params(current_p)
                kpis = sim.run(steps=p['steps'])
                
                res = kpis.copy()
                res.update({'res_n_servers': current_p['res_n_servers']})
                results.append(res)
            
            df = pd.DataFrame(results)
            output_path = 'reports/batch_results.csv'
            df.to_csv(output_path, index=False)
            self.log_queue.put(("log", f"Batch complete. Results saved to {output_path}\n"))
            
            # 最後の結果を表示
            tech_status = sim.get_tech_status()
            from analysis.metrics import calculate_metrics
            nodes_stats = [node.get_stats() for node in sim.engine.nodes.values() if hasattr(node, 'get_stats')]
            metrics = calculate_metrics(
                sim.engine.results["completed_jobs"],
                nodes_stats,
                sim.engine.now,
                sim.engine.results["wip_history"]
            )
            self.last_sim = sim
            self.last_params = p
            
            self.log_queue.put(("result", (kpis, p, tech_status, metrics)))
            
        except Exception as e:
            self.log_queue.put(("error", str(e)))
        finally:
            self.is_running = False

    def periodic_check(self):
        while not self.log_queue.empty():
            msg_type, data = self.log_queue.get()
            if msg_type == "log":
                self.results_text.insert(tk.END, data)
                self.results_text.see(tk.END)
            elif msg_type == "progress":
                idx, progress, elapsed, cpu, mem = data
                if idx < len(self.progress_bars):
                    self.progress_values[idx] = max(self.progress_values[idx], progress)
                    self.elapsed_times[idx] = max(self.elapsed_times[idx], elapsed)
                    if progress >= 100.0:
                        self.scenario_states[idx] = "done"
                    self._update_progress_widgets(idx)
                    self._maybe_log_progress(idx)
                if cpu > 0:
                    self.cpu_mem_lbl.config(text=f"CPU: {cpu:.1f}% | Mem: {mem:.1f}%")
            elif msg_type == "scenario_done":
                idx, elapsed = data
                if idx < len(self.progress_bars):
                    self.elapsed_times[idx] = max(self.elapsed_times[idx], elapsed)
                    self.progress_values[idx] = 100.0
                    self.scenario_states[idx] = "done"
                    self._update_progress_widgets(idx)
            elif msg_type == "partial_result":
                kpis, tech_status, metrics = data
                self.pending_single_plot = (kpis, tech_status, metrics)
                self._schedule_plot_refresh("single")
            elif msg_type == "comp_partial_result":
                label, kpis, params, tech_status, metrics = data
                self.comparison_data[label] = (kpis, params, tech_status, metrics)
                if len(self.comparison_data) >= 1: # Update even if only one has progress
                    self.pending_comparison_plot = dict(self.comparison_data)
                    self._schedule_plot_refresh("comparison")
            elif msg_type == "result":
                kpis, params, tech_status, metrics = data
                self.display_results(kpis, params, tech_status, metrics)
            elif msg_type == "comp_result":
                self.display_comparison_results(data)
            elif msg_type == "error":
                messagebox.showerror("Simulation Error", data)
                self.results_text.insert(tk.END, f"\nERROR: {data}\n")
        self._refresh_running_elapsed_labels()
        
        if not self.is_running:
            self.run_btn.config(state=tk.NORMAL)
            self.batch_btn.config(state=tk.NORMAL)
            
        self.root.after(100, self.periodic_check)

    def _format_progress_label(self, idx):
        state = self.scenario_states[idx]
        suffix = ""
        if state == "done":
            suffix = " ✓"
        elif state == "running":
            suffix = " ..."
        return f"Scenario {chr(65+idx)}: {self.progress_values[idx]:.1f}% ({self.elapsed_times[idx]:.1f}s){suffix}"

    def _update_progress_widgets(self, idx):
        self.progress_bars[idx]['value'] = self.progress_values[idx]
        self.progress_labels[idx].config(text=self._format_progress_label(idx))

    def _refresh_running_elapsed_labels(self):
        now_perf = time.perf_counter()
        for idx, started_at in enumerate(self.start_times):
            if self.scenario_states[idx] != "running" or started_at is None:
                continue
            self.elapsed_times[idx] = max(self.elapsed_times[idx], now_perf - started_at)
            self._update_progress_widgets(idx)

    def _maybe_log_progress(self, idx):
        progress = self.progress_values[idx]
        elapsed = self.elapsed_times[idx]
        now_perf = time.perf_counter()
        should_log = False

        if progress >= self.progress_log_next[idx]:
            should_log = True
            while self.progress_log_next[idx] <= progress:
                self.progress_log_next[idx] += 10.0
        elif now_perf - self.last_progress_log_at[idx] >= 5.0 and progress > 0.0 and self.scenario_states[idx] == "running":
            should_log = True

        if should_log:
            self.last_progress_log_at[idx] = now_perf
            self.results_text.insert(
                tk.END,
                f"Scenario {chr(65+idx)} progress: {progress:.1f}% / elapsed {elapsed:.1f}s\n"
            )
            self.results_text.see(tk.END)

    def _schedule_plot_refresh(self, mode):
        self.pending_plot_mode = mode
        if self.plot_render_in_progress:
            self.plot_refresh_requested = True
            return

        if self.plot_refresh_job is not None:
            return

        elapsed_ms = int((time.perf_counter() - self.last_plot_refresh_at) * 1000)
        delay = max(200, self.plot_refresh_interval_ms - elapsed_ms)
        self.plot_refresh_job = self.root.after(delay, self._render_pending_plots)

    def _render_pending_plots(self):
        self.plot_refresh_job = None
        if self.plot_render_in_progress:
            self.plot_refresh_requested = True
            return

        mode = self.pending_plot_mode
        self.pending_plot_mode = None
        self.plot_render_in_progress = True
        try:
            self.last_plot_refresh_at = time.perf_counter()
            if mode == "comparison" and self.pending_comparison_plot is not None:
                self.update_comparison_plots(self.pending_comparison_plot)
            elif mode == "single" and self.pending_single_plot is not None:
                kpis, tech_status, metrics = self.pending_single_plot
                self.update_plots(kpis, tech_status, metrics)
        finally:
            self.plot_render_in_progress = False
            if self.plot_refresh_requested:
                self.plot_refresh_requested = False
                self._schedule_plot_refresh(mode or "comparison")

    def output_viz_report(self):
        # 直近の結果に基づいて、単体/比較どちらのケースでも全シナリオ分のレポートを出力する
        try:
            from core.scenario import ScenarioManager
            manager = ScenarioManager()

            scenarios = []  # [(label, params, sim, metrics)]

            # 1) 直近の表示結果が比較(dict)ならそれを優先
            if isinstance(getattr(self, 'last_results', None), dict) and self.last_results:
                for label, tup in self.last_results.items():
                    # tup is (kpis, params, tech_status, metrics)
                    if isinstance(tup, (list, tuple)) and len(tup) == 4:
                        _, params, _ts, metrics = tup
                        scenarios.append((label, params, None, metrics))

            # 2) 比較データのバッファがあれば補完（上と重複しないよう不足分のみ）
            if getattr(self, 'comparison_data', None):
                for label, tup in self.comparison_data.items():
                    if any(lbl == label for lbl, *_ in scenarios):
                        continue
                    if isinstance(tup, (list, tuple)) and len(tup) == 4:
                        _, params, _ts, metrics = tup
                        scenarios.append((label, params, None, metrics))

            # 3) 単体実行のみのケース（last_sim/last_params）
            if not scenarios and hasattr(self, 'last_sim') and hasattr(self, 'last_params'):
                sim = self.last_sim
                params = self.last_params
                try:
                    from analysis.metrics import calculate_metrics
                    nodes_stats = [n.get_stats() for n in sim.engine.nodes.values() if hasattr(n, 'get_stats')]
                    metrics = calculate_metrics(
                        sim.engine.results["completed_jobs"],
                        nodes_stats,
                        sim.engine.now,
                        sim.engine.results["wip_history"]
                    )
                    metrics['tech_history'] = getattr(sim, 'tech_history', [])
                except Exception:
                    metrics = getattr(self, 'last_metrics', {}) if hasattr(self, 'last_metrics') else {}
                scenarios.append((params.get('scenario_id', 'Single'), params, sim, metrics))

            if not scenarios:
                messagebox.showwarning("Warning", "シミュレーション結果が見つかりません。先に実行してください。")
                return

            # 出力実行
            generated = []
            for label, params, sim, metrics in scenarios:
                scenario_id = params.get('scenario_id', label) if isinstance(params, dict) else (label or 'result')
                # 可視化に最低限必要なキーをチェック（不足していても可能な範囲で実行）
                try:
                    manager.visualize_single(sim, metrics or {}, scenario_id)
                    generated.append(scenario_id)
                except Exception as e:
                    # 個別失敗は続行し、最後にまとめて通知
                    self.results_text.insert(tk.END, f"Visualization failed for {scenario_id}: {e}\n")
                    self.results_text.see(tk.END)

                # Ver2: 3D静止画/動画の生成（オプション）
                try:
                    if self.export_video_var.get():
                        # 出力先はPlan7配下のreports/vizに統一
                        base_dir = os.path.dirname(os.path.abspath(__file__))
                        viz_dir = os.path.join(base_dir, "reports", "viz")
                        videos_dir = os.path.join(viz_dir, "videos")
                        os.makedirs(viz_dir, exist_ok=True)
                        os.makedirs(videos_dir, exist_ok=True)
                        from analysis.viz3d import (
                            save_wip_surface_png, render_wip_surface_video,
                            save_leadtime_surface_png, render_leadtime_surface_video,
                            save_tech_diffusion_scatter_png, render_tech_diffusion_video,
                            save_network3d_png, render_network3d_video,
                        )
                        wip_hist = None
                        if isinstance(metrics, dict):
                            wip_hist = ((metrics.get('wip') or {}).get('history'))
                        if not wip_hist and sim is not None and hasattr(sim, 'engine'):
                            wip_hist = getattr(sim.engine, 'results', {}).get('wip_history')
                        frame_step = int(self.video_frame_step_var.get() or 1)
                        rotate_flag = bool(self.video_rotate_var.get())
                        try:
                            elev_val = float(self.video_elev_var.get())
                        except Exception:
                            elev_val = 25.0
                        try:
                            azim_val = float(self.video_azim_var.get())
                        except Exception:
                            azim_val = 45.0

                        # スタイル選択の取得
                        wip_style = (self.viz3d_wip_style_var.get() or "Default")
                        lt_style = (self.viz3d_lt_style_var.get() or "Default")
                        tech_style = (self.viz3d_tech_style_var.get() or "Default")
                        net_style = (self.viz3d_net_style_var.get() or "Default")

                        # 3D WIPサーフェス
                        if self.viz3d_wip_var.get() and wip_hist:
                            try:
                                png_out = os.path.join(viz_dir, f"{scenario_id}_wip_3d.png")
                                vid_out = os.path.join(videos_dir, f"{scenario_id}_wip_3d.mp4")
                                saved_png = save_wip_surface_png(wip_hist, png_out, elev=elev_val, azim=azim_val, style=wip_style)
                                actual_vid = render_wip_surface_video(wip_hist, vid_out, fps=10, rotate=rotate_flag, frame_step=frame_step, elev=elev_val, azim=azim_val, style=wip_style)
                                rel_png = os.path.relpath(saved_png, base_dir)
                                rel_vid = os.path.relpath(actual_vid, base_dir)
                                self.results_text.insert(tk.END, f"3D(WIP): {rel_png}, {rel_vid}\n")
                            except Exception as ex:
                                self.results_text.insert(tk.END, f"3D(WIP)生成失敗 [{scenario_id}]: {ex}\n")
                            self.results_text.see(tk.END)

                        # 3D リードタイム分解
                        if self.viz3d_lt_var.get():
                            try:
                                png_out = os.path.join(viz_dir, f"{scenario_id}_leadtime_3d.png")
                                vid_out = os.path.join(videos_dir, f"{scenario_id}_leadtime_3d.mp4")
                                saved_png = save_leadtime_surface_png(metrics or {}, png_out, elev=elev_val, azim=azim_val, style=lt_style)
                                actual_vid = render_leadtime_surface_video(metrics or {}, vid_out, fps=10, rotate=rotate_flag, frame_step=frame_step, elev=elev_val, azim=azim_val, style=lt_style)
                                rel_png = os.path.relpath(saved_png, base_dir)
                                rel_vid = os.path.relpath(actual_vid, base_dir)
                                self.results_text.insert(tk.END, f"3D(LT分解): {rel_png}, {rel_vid}\n")
                            except Exception as ex:
                                self.results_text.insert(tk.END, f"3D(LT分解)生成失敗 [{scenario_id}]: {ex}\n")
                            self.results_text.see(tk.END)

                        # 3D 技術成熟の拡散
                        if self.viz3d_tech_var.get():
                            try:
                                tech_hist = None
                                if isinstance(metrics, dict):
                                    tech_hist = metrics.get('tech_history') or getattr(self, 'last_tech_history', None)
                                if tech_hist is None:
                                    tech_hist = getattr(sim, 'tech_history', None) if sim is not None else None
                                png_out = os.path.join(viz_dir, f"{scenario_id}_tech_diff3d.png")
                                vid_out = os.path.join(videos_dir, f"{scenario_id}_tech_diff3d.mp4")
                                saved_png = save_tech_diffusion_scatter_png(tech_hist or [], png_out, elev=elev_val, azim=azim_val, style=tech_style)
                                actual_vid = render_tech_diffusion_video(tech_hist or [], vid_out, fps=10, rotate=rotate_flag, frame_step=frame_step, elev=elev_val, azim=azim_val, style=tech_style)
                                rel_png = os.path.relpath(saved_png, base_dir)
                                rel_vid = os.path.relpath(actual_vid, base_dir)
                                self.results_text.insert(tk.END, f"3D(技術拡散): {rel_png}, {rel_vid}\n")
                            except Exception as ex:
                                self.results_text.insert(tk.END, f"3D(技術拡散)生成失敗 [{scenario_id}]: {ex}\n")
                            self.results_text.see(tk.END)

                        # 立体ネットワーク図
                        if self.viz3d_net_var.get():
                            try:
                                png_out = os.path.join(viz_dir, f"{scenario_id}_network3d.png")
                                vid_out = os.path.join(videos_dir, f"{scenario_id}_network3d.mp4")
                                saved_png = save_network3d_png(wip_hist or [], png_out, elev=elev_val, azim=azim_val, style=net_style)
                                actual_vid = render_network3d_video(wip_hist or [], vid_out, fps=10, rotate=rotate_flag, frame_step=frame_step, elev=elev_val, azim=azim_val, style=net_style)
                                rel_png = os.path.relpath(saved_png, base_dir)
                                rel_vid = os.path.relpath(actual_vid, base_dir)
                                self.results_text.insert(tk.END, f"3D(ネットワーク): {rel_png}, {rel_vid}\n")
                            except Exception as ex:
                                self.results_text.insert(tk.END, f"3D(ネットワーク)生成失敗 [{scenario_id}]: {ex}\n")
                            self.results_text.see(tk.END)

                        # Ver2: 部署/ハンドオフ関連の簡易出力（PNG/CSV）
                        try:
                            base_dir = os.path.dirname(os.path.abspath(__file__))
                            viz_dir = os.path.join(base_dir, "reports", "viz")
                            os.makedirs(viz_dir, exist_ok=True)

                            # Handoffネットワーク（既存簡易3Dの静止画を流用）
                            cfg_flags = params if isinstance(params, dict) else (getattr(sim, 'params', {}) if sim is not None else {})
                            if (cfg_flags.get('consider_handoffs') or cfg_flags.get('consider_departments')):
                                from analysis.viz3d import save_network3d_png
                                png_out2 = os.path.join(viz_dir, f"{scenario_id}_dept_network.png")
                                saved_png2 = save_network3d_png(wip_hist or [], png_out2)
                                self.results_text.insert(tk.END, f"Dept/Net: {os.path.relpath(saved_png2, base_dir)}\n")
                                self.results_text.see(tk.END)

                            # KPI由来のSLA違反/ハンドオフ回数をCSV保存
                            if sim is not None and hasattr(sim, 'kpis'):
                                import csv
                                csv_out = os.path.join(viz_dir, f"{scenario_id}_dept_summary.csv")
                                with open(csv_out, 'w', newline='', encoding='utf-8') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(["metric", "value"]) 
                                    writer.writerow(["sla_violations", sim.kpis.get('sla_violations', 0)])
                                    writer.writerow(["handoff_events", sim.kpis.get('handoff_events', 0)])
                                    writer.writerow(["dept_cost_time", f"{sim.kpis.get('dept_cost_time', 0.0):.4f}"])
                                self.results_text.insert(tk.END, f"Dept Summary CSV: {os.path.relpath(csv_out, base_dir)}\n")
                                self.results_text.see(tk.END)
                        except Exception as ex2:
                            self.results_text.insert(tk.END, f"部署/ハンドオフ出力に失敗 [{scenario_id}]: {ex2}\n")
                            self.results_text.see(tk.END)
                except Exception as e:
                    self.results_text.insert(tk.END, f"3D/動画の生成に失敗しました [{scenario_id}]: {e}\n")
                    self.results_text.see(tk.END)

            if generated:
                # 複数シナリオの出力先を簡潔に案内
                examples = ", ".join([f"{sid}_summary.png" for sid in generated[:3]])
                more = "" if len(generated) <= 3 else f" ほか{len(generated)-3}件"
                messagebox.showinfo(
                    "Success",
                    f"詳細レポートを出力しました（各シナリオ）：\nreports/viz/{examples}{more}"
                )
            else:
                messagebox.showwarning("Warning", "レポート出力に失敗しました。ログを確認してください。")
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")

    def _refresh_plots(self):
        # Helper to trigger plot update manually if needed
        if hasattr(self, 'last_results'):
             if isinstance(self.last_results, dict):
                 self.pending_comparison_plot = dict(self.last_results)
                 self._schedule_plot_refresh("comparison")
             else:
                 kpis, tech_status, metrics = self.last_results
                 self.pending_single_plot = (kpis, tech_status, metrics)
                 self._schedule_plot_refresh("single")
        elif self.comparison_data:
             self.pending_comparison_plot = dict(self.comparison_data)
             self._schedule_plot_refresh("comparison")

    def display_results(self, kpis, params, tech_status, metrics):
        self.results_text.insert(tk.END, f"\n--- Final KPIs: {params['scenario_id']} ---\n")
        self.results_text.insert(tk.END, f"  Gain: {kpis['total_gain']:.2f} / Completed: {kpis.get('completed_jobs', 0)}\n")
        self.results_text.insert(tk.END, f"  Experiments: {kpis['total_experiments']} (Tech Fail: {kpis['technical_failures']})\n")
        self.results_text.see(tk.END)
        self.last_results = (kpis, tech_status, metrics)
        self.update_plots(kpis, tech_status, metrics)

    def display_comparison_results(self, results):
        self.results_text.insert(tk.END, f"\n--- Comparison Results ---\n")
        for label, (kpis, params, tech_status, metrics) in results.items():
            self.results_text.insert(tk.END, f"[{label} ({params['scenario_id']})]:\n")
            self.results_text.insert(tk.END, f"  Gain: {kpis['total_gain']:.2f} / Completed: {kpis.get('completed_jobs', 0)}\n")
            self.results_text.insert(tk.END, f"  Experiments: {kpis['total_experiments']} (Tech Fail: {kpis['technical_failures']})\n")
        self.results_text.see(tk.END)
        self.last_results = results
        self.update_comparison_plots(results)

    def update_plots(self, kpis, tech_status, metrics):
        # Indicate rendering status
        self.results_text.insert(tk.END, "単一シナリオ・グラフ描画中... / Rendering charts...\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

        # Clear previous plots
        for canvas in self.fig_canvases:
            canvas.get_tk_widget().destroy()
        self.fig_canvases = []

        # List of plot functions to call
        plot_configs = []
        if self.plot_visibility['kpi'].get():
            plot_configs.append((self._plot_kpi_summary, (kpis,)))
        if self.plot_visibility['tech_status'].get():
            plot_configs.append((self._plot_tech_status, (tech_status,)))
        if self.plot_visibility['tech_history'].get():
            plot_configs.append((self._plot_tech_history, (metrics.get('tech_history', []),)))
        if self.plot_visibility['wip'].get():
            plot_configs.append((self._plot_wip_heatmap, (metrics.get('wip_history', []),)))
        if self.plot_visibility['lead_time'].get():
            plot_configs.append((self._plot_lead_time_dist, (metrics.get('raw_lead_times', []),)))
        if self.plot_visibility['gate_waits'].get():
            plot_configs.append((self._plot_gate_waits, (metrics.get('gate_stats', []),)))
        if self.plot_visibility['resource_util'].get():
            plot_configs.append((self._plot_resource_utilization, (metrics.get('wip_history', []),)))
        if self.plot_visibility['cost_trend'].get():
            plot_configs.append((self._plot_cost_trend, (metrics.get('tech_history', []),)))
        if self.plot_visibility['lt_breakdown'].get():
            plot_configs.append((self._plot_lead_time_breakdown, (metrics.get('loss', {}),)))

        cols = self.layout_cols.get()
        for i, (plot_func, args) in enumerate(plot_configs):
            fig = plot_func(*args)
            if fig:
                # Adjust figure size for columns
                if cols > 1:
                    fig.set_size_inches(fig.get_size_inches()[0]*0.8, fig.get_size_inches()[1])

                canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
                canvas.draw()
                widget = canvas.get_tk_widget()
                
                row = i // cols
                col = i % cols
                widget.grid(row=row, column=col, sticky="nsew", padx=5, pady=10)
                self.scrollable_frame.columnconfigure(col, weight=1)
                self.fig_canvases.append(canvas)
        
        # Update scrollregion
        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_comparison_plots(self, results):
        # Indicate rendering status
        self.results_text.insert(tk.END, "比較グラフ描画中... / Rendering comparison charts...\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

        # Clear previous plots
        for canvas in self.fig_canvases:
            canvas.get_tk_widget().destroy()
        self.fig_canvases = []

        # List of comparison plots
        plot_figs = []
        if self.plot_visibility['kpi'].get():
            plot_figs.append(self._plot_kpi_comparison(results))

        if self.plot_visibility['tech_status'].get():
            plot_figs.append(self._plot_tech_status_comparison(results))
        
        if self.plot_visibility['tech_history'].get():
            plot_figs.append(self._plot_tech_history_comparison(results))
            
        if self.plot_visibility['wip'].get():
             plot_figs.append(self._plot_wip_comparison(results))
        if self.plot_visibility['resource_util'].get():
             plot_figs.append(self._plot_resource_utilization_comparison(results))
        if self.plot_visibility['cost_trend'].get():
             plot_figs.append(self._plot_cost_trend_comparison(results))
        if self.plot_visibility['lt_breakdown'].get():
             plot_figs.append(self._plot_lead_time_breakdown_comparison(results))

        cols = self.layout_cols.get()
        for i, fig in enumerate(plot_figs):
            if fig:
                canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
                canvas.draw()
                widget = canvas.get_tk_widget()
                
                row = i // cols
                col = i % cols
                widget.grid(row=row, column=col, sticky="nsew", padx=5, pady=10)
                self.scrollable_frame.columnconfigure(col, weight=1)
                self.fig_canvases.append(canvas)

        # Update scrollregion
        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _add_fig_to_canvas(self, fig):
        # Not used anymore with grid layout, but keep for compatibility if needed
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
            self.fig_canvases.append(canvas)

    def _plot_wip_comparison(self, results):
        fig = plt.Figure(figsize=(12, 5), dpi=90)
        labels = list(results.keys())
        if not labels: return None
        
        for i, label in enumerate(labels):
            ax = fig.add_subplot(1, len(labels), i+1)
            metrics = results[label][3]
            wip_history = metrics.get('wip_history', [])
            if not wip_history: continue
            
            data = []
            for h in wip_history:
                for node_id, wip in h['node_wip'].items():
                    data.append({"time": h['time'], "node": node_id, "wip": wip})
            
            df = pd.DataFrame(data)
            if df.empty: continue
            df_pivot = df.pivot(index="node", columns="time", values="wip")
            sns.heatmap(df_pivot, cmap="YlOrRd", ax=ax, cbar=(i == len(labels)-1))
            ax.set_title(f"WIP: {label}")
        
        fig.tight_layout()
        return fig

    def _plot_resource_utilization_comparison(self, results):
        fig = plt.Figure(figsize=(12, 5), dpi=90)
        labels = list(results.keys())
        if not labels: return None
        for i, label in enumerate(labels):
            ax = fig.add_subplot(1, len(labels), i+1)
            metrics = results[label][3]
            wip_history = metrics.get('wip_history', [])
            if not wip_history: continue
            data = []
            for h in wip_history:
                if 'node_busy' not in h: continue
                for node_id, busy in h['node_busy'].items():
                    data.append({"time": h['time'], "node": node_id, "busy": busy})
            if not data: continue
            df = pd.DataFrame(data)
            sns.lineplot(data=df, x="time", y="busy", hue="node", ax=ax)
            ax.set_title(f"Utilization: {label}")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_cost_trend_comparison(self, results):
        fig = plt.Figure(figsize=(12, 5), dpi=90)
        labels = list(results.keys())
        if not labels: return None
        ax = fig.add_subplot(111)
        colors = ['red', 'blue', 'green']
        for i, label in enumerate(labels):
            metrics = results[label][3]
            tech_history = metrics.get('tech_history', [])
            if not tech_history: continue
            times = [h['time'] for h in tech_history]
            costs = [h.get('cumulative_cost', 0.0) for h in tech_history]
            ax.plot(times, costs, label=f"Cost: {label}", color=colors[i % len(colors)])
            ax.fill_between(times, costs, alpha=0.1, color=colors[i % len(colors)])
        ax.set_title("Cumulative Cost Comparison")
        ax.set_ylabel("Cost")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_lead_time_breakdown_comparison(self, results):
        """
        シナリオ間の平均リードタイム内訳（実作業/待機/判定待ち）を比較表示する。
        部分結果（計算途中）では metrics['loss'] が未計算のため、ゼロ埋めになることがある。
        その場合でも「データ待ち」プレースホルダを描画して、
        “表示されない” と見えないようにする。
        """
        labels = list(results.keys())
        if not labels:
            return None

        # まずデータを収集（loss が無い場合は 0 で埋める）
        rows = []
        have_primary_any = False
        for label in labels:
            try:
                metrics = results[label][3] if isinstance(results[label], (list, tuple)) and len(results[label]) >= 4 else {}
            except Exception:
                metrics = {}
            primary = (
                metrics.get('loss', {})
                .get('time', {})
                .get('primary', {})
            ) or {}
            if primary:
                have_primary_any = True
            rows.append({
                'Scenario': label,
                'Work': float(primary.get('avg_work', 0.0) or 0.0),
                'Wait': float(primary.get('avg_wait', 0.0) or 0.0),
                'Decision': float(primary.get('avg_decision', 0.0) or 0.0),
            })

        df = pd.DataFrame(rows)

        # 図の生成（常に返す。ゼロのみのときはプレースホルダ文言を入れる）
        fig = plt.Figure(figsize=(12, 5), dpi=90)
        ax = fig.add_subplot(111)

        if df.empty:
            ax.text(0.5, 0.5, 'No scenarios to compare', ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return fig

        value_cols = ['Work', 'Wait', 'Decision']

        # primary が一つでも生成済みなら、ゼロでも棒グラフを描画
        if have_primary_any:
            df.set_index('Scenario')[value_cols].plot(
                kind='bar', stacked=True, ax=ax,
                color=['skyblue', 'orange', 'lightgreen']
            )
            ax.set_ylabel('Avg Days / 平均日数')
            ax.legend(loc='upper right')
        else:
            # プレースホルダ表示（最終メトリクス計算前 or データ無し）
            ax.text(
                0.5, 0.5,
                '最終メトリクス計算待ち（loss/time/primary）\nWaiting for final metrics (loss/time/primary)',
                ha='center', va='center', fontsize=11
            )
            ax.set_axis_off()

        ax.set_title('Lead Time Breakdown Comparison / リードタイム内訳（比較）')
        fig.tight_layout()
        return fig

    def _plot_kpi_comparison(self, results):
        fig = plt.Figure(figsize=(8, 5), dpi=90)
        ax = fig.add_subplot(111)
        data = []
        for label, (kpis, _, _, _) in results.items():
            for k in ['total_gain', 'completed_jobs', 'total_experiments']:
                data.append({'Scenario': label, 'KPI': k, 'Value': kpis.get(k, 0)})
        df = pd.DataFrame(data)
        sns.barplot(data=df, x='KPI', y='Value', hue='Scenario', ax=ax, palette="viridis")
        ax.set_title("KPI Comparison")
        fig.tight_layout()
        return fig

    def _plot_tech_status_comparison(self, results):
        # We'll just show A and B side by side in two subplots
        fig = plt.Figure(figsize=(10, 5), dpi=90)
        labels = list(results.keys())
        for i, label in enumerate(labels):
            ax = fig.add_subplot(1, len(labels), i+1)
            tech_status = results[label][2]
            techs = list(tech_status.keys())
            maturities = [tech_status[t]['maturity'] for t in techs]
            uncertainties = [tech_status[t]['uncertainty'] for t in techs]
            x = np.arange(len(techs))
            width = 0.35
            ax.bar(x - width/2, maturities, width, label='Maturity', color='green')
            ax.bar(x + width/2, uncertainties, width, label='Uncertainty', color='orange')
            ax.set_xticks(x)
            ax.set_xticklabels(techs, rotation=15)
            ax.set_title(f"Status: {label}")
            if i == 0: ax.legend()
        fig.tight_layout()
        return fig

    def _plot_tech_history_comparison(self, results):
        fig = plt.Figure(figsize=(10, 6), dpi=90)
        ax1 = fig.add_subplot(211) # Maturity
        ax2 = fig.add_subplot(212) # Uncertainty
        
        linestyles = ['-', '--', ':', '-.']
        for i, (label, (_, _, _, metrics)) in enumerate(results.items()):
            tech_history = metrics.get('tech_history', [])
            if not tech_history: continue
            
            times = [h['time'] for h in tech_history]
            tech_names = tech_history[0]['tech_items'].keys()
            ls = linestyles[i % len(linestyles)]
            
            for name in tech_names:
                maturities = [h['tech_items'][name]['maturity'] for h in tech_history]
                uncertainties = [h['tech_items'][name]['uncertainty'] for h in tech_history]
                ax1.plot(times, maturities, label=f"{label}: {name}", linestyle=ls)
                ax2.plot(times, uncertainties, label=f"{label}: {name}", linestyle=ls)
                
        ax1.set_title("Maturity Evolution Comparison")
        ax2.set_title("Uncertainty Evolution Comparison")
        ax1.legend(fontsize='x-small', ncol=2)
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_kpi_summary(self, kpis):
        fig = plt.Figure(figsize=(6, 4), dpi=90)
        ax = fig.add_subplot(111)
        keys = ['total_gain', 'completed_jobs', 'total_experiments', 'technical_failures']
        values = [kpis.get(k, 0) for k in keys]
        sns.barplot(x=keys, y=values, hue=keys, ax=ax, palette="viridis", legend=False)
        ax.set_title("Key Performance Indicators")
        ax.tick_params(axis='x', rotation=15)
        fig.tight_layout()
        return fig

    def _plot_tech_status(self, tech_status):
        fig = plt.Figure(figsize=(6, 4), dpi=90)
        ax = fig.add_subplot(111)
        techs = list(tech_status.keys())
        maturities = [tech_status[t]['maturity'] for t in techs]
        uncertainties = [tech_status[t]['uncertainty'] for t in techs]
        
        x = np.arange(len(techs))
        width = 0.35
        ax.bar(x - width/2, maturities, width, label='Maturity', color='green')
        ax.bar(x + width/2, uncertainties, width, label='Uncertainty', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(techs)
        ax.set_title("Current Technology Status")
        ax.legend()
        fig.tight_layout()
        return fig

    def _plot_tech_history(self, tech_history):
        if not tech_history: return None
        fig = plt.Figure(figsize=(8, 5), dpi=90)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        times = [h['time'] for h in tech_history]
        tech_names = tech_history[0]['tech_items'].keys()
        
        for name in tech_names:
            maturities = [h['tech_items'][name]['maturity'] for h in tech_history]
            uncertainties = [h['tech_items'][name]['uncertainty'] for h in tech_history]
            ax1.plot(times, maturities, label=name)
            ax2.plot(times, uncertainties, label=name, linestyle='--')
            
        ax1.set_title("Maturity Evolution")
        ax1.set_ylabel("Maturity")
        ax1.grid(True, alpha=0.3)
        ax2.set_title("Uncertainty Evolution")
        ax2.set_ylabel("Uncertainty")
        ax2.set_xlabel("Time (Days)")
        ax2.grid(True, alpha=0.3)
        ax1.legend(fontsize='small', loc='upper left')
        fig.tight_layout()
        return fig

    def _plot_wip_heatmap(self, wip_history):
        if not wip_history: return None
        fig = plt.Figure(figsize=(10, 5), dpi=90)
        ax = fig.add_subplot(111)
        
        data = []
        for h in wip_history:
            for node_id, wip in h['node_wip'].items():
                data.append({"time": h['time'], "node": node_id, "wip": wip})
        
        df = pd.DataFrame(data)
        if df.empty: return None
        df_pivot = df.pivot(index="node", columns="time", values="wip")
        
        sns.heatmap(df_pivot, cmap="YlOrRd", ax=ax, cbar_kws={'label': 'WIP'})
        ax.set_title("WIP Heatmap over Time")
        fig.tight_layout()
        return fig

    def _plot_lead_time_dist(self, lead_times):
        if not lead_times: return None
        fig = plt.Figure(figsize=(6, 4), dpi=90)
        ax = fig.add_subplot(111)
        sns.violinplot(y=lead_times, ax=ax, color="skyblue")
        ax.set_title("Lead Time Distribution (Completed Projects)")
        ax.set_ylabel("Days")
        fig.tight_layout()
        return fig

    def _plot_gate_waits(self, gate_stats):
        if not gate_stats: return None
        fig = plt.Figure(figsize=(6, 4), dpi=90)
        ax = fig.add_subplot(111)
        names = [s["node_id"] for s in gate_stats]
        waits = [s["avg_wait_time"] for s in gate_stats]
        sns.barplot(x=names, y=waits, hue=names, ax=ax, palette="magma", legend=False)
        ax.set_title("Average Wait Time per Gate")
        ax.set_ylabel("Days")
        fig.tight_layout()
        return fig

    def _plot_resource_utilization(self, wip_history):
        if not wip_history: return None
        fig = plt.Figure(figsize=(8, 4), dpi=90)
        ax = fig.add_subplot(111)
        
        data = []
        for h in wip_history:
            if 'node_busy' not in h: continue
            for node_id, busy in h['node_busy'].items():
                data.append({"time": h['time'], "node": node_id, "busy": busy})
        
        if not data: return None
        df = pd.DataFrame(data)
        sns.lineplot(data=df, x="time", y="busy", hue="node", ax=ax)
        ax.set_title("Resource Utilization (Busy Count)")
        ax.set_ylabel("Number of Busy Servers")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_cost_trend(self, tech_history):
        if not tech_history: return None
        fig = plt.Figure(figsize=(8, 4), dpi=90)
        ax = fig.add_subplot(111)
        
        times = [h['time'] for h in tech_history]
        costs = [h.get('cumulative_cost', 0.0) for h in tech_history]
        
        ax.plot(times, costs, color='red', label='Cumulative DR Cost')
        ax.fill_between(times, costs, alpha=0.2, color='red')
        ax.set_title("Cumulative Cost Trend")
        ax.set_ylabel("Cost (Arbitrary Unit)")
        ax.set_xlabel("Time (Days)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_lead_time_breakdown(self, metrics_loss):
        if not metrics_loss or 'time' not in metrics_loss: return None
        time_metrics = metrics_loss['time']
        primary = time_metrics.get('primary', {})
        if not primary: return None
        
        fig = plt.Figure(figsize=(6, 4), dpi=90)
        ax = fig.add_subplot(111)
        
        labels = ['Primary Projects']
        work = [primary.get('avg_work', 0)]
        wait = [primary.get('avg_wait', 0)]
        decision = [primary.get('avg_decision', 0)]
        
        ax.bar(labels, work, label='Work Time', color='skyblue')
        ax.bar(labels, wait, bottom=work, label='Wait Time', color='orange')
        ax.bar(labels, decision, bottom=[w+wt for w, wt in zip(work, wait)], label='Decision Latency', color='lightgreen')
        
        ax.set_title("Average Lead Time Breakdown")
        ax.set_ylabel("Days")
        ax.legend()
        fig.tight_layout()
        return fig

    def save_results_to_csv(self, kpis, params):
        df = pd.DataFrame([kpis])
        df.to_csv(f'reports/result_{params["scenario_id"]}.csv', index=False)

    def save_params_json(self):
        all_scenarios = []
        for i in range(3):
            all_scenarios.append(self.get_parsed_params(self.scenario_params[i]))
            
        filename = filedialog.asksaveasfilename(
            initialdir="configs",
            defaultextension=".json",
            title="設定を保存",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump({"scenarios": all_scenarios}, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("Success", f"Settings saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def load_params_json(self):
        filename = filedialog.askopenfilename(
            initialdir="configs",
            title="設定を読込",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                scenarios = data.get("scenarios", [])
                for i, p in enumerate(scenarios):
                    if i >= 3: break
                    for key, val in p.items():
                        if key in self.scenario_params[i]:
                            self.scenario_params[i][key].set(val)
                
                messagebox.showinfo("Success", f"Settings loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load settings: {str(e)}")

if __name__ == "__main__":
    if not os.path.exists('reports'): os.makedirs('reports')
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()
