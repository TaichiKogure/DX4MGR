import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import threading
import queue
import time

# Simulationクラスのインポート
try:
    from core.simulation import Simulation
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.simulation import Simulation

class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plan 6: Hybrid R&D Simulation Dashboard (v2.0)")
        self.root.geometry("1300x900")

        self.log_queue = queue.Queue()
        self.is_running = False
        
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

        # --- Left Frame: Tabs for Input ---
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.params = {}
        
        # Tab 1: Project Settings
        self.tab_project = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_project, text="基本設定")
        self.setup_project_tab()

        # Tab 2: Resource Settings
        self.tab_resource = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_resource, text="リソース")
        self.setup_resource_tab()

        # Tab 3: Tech/Gate Settings
        self.tab_tech = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_tech, text="技術・ゲート")
        self.setup_tech_tab()

        # Tab 4: Communication/DOE
        self.tab_comm = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_comm, text="通信・DOE")
        self.setup_comm_tab()

        # --- Action Buttons (Bottom of Left Frame) ---
        btn_frame = ttk.Frame(left_frame, padding=10)
        btn_frame.pack(fill=tk.X)

        self.run_btn = ttk.Button(btn_frame, text="シミュレーション実行", command=self.start_simulation)
        self.run_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.batch_btn = ttk.Button(btn_frame, text="DOE一括実行 (5回)", command=self.start_batch_simulation)
        self.batch_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.viz_btn = ttk.Button(btn_frame, text="詳細レポート出力", command=self.output_viz_report)
        self.viz_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # --- Right Frame: Dashboard ---
        # Top: KPI Results Text
        self.results_text = tk.Text(right_frame, height=15, width=60, font=("Courier", 10))
        self.results_text.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Bottom: Plot area with Scrollbar
        plot_wrapper = ttk.Frame(right_frame)
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

    def _on_mousewheel(self, event):
        if sys.platform == 'darwin':
            self.canvas.yview_scroll(int(-1 * (event.delta)), "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def add_entry(self, parent, label, key, default, entry_type="entry", values=None):
        row = getattr(parent, 'row_counter', 0)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3, padx=5)
        
        if entry_type == "entry":
            var = tk.StringVar(value=str(default))
            ent = ttk.Entry(parent, textvariable=var)
            ent.grid(row=row, column=1, sticky=tk.EW, pady=3, padx=5)
            self.params[key] = var
        elif entry_type == "check":
            var = tk.BooleanVar(value=default)
            chk = ttk.Checkbutton(parent, variable=var)
            chk.grid(row=row, column=1, sticky=tk.W, pady=3, padx=5)
            self.params[key] = var
        elif entry_type == "combo":
            var = tk.StringVar(value=default)
            cmb = ttk.Combobox(parent, textvariable=var, values=values)
            cmb.grid(row=row, column=1, sticky=tk.EW, pady=3, padx=5)
            self.params[key] = var
        elif entry_type == "slider":
            var = tk.DoubleVar(value=float(default))
            sld = ttk.Scale(parent, from_=values[0], to=values[1], variable=var, orient=tk.HORIZONTAL)
            sld.grid(row=row, column=1, sticky=tk.EW, pady=3, padx=5)
            # 値を隣に表示
            val_lbl = ttk.Label(parent, text=f"{default}")
            val_lbl.grid(row=row, column=2, sticky=tk.W)
            var.trace_add("write", lambda *args: val_lbl.config(text=f"{var.get():.1f}"))
            self.params[key] = var
            
        parent.row_counter = row + 1
        parent.columnconfigure(1, weight=1)

    def setup_project_tab(self):
        self.tab_project.row_counter = 0
        self.add_entry(self.tab_project, "シナリオID:", "scenario_id", "Plan6_GUI_v2")
        self.add_entry(self.tab_project, "シミュレーション日数:", "steps", 500)
        self.add_entry(self.tab_project, "実験戦略 (Strategy):", "strategy", "strategic", "combo", ["strategic", "random"])
        self.add_entry(self.tab_project, "デジタルツイン使用:", "use_digital_twin", True, "check")
        self.add_entry(self.tab_project, "乱数シード (Seed):", "seed", 42)

    def setup_resource_tab(self):
        self.tab_resource.row_counter = 0
        self.add_entry(self.tab_resource, "Research エンジニア数:", "res_n_servers", 5, "slider", [1, 20])
        self.add_entry(self.tab_resource, "Prototype エンジニア数:", "proto_n_servers", 3, "slider", [1, 20])
        self.add_entry(self.tab_resource, "Mass Production エンジニア数:", "mass_n_servers", 2, "slider", [1, 20])
        ttk.Separator(self.tab_resource, orient=tk.HORIZONTAL).grid(row=self.tab_resource.row_counter, column=0, columnspan=3, sticky=tk.EW, pady=10)
        self.tab_resource.row_counter += 1
        self.add_entry(self.tab_resource, "Research WIP制限:", "wip_limit_res", 2, "slider", [1, 10])
        self.add_entry(self.tab_resource, "Prototype WIP制限:", "wip_limit_proto", 1, "slider", [1, 10])

    def setup_tech_tab(self):
        self.tab_tech.row_counter = 0
        self.add_entry(self.tab_tech, "DR1 不確実性閾値:", "dr1_uncert_limit", 0.4, "slider", [0.0, 1.0])
        self.add_entry(self.tab_tech, "DR2 成熟度閾値:", "dr2_matur_limit", 0.5, "slider", [0.0, 1.0])
        self.add_entry(self.tab_tech, "DR3 成熟度閾値:", "dr3_matur_limit", 0.8, "slider", [0.0, 1.0])
        ttk.Separator(self.tab_tech, orient=tk.HORIZONTAL).grid(row=self.tab_tech.row_counter, column=0, columnspan=3, sticky=tk.EW, pady=10)
        self.tab_tech.row_counter += 1
        self.add_entry(self.tab_tech, "知識減衰率 (Decay):", "decay", 0.7, "slider", [0.0, 1.0])
        self.add_entry(self.tab_tech, "リワーク負荷係数:", "rework_load_factor", 0.5, "slider", [0.0, 1.0])
        self.add_entry(self.tab_tech, "最大リワーク回数:", "max_rework_cycles", 5, "slider", [1, 10])
        self.add_entry(self.tab_tech, "暗黙知度 (Tacitness):", "tacitness", 0.5, "slider", [0.0, 1.0])

    def setup_comm_tab(self):
        self.tab_comm.row_counter = 0
        self.add_entry(self.tab_comm, "Res->Proto 欠落率:", "loss_res_proto", 0.1, "slider", [0.0, 0.5])
        self.add_entry(self.tab_comm, "Proto->Ana 歪み率:", "dist_proto_ana", 0.05, "slider", [0.0, 0.5])
        self.add_entry(self.tab_comm, "Ana->Res 遅延:", "delay_ana_res", 2, "slider", [0, 10])
        ttk.Separator(self.tab_comm, orient=tk.HORIZONTAL).grid(row=self.tab_comm.row_counter, column=0, columnspan=3, sticky=tk.EW, pady=10)
        self.tab_comm.row_counter += 1
        self.add_entry(self.tab_comm, "DR1 会議周期 (日):", "dr1_period", 30, "slider", [7, 90])
        self.add_entry(self.tab_comm, "レビュー単価 (Cost):", "cost_per_review", 100, "slider", [10, 500])

    def get_parsed_params(self):
        p = {}
        for k, v in self.params.items():
            val = v.get()
            # 型変換
            if k in ['steps', 'seed', 'dr1_period']:
                p[k] = int(float(val))
            elif k in ['res_n_servers', 'proto_n_servers', 'mass_n_servers', 'wip_limit_res', 'wip_limit_proto', 'max_rework_cycles']:
                p[k] = int(float(val))
            elif isinstance(val, (float, int)) or (isinstance(val, str) and val.replace('.','',1).isdigit()):
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
        p['dr_threshold'] = 5.0 # Fixed for now
        p['past_logs_file'] = 'configs/past_logs.csv'
        return p

    def start_simulation(self):
        if self.is_running: return
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "シミュレーション開始中...\n")
        
        params = self.get_parsed_params()
        thread = threading.Thread(target=self.run_simulation_thread, args=(params,))
        thread.daemon = True
        thread.start()

    def run_simulation_thread(self, p):
        try:
            self.log_queue.put(("log", f"Initializing Simulation with Strategy: {p['strategy']}\n"))
            sim = Simulation()
            sim.setup_with_params(p)
            
            self.log_queue.put(("log", f"Running for {p['steps']} days...\n"))
            kpis = sim.run(steps=p['steps'])
            tech_status = sim.get_tech_status()
            
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
            self.log_queue.put(("log", "Simulation completed successfully.\n"))
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
            elif msg_type == "result":
                kpis, params, tech_status, metrics = data
                self.display_results(kpis, params, tech_status, metrics)
            elif msg_type == "error":
                messagebox.showerror("Simulation Error", data)
                self.results_text.insert(tk.END, f"\nERROR: {data}\n")
        
        if not self.is_running:
            self.run_btn.config(state=tk.NORMAL)
            self.batch_btn.config(state=tk.NORMAL)
            
        self.root.after(100, self.periodic_check)

    def output_viz_report(self):
        if not hasattr(self, 'last_sim'):
            messagebox.showwarning("Warning", "シミュレーションを先に実行してください。")
            return
        
        try:
            from core.scenario import ScenarioManager
            manager = ScenarioManager()
            from analysis.metrics import calculate_metrics
            sim = self.last_sim
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
            
            scenario_id = self.last_params.get('scenario_id', 'gui_result')
            manager.visualize_single(sim, metrics, scenario_id)
            
            messagebox.showinfo("Success", f"詳細レポートを出力しました:\nreports/viz/{scenario_id}_summary.png 等")
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")

    def display_results(self, kpis, params, tech_status, metrics):
        self.results_text.insert(tk.END, f"\n--- Final KPIs: {params['scenario_id']} ---\n")
        self.results_text.insert(tk.END, f"  Gain: {kpis['total_gain']:.2f} / Completed: {kpis.get('completed_jobs', 0)}\n")
        self.results_text.insert(tk.END, f"  Experiments: {kpis['total_experiments']} (Tech Fail: {kpis['technical_failures']})\n")
        self.results_text.see(tk.END)
        self.update_plots(kpis, tech_status, metrics)

    def update_plots(self, kpis, tech_status, metrics):
        # Clear previous plots
        for canvas in self.fig_canvases:
            canvas.get_tk_widget().destroy()
        self.fig_canvases = []

        # List of plot functions to call
        plot_configs = [
            (self._plot_kpi_summary, (kpis,)),
            (self._plot_tech_status, (tech_status,)),
            (self._plot_tech_history, (metrics.get('tech_history', []),)),
            (self._plot_wip_heatmap, (metrics.get('wip_history', []),)),
            (self._plot_lead_time_dist, (metrics.get('raw_lead_times', []),)),
            (self._plot_gate_waits, (metrics.get('gate_stats', []),))
        ]

        for plot_func, args in plot_configs:
            fig = plot_func(*args)
            if fig:
                canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X, expand=True, pady=10)
                self.fig_canvases.append(canvas)
        
        # Update scrollregion
        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _plot_kpi_summary(self, kpis):
        fig = plt.Figure(figsize=(6, 4), dpi=90)
        ax = fig.add_subplot(111)
        keys = ['total_gain', 'completed_jobs', 'total_experiments', 'technical_failures']
        values = [kpis.get(k, 0) for k in keys]
        sns.barplot(x=keys, y=values, ax=ax, palette="viridis")
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
        sns.barplot(x=names, y=waits, ax=ax, palette="magma")
        ax.set_title("Average Wait Time per Gate")
        ax.set_ylabel("Days")
        fig.tight_layout()
        return fig

    def save_results_to_csv(self, kpis, params):
        # Already handled by run_batch_thread or separate call
        pass

if __name__ == "__main__":
    if not os.path.exists('reports'): os.makedirs('reports')
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()
