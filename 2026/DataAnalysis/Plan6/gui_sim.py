import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# Simulationクラスのインポート
try:
    from core.simulation import Simulation
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.simulation import Simulation

class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plan 6: Hybrid R&D Simulation Dashboard")
        self.root.geometry("1200x850")

        self.setup_ui()

    def setup_ui(self):
        # Main frames
        left_frame = ttk.LabelFrame(self.root, text="入力パラメータ (Plan 6 統合設定)", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right_frame = ttk.LabelFrame(self.root, text="ダッシュボード (Dashboard)", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Input variables
        self.params = {}
        self.row_counter = 0
        
        def add_entry(label, key, default, entry_type="entry"):
            ttk.Label(left_frame, text=label).grid(row=self.row_counter, column=0, sticky=tk.W, pady=2)
            if entry_type == "entry":
                var = tk.StringVar(value=str(default))
                ent = ttk.Entry(left_frame, textvariable=var)
                ent.grid(row=self.row_counter, column=1, sticky=tk.EW, pady=2)
                self.params[key] = var
            elif entry_type == "check":
                var = tk.BooleanVar(value=default)
                chk = ttk.Checkbutton(left_frame, variable=var)
                chk.grid(row=self.row_counter, column=1, sticky=tk.W, pady=2)
                self.params[key] = var
            elif entry_type == "combo":
                var = tk.StringVar(value=default[0])
                cmb = ttk.Combobox(left_frame, textvariable=var, values=default)
                cmb.grid(row=self.row_counter, column=1, sticky=tk.EW, pady=2)
                self.params[key] = var
            self.row_counter += 1

        # 基本設定 (Plan5x)
        ttk.Label(left_frame, text="[基本設定]", font=('Helvetica', 10, 'bold')).grid(row=self.row_counter, column=0, columnspan=2, pady=5)
        self.row_counter += 1
        add_entry("シナリオID:", "scenario_id", "Plan6_Integrated")
        add_entry("シミュレーション日数:", "steps", 500)
        add_entry("実験戦略 (Strategy):", "strategy", ["strategic", "random"], "combo")
        add_entry("デジタルツイン使用:", "use_digital_twin", True, "check")

        # DES/ゲート設定 (Ver14)
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=self.row_counter, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.row_counter += 1
        ttk.Label(left_frame, text="[DES / ゲート制約]", font=('Helvetica', 10, 'bold')).grid(row=self.row_counter, column=0, columnspan=2, pady=5)
        self.row_counter += 1
        add_entry("DR1 周期 (日):", "dr1_period", 30)
        add_entry("リワーク負荷係数:", "rework_load_factor", 0.5)
        add_entry("最大リワーク回数:", "max_rework_cycles", 5)
        add_entry("知識減衰率 (Decay):", "decay", 0.7)

        # 組織間通信 (Plan5x)
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=self.row_counter, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.row_counter += 1
        ttk.Label(left_frame, text="[組織間通信品質]", font=('Helvetica', 10, 'bold')).grid(row=self.row_counter, column=0, columnspan=2, pady=5)
        self.row_counter += 1
        add_entry("Res->Proto 欠落率:", "loss_res_proto", 0.1)
        add_entry("Proto->Ana 歪み率:", "dist_proto_ana", 0.05)
        add_entry("Ana->Res 遅延:", "delay_ana_res", 2)

        # Output options
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=self.row_counter, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.row_counter += 1
        self.save_csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="結果をCSV保存 (reports/plan6_results.csv)", variable=self.save_csv_var).grid(row=self.row_counter, column=0, columnspan=2, sticky=tk.W)
        self.row_counter += 1

        run_btn = ttk.Button(left_frame, text="シミュレーション実行", command=self.run_simulation)
        run_btn.grid(row=self.row_counter, column=0, columnspan=2, pady=20)

        # Right frame components
        self.results_text = tk.Text(right_frame, height=12, width=60)
        self.results_text.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.plot_container = ttk.Frame(right_frame)
        self.plot_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig_canvas = None

    def run_simulation(self):
        try:
            p = {}
            for k, v in self.params.items():
                val = v.get()
                if k in ['steps', 'dr1_period', 'max_rework_cycles', 'delay_ana_res']:
                    p[k] = int(val)
                elif k in ['rework_load_factor', 'decay', 'loss_res_proto', 'dist_proto_ana']:
                    p[k] = float(val)
                else:
                    p[k] = val
            
            p['past_logs_file'] = 'configs/past_logs.csv'
            
            sim = Simulation()
            sim.setup_with_params(p)
            kpis = sim.run(steps=p['steps'])
            tech_status = sim.get_tech_status()
            
            self.display_results(kpis, p, tech_status)
            
            if self.save_csv_var.get():
                self.save_results_to_csv(kpis, p)
                
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Simulation failed: {str(e)}\n{traceback.format_exc()}")

    def display_results(self, kpis, params, tech_status):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"--- Scenario: {params['scenario_id']} (Plan 6 Hybrid) ---\n")
        self.results_text.insert(tk.END, f"【経営KPI】\n")
        self.results_text.insert(tk.END, f"  累計Gain: {kpis['total_gain']:.2f}\n")
        self.results_text.insert(tk.END, f"  完了ジョブ数: {kpis.get('completed_jobs', 0)}\n")
        self.results_text.insert(tk.END, f"  DR総コスト: {kpis.get('dr_cost', 0):.1f}\n")
        self.results_text.insert(tk.END, f"【開発KPI】\n")
        self.results_text.insert(tk.END, f"  総実験回数: {kpis['total_experiments']}\n")
        self.results_text.insert(tk.END, f"  技術的失敗: {kpis['technical_failures']} / 運用的失敗: {kpis['operational_failures']}\n")
        
        self.results_text.insert(tk.END, f"【技術成熟度 (Tech Status)】\n")
        for tech, data in tech_status.items():
            self.results_text.insert(tk.END, f"  - {tech}: 成熟度={data['maturity']:.2f}, 不確実性={data['uncertainty']:.2f}\n")

        self.update_plots(kpis, tech_status)

    def update_plots(self, kpis, tech_status):
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()
            
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        
        # Plot 1: Main KPIs
        ax1 = fig.add_subplot(211)
        metrics = ['total_gain', 'completed_jobs', 'total_experiments']
        values = [kpis.get(m, 0) for m in metrics]
        ax1.bar(metrics, values, color=['skyblue', 'salmon', 'lightgreen'])
        ax1.set_title("Integrated KPIs")
        
        # Plot 2: Tech Maturity
        ax2 = fig.add_subplot(212)
        techs = list(tech_status.keys())
        maturities = [tech_status[t]['maturity'] for t in techs]
        uncertainties = [tech_status[t]['uncertainty'] for t in techs]
        
        x = range(len(techs))
        ax2.bar(x, maturities, width=0.4, label='Maturity', align='center', color='green')
        ax2.bar(x, uncertainties, width=0.4, label='Uncertainty', align='edge', color='orange')
        ax2.set_xticks(x)
        ax2.set_xticklabels(techs)
        ax2.set_title("Technology Maturity vs Uncertainty")
        ax2.legend()
        
        fig.tight_layout()
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def save_results_to_csv(self, kpis, params):
        output_path = 'reports/plan6_results.csv'
        res = kpis.copy()
        res.update(params)
        df = pd.DataFrame([res])
        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    if not os.path.exists('reports'): os.makedirs('reports')
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()
