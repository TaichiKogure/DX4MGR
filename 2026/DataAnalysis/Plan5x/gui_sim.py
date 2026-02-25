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
    # 実行場所がPlan5x直下でない場合の対応
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.simulation import Simulation

class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plan 5x R&D Simulation Interface")
        self.root.geometry("1000x700")

        self.setup_ui()

    def setup_ui(self):
        # Main frames
        left_frame = ttk.LabelFrame(self.root, text="入力パラメータ (Input Parameters)", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right_frame = ttk.LabelFrame(self.root, text="実行結果 (Results)", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Input variables
        self.params = {}
        self.row_counter = 0
        
        # Helper to add entries
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

        add_entry("シナリオID:", "scenario_id", "Interactive_Sim")
        add_entry("シミュレーション日数:", "steps", 500)
        add_entry("DRしきい値 (Evidence):", "dr_threshold", 5.0)
        add_entry("実験戦略 (Strategy):", "strategy", ["strategic", "random"], "combo")
        add_entry("デジタルツイン使用:", "use_digital_twin", True, "check")
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=self.row_counter, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.row_counter += 1
        
        ttk.Label(left_frame, text="通信・組織パラメータ", font=('Helvetica', 10, 'bold')).grid(row=self.row_counter, column=0, columnspan=2, pady=5)
        self.row_counter += 1
        
        add_entry("Res -> Proto 遅延:", "delay_res_proto", 2)
        add_entry("Res -> Proto 欠落率:", "loss_res_proto", 0.1)
        add_entry("Res -> Proto 歪み率:", "dist_res_proto", 0.1)
        
        add_entry("Proto -> Ana 遅延:", "delay_proto_ana", 1)
        add_entry("Proto -> Ana 欠落率:", "loss_proto_ana", 0.05)
        add_entry("Proto -> Ana 歪み率:", "dist_proto_ana", 0.05)
        
        add_entry("Ana -> Res 遅延:", "delay_ana_res", 2)
        add_entry("Ana -> Res 欠落率:", "loss_ana_res", 0.1)
        add_entry("Ana -> Res 歪み率:", "dist_ana_res", 0.2)

        # Output options
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=self.row_counter, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.row_counter += 1
        
        self.save_csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="CSVに出力 (reports/gui_results.csv)", variable=self.save_csv_var).grid(row=self.row_counter, column=0, columnspan=2, sticky=tk.W)
        self.row_counter += 1

        # Run button
        run_btn = ttk.Button(left_frame, text="シミュレーション実行", command=self.run_simulation)
        run_btn.grid(row=self.row_counter, column=0, columnspan=2, pady=20)
        self.row_counter += 1

        # Right frame components
        self.results_text = tk.Text(right_frame, height=10, width=50)
        self.results_text.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        self.fig_canvas = None
        self.plot_frame = ttk.Frame(right_frame)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def run_simulation(self):
        try:
            # Gather params
            p = {}
            for k, v in self.params.items():
                val = v.get()
                # Type conversion
                if k in ['steps', 'delay_res_proto', 'delay_proto_ana', 'delay_ana_res']:
                    p[k] = int(val)
                elif k in ['dr_threshold', 'loss_res_proto', 'dist_res_proto', 'loss_proto_ana', 'dist_proto_ana', 'loss_ana_res', 'dist_ana_res']:
                    p[k] = float(val)
                else:
                    p[k] = val
            
            p['past_logs_file'] = 'configs/past_logs.csv'
            
            # Run simulation
            sim = Simulation()
            sim.setup_with_params(p)
            kpis = sim.run(steps=p['steps'])
            
            # Update display
            self.display_results(kpis, p)
            
            # Save to CSV if requested
            if self.save_csv_var.get():
                self.save_results_to_csv(kpis, p)
                
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")

    def display_results(self, kpis, params):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Scenario: {params['scenario_id']}\n")
        self.results_text.insert(tk.END, "-"*40 + "\n")
        self.results_text.insert(tk.END, f"Total Gain: {kpis['total_gain']:.2f}\n")
        self.results_text.insert(tk.END, f"Integration Days: {kpis['integration_days']:.1f}\n")
        self.results_text.insert(tk.END, f"Market Complaints: {kpis['market_complaints']}\n")
        self.results_text.insert(tk.END, f"Total Experiments: {kpis['total_experiments']}\n")
        self.results_text.insert(tk.END, f"Technical Failures: {kpis['technical_failures']}\n")
        self.results_text.insert(tk.END, f"Operational Failures: {kpis['operational_failures']}\n")
        self.results_text.insert(tk.END, f"Rework Count: {kpis['rework_count']}\n")

        # Plotting
        self.update_plot(kpis)

    def update_plot(self, kpis):
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()
            
        fig, ax = plt.subplots(figsize=(5, 4))
        metrics = ['total_gain', 'integration_days', 'market_complaints', 'total_experiments']
        values = [kpis[m] for m in metrics]
        
        ax.bar(metrics, values, color=['blue', 'green', 'red', 'orange'])
        ax.set_title("Key Performance Indicators")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def save_results_to_csv(self, kpis, params):
        output_path = 'reports/gui_results.csv'
        res = kpis.copy()
        res.update(params)
        df = pd.DataFrame([res])
        
        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        # also update the main report if needed? Maybe just notify
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    if not os.path.exists('reports'):
        os.makedirs('reports')
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()
