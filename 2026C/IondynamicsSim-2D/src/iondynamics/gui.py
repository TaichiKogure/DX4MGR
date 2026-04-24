import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os
import subprocess
import threading
import logging
import yaml
import dataclasses

# Add src directory to sys.path to support direct execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iondynamics.config import load_config, SimConfig, validate_config
from iondynamics.cli import execute_simulation, execute_animation, run_comparison
from iondynamics.sweep import run_sweep

# ロギング設定 (GUIのテキストボックスに出力するため)
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        self.text_widget.after(0, append)

class IondynamicsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IondynamicsSim - Advanced Battery Simulation")
        self.root.geometry("1000x800")

        self.config_path = "configs/default.yaml"
        self.cfg = None
        self._load_config_file(self.config_path)

        self.last_run_dir = None
        self.entries = {}
        self._create_widgets()
        self._setup_logging()

    def _load_config_file(self, path):
        try:
            self.cfg = load_config(path)
            self.config_path = path
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
            return False

    def _save_config_file(self, path):
        if not self.update_config_from_ui():
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(dataclasses.asdict(self.cfg), f, allow_unicode=True)
            messagebox.showinfo("Success", f"Config saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")

    def _create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # タブコントロール
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # Tab 1: Configuration
        self.tab_config = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_config, text="Settings")
        self._build_config_tab()

        # Tab 2: Execution
        self.tab_exec = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_exec, text="Run / Compare")
        self._build_exec_tab()

        # Tab 3: History & Analysis
        self.tab_history = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_history, text="Analysis")
        self._build_history_tab()

        # 下部: ログエリア
        log_frame = ttk.LabelFrame(main_frame, text="System Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=False, side=tk.BOTTOM)
        self.log_text = tk.Text(log_frame, height=8, state='disabled', wrap='word', bg="#f0f0f0")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_config_tab(self):
        # ファイル操作ボタン
        file_btn_frame = ttk.Frame(self.tab_config, padding="5")
        file_btn_frame.pack(fill=tk.X)
        ttk.Button(file_btn_frame, text="Load YAML", command=self._on_load_yaml).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_btn_frame, text="Save YAML", command=self._on_save_yaml).pack(side=tk.LEFT, padx=5)
        self.lbl_config_path = ttk.Label(file_btn_frame, text=f"Current: {self.config_path}")
        self.lbl_config_path.pack(side=tk.LEFT, padx=20)

        # スクロール可能な領域
        canvas = tk.Canvas(self.tab_config)
        scrollbar = ttk.Scrollbar(self.tab_config, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # パラメータ入力エリア
        # Meta
        meta_f = ttk.LabelFrame(scrollable_frame, text="Meta Data", padding="5")
        meta_f.pack(fill=tk.X, padx=5, pady=5)
        self._add_entry(meta_f, "Case Name", "meta.case_name", self.cfg.meta.case_name)
        
        # Electrode
        elec_f = ttk.LabelFrame(scrollable_frame, text="Electrode Structure", padding="5")
        elec_f.pack(fill=tk.X, padx=5, pady=5)
        self._add_entry(elec_f, "Thickness (um)", "electrode.thickness_um", self.cfg.electrode.thickness_um)
        self._add_entry(elec_f, "Porosity", "electrode.porosity", self.cfg.electrode.porosity)
        self._add_entry(elec_f, "Particle Radius (um)", "electrode.particle_radius_um", self.cfg.electrode.particle_radius_um)

        # Resistances
        res_f = ttk.LabelFrame(scrollable_frame, text="Material Properties", padding="5")
        res_f.pack(fill=tk.X, padx=5, pady=5)
        self._add_entry(res_f, "Elec. Cond. (S/m)", "resistances.electronic_conductivity_S_m", self.cfg.resistances.electronic_conductivity_S_m)
        self._add_entry(res_f, "Ionic Cond. (S/m)", "resistances.ionic_conductivity_S_m", self.cfg.resistances.ionic_conductivity_S_m)
        self._add_entry(res_f, "Solid Diff. (m2/s)", "resistances.solid_diffusivity_m2_s", self.cfg.resistances.solid_diffusivity_m2_s)

        # Operation
        oper_f = ttk.LabelFrame(scrollable_frame, text="Operation Conditions", padding="5")
        oper_f.pack(fill=tk.X, padx=5, pady=5)
        self._add_entry(oper_f, "C-rate", "operation.c_rate", self.cfg.operation.c_rate)
        self._add_entry(oper_f, "Cutoff Voltage (V)", "operation.cutoff_voltage_V", self.cfg.operation.cutoff_voltage_V)

        # Particles (Advanced)
        part_f = ttk.LabelFrame(scrollable_frame, text="Particle Visualization (Simulated 2D)", padding="5")
        part_f.pack(fill=tk.X, padx=5, pady=5)
        self._add_entry(part_f, "Particle Count", "particles.count", self.cfg.particles.count)
        self._add_entry(part_f, "Seed", "particles.seed", self.cfg.particles.seed)

    def _build_exec_tab(self):
        # 実行ボタン群
        top_frame = ttk.Frame(self.tab_exec, padding="10")
        top_frame.pack(fill=tk.X)

        self.run_btn = ttk.Button(top_frame, text="▶ Run Single Simulation", command=self.run_simulation_thread, style="Accent.TButton")
        self.run_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.anim_btn = ttk.Button(top_frame, text="🎬 Generate Animation", command=self.run_animation_thread)
        self.anim_btn.pack(side=tk.LEFT, padx=10, pady=10)

        # 比較実行フレーム
        comp_frame = ttk.LabelFrame(self.tab_exec, text="Parameter Study & Comparison", padding="15")
        comp_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(comp_frame, text="Comparison Axis:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.comp_axis = tk.StringVar(value="thickness")
        combo = ttk.Combobox(comp_frame, textvariable=self.comp_axis, values=["thickness", "porosity", "particle"])
        combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(comp_frame, text="Values (comma separated):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.comp_values = tk.StringVar(value="60, 80, 100")
        ttk.Entry(comp_frame, textvariable=self.comp_values).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        self.comp_btn = ttk.Button(comp_frame, text="📊 Run Comparison Report", command=self.run_comparison_thread)
        self.comp_btn.grid(row=2, column=0, columnspan=2, pady=10)

        # スイープ実行（新規）
        sweep_frame = ttk.LabelFrame(self.tab_exec, text="Custom Sweep (YAML spec)", padding="15")
        sweep_frame.pack(fill=tk.X, padx=10, pady=10)
        self.sweep_spec_path = tk.StringVar()
        ttk.Entry(sweep_frame, textvariable=self.sweep_spec_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(sweep_frame, text="Browse Spec", command=self._on_browse_sweep_spec).pack(side=tk.LEFT, padx=5)
        self.sweep_btn = ttk.Button(sweep_frame, text="🔍 Run Sweep", command=self.run_sweep_thread)
        self.sweep_btn.pack(side=tk.LEFT, padx=5)

    def _build_history_tab(self):
        self.result_frame = ttk.LabelFrame(self.tab_history, text="Access Results", padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(self.result_frame, text="Latest run directory:").pack(anchor=tk.W)
        self.lbl_last_dir = ttk.Label(self.result_frame, text="None", foreground="blue")
        self.lbl_last_dir.pack(anchor=tk.W, pady=5)

        btn_grid = ttk.Frame(self.result_frame)
        btn_grid.pack(fill=tk.X)

        self.btn_map = {
            "Open Results Directory": self.open_dir,
            "Open Results CSV": lambda: self.open_file("results.csv"),
            "Open Voltage Plot": lambda: self.open_file("voltage_time.png"),
            "Open Ce Profiles": lambda: self.open_file("thickness_ce_profiles.png"),
            "Open KPI Plot": lambda: self.open_file("kpi_time_series.png"),
            "Open Resistance Breakdown": lambda: self.open_file("breakdown_stack.png"),
            "Open Animation": lambda: self.open_file("animation.mp4"),
            "Open Comparison Report": lambda: self.open_file("report.md")
        }

        self.result_buttons = {}
        for i, (text, cmd) in enumerate(self.btn_map.items()):
            btn = ttk.Button(btn_grid, text=text, command=cmd, state=tk.DISABLED)
            btn.grid(row=i//2, column=i%2, sticky=(tk.W, tk.E), padx=5, pady=5)
            self.result_buttons[text] = btn

    def _on_load_yaml(self):
        path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if path:
            if self._load_config_file(path):
                self._refresh_ui_from_config()
                self.lbl_config_path.config(text=f"Current: {path}")

    def _on_save_yaml(self):
        path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
        if path:
            self._save_config_file(path)

    def _on_browse_sweep_spec(self):
        path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if path:
            self.sweep_spec_path.set(path)

    def _refresh_ui_from_config(self):
        # 既存のEntry変数を更新
        mapping = {
            "meta.case_name": self.cfg.meta.case_name,
            "electrode.thickness_um": self.cfg.electrode.thickness_um,
            "electrode.porosity": self.cfg.electrode.porosity,
            "electrode.particle_radius_um": self.cfg.electrode.particle_radius_um,
            "resistances.electronic_conductivity_S_m": self.cfg.resistances.electronic_conductivity_S_m,
            "resistances.ionic_conductivity_S_m": self.cfg.resistances.ionic_conductivity_S_m,
            "resistances.solid_diffusivity_m2_s": self.cfg.resistances.solid_diffusivity_m2_s,
            "operation.c_rate": self.cfg.operation.c_rate,
            "operation.cutoff_voltage_V": self.cfg.operation.cutoff_voltage_V,
            "particles.count": self.cfg.particles.count,
            "particles.seed": self.cfg.particles.seed,
        }
        for path, val in mapping.items():
            if path in self.entries:
                self.entries[path].set(str(val))

    def _add_entry(self, parent, label_text, attr_path, default_val):
        row = len(self.entries)
        ttk.Label(parent, text=label_text, width=25).grid(row=row, column=0, sticky=tk.W, pady=2)
        var = tk.StringVar(value=str(default_val))
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.entries[attr_path] = var

    def _setup_logging(self):
        handler = TextHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def update_config_from_ui(self):
        try:
            self.cfg.meta.case_name = self.entries["meta.case_name"].get()
            self.cfg.electrode.thickness_um = float(self.entries["electrode.thickness_um"].get())
            self.cfg.electrode.porosity = float(self.entries["electrode.porosity"].get())
            self.cfg.electrode.particle_radius_um = float(self.entries["electrode.particle_radius_um"].get())
            self.cfg.resistances.electronic_conductivity_S_m = float(self.entries["resistances.electronic_conductivity_S_m"].get())
            self.cfg.resistances.ionic_conductivity_S_m = float(self.entries["resistances.ionic_conductivity_S_m"].get())
            self.cfg.resistances.solid_diffusivity_m2_s = float(self.entries["resistances.solid_diffusivity_m2_s"].get())
            self.cfg.operation.c_rate = float(self.entries["operation.c_rate"].get())
            self.cfg.operation.cutoff_voltage_V = float(self.entries["operation.cutoff_voltage_V"].get())
            self.cfg.particles.count = int(self.entries["particles.count"].get())
            self.cfg.particles.seed = int(self.entries["particles.seed"].get())
            
            validate_config(self.cfg)
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Configuration error: {e}")
            return False

    def run_simulation_thread(self):
        if not self.update_config_from_ui():
            return
        self.run_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_sim, daemon=True).start()

    def _run_sim(self):
        logging.info("Starting simulation...")
        try:
            run_dir = execute_simulation(self.cfg)
            self.last_run_dir = run_dir
            logging.info(f"Simulation finished. Results in {run_dir}")
            self.root.after(0, self._enable_result_buttons)
        except Exception as e:
            logging.error(f"Simulation failed: {e}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def run_animation_thread(self):
        if not self.update_config_from_ui():
            return
        self.anim_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_anim, daemon=True).start()

    def _run_anim(self):
        logging.info("Starting animation generation...")
        try:
            if not self.last_run_dir:
                self.last_run_dir = execute_simulation(self.cfg)
            out_path = os.path.join(self.last_run_dir, "animation.mp4")
            execute_animation(self.cfg, out_path)
            logging.info(f"Animation saved to {out_path}")
            self.root.after(0, self._enable_result_buttons)
        except Exception as e:
            logging.error(f"Animation failed: {e}")
        finally:
            self.root.after(0, lambda: self.anim_btn.config(state=tk.NORMAL))

    def run_comparison_thread(self):
        self.comp_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_comp, daemon=True).start()

    def _run_comp(self):
        axis = self.comp_axis.get()
        try:
            val_str = self.comp_values.get()
            values = [float(v.strip()) for v in val_str.split(',')]
            logging.info(f"Starting comparison sweep on {axis}: {values}")
            # GUIから直接最新のコンフィグを一時保存して使う
            temp_config = "configs/temp_gui_config.yaml"
            self._save_config_file(temp_config)
            run_comparison(temp_config, axis, values)
            logging.info("Comparison report generated.")
        except Exception as e:
            logging.error(f"Comparison failed: {e}")
        finally:
            self.root.after(0, lambda: self.comp_btn.config(state=tk.NORMAL))

    def run_sweep_thread(self):
        spec = self.sweep_spec_path.get()
        if not spec or not os.path.exists(spec):
            messagebox.showwarning("Warning", "Please specify a valid sweep spec YAML.")
            return
        self.sweep_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_sweep, args=(spec,), daemon=True).start()

    def _run_sweep(self, spec_path):
        logging.info("Starting custom sweep...")
        try:
            with open(spec_path, 'r') as f:
                spec_data = yaml.safe_load(f)
            run_sweep(self.cfg, spec_data)
            logging.info("Sweep completed.")
        except Exception as e:
            logging.error(f"Sweep failed: {e}")
        finally:
            self.root.after(0, lambda: self.sweep_btn.config(state=tk.NORMAL))

    def _enable_result_buttons(self):
        if self.last_run_dir:
            self.lbl_last_dir.config(text=self.last_run_dir)
            for btn in self.result_buttons.values():
                btn.config(state=tk.NORMAL)

    def open_dir(self):
        if self.last_run_dir:
            os.startfile(os.path.abspath(self.last_run_dir))

    def open_file(self, filename):
        if self.last_run_dir:
            path = os.path.join(self.last_run_dir, filename)
            # 特殊対応: 比較レポートは最新の比較ディレクトリを探す必要があるかもしれないが、
            # 現状は last_run_dir に保存される前提
            if os.path.exists(path):
                os.startfile(os.path.abspath(path))
            else:
                messagebox.showwarning("File not found", f"File {filename} not found.")

def main():
    root = tk.Tk()
    # スタイル設定
    style = ttk.Style()
    try:
        # 可能であればモダンなテーマを試みる
        style.theme_use('clam')
    except:
        pass
    
    app = IondynamicsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
