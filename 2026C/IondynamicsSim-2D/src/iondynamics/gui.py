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
        self.last_phase_map = None
        self.last_micro_config = None
        self.last_transport_map = None
        self.last_micro_stats = None

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

        # Tab 4: Microstructure
        self.tab_micro = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_micro, text="Microstructure")
        self._build_micro_tab()

        # Tab 5: Transport 2D
        self.tab_trans2d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_trans2d, text="Transport 2D")
        self._build_transport2d_tab()

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

    def _open_trans_out_dir(self):
        out_dir = getattr(self, 'trans_out_dir', "outputs/gui_trans2d")
        if os.path.exists(out_dir):
            os.startfile(os.path.abspath(out_dir))
        else:
            messagebox.showinfo("Info", "Output directory does not exist yet.")

    def _build_micro_tab(self):
        main_f = ttk.Frame(self.tab_micro, padding="10")
        main_f.pack(fill=tk.BOTH, expand=True)

        left_f = ttk.Frame(main_f, width=300)
        left_f.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_f, text="Microstructure Generator", font=("", 12, "bold")).pack(pady=10)
        
        # パラメータ入力 (簡易版)
        self.micro_vars = {}
        fields = [
            ("Width (um)", "width_um", 100.0),
            ("Resolution (um/px)", "resolution_um_per_px", 0.5),
            ("Target Porosity", "target_porosity", 0.3),
            ("Active Material Fraction", "target_active_fraction", 0.6),
            ("CBD Fraction", "target_cbd_fraction", 0.1),
            ("Calendaring Ratio", "calendaring_ratio", 1.0),
            ("Random Seed", "random_seed", 42)
        ]
        
        for label, key, default in fields:
            f = ttk.Frame(left_f)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=label, width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            ttk.Entry(f, textvariable=var, width=10).pack(side=tk.LEFT)
            self.micro_vars[key] = var

        ttk.Button(left_f, text="🚀 Generate structure", command=self.run_micro_gen_thread).pack(fill=tk.X, pady=20)
        
        # 進捗表示エリア
        micro_progress_f = ttk.LabelFrame(left_f, text="Generation Status", padding="5")
        micro_progress_f.pack(fill=tk.X, pady=5)
        
        self.micro_status_var = tk.StringVar(value="Ready")
        ttk.Label(micro_progress_f, textvariable=self.micro_status_var).pack(fill=tk.X)
        
        self.micro_progress = ttk.Progressbar(micro_progress_f, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.micro_progress.pack(fill=tk.X, pady=5)

        self.micro_info_text = tk.Text(left_f, height=10, width=40, state='disabled')
        self.micro_info_text.pack(fill=tk.X, pady=10)

        # 右側: プレビュー表示
        right_f = ttk.LabelFrame(main_f, text="Preview")
        right_f.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.micro_img_label = ttk.Label(right_f, text="Generated image will appear here")
        self.micro_img_label.pack(fill=tk.BOTH, expand=True)

    def run_micro_gen_thread(self):
        threading.Thread(target=self._run_micro_gen, daemon=True).start()

    def _run_micro_gen(self):
        from iondynamics.microstructure import MicrostructureConfig, MicrostructureGenerator, MicrostructureAnalyzer, visualize_microstructure
        
        logging.info("Generating microstructure...")
        self.root.after(0, lambda: self.micro_status_var.set("Initializing..."))
        self.root.after(0, lambda: self.micro_progress.config(value=0))
        
        def micro_progress_cb(data):
            msg = data.get("message", "")
            step = data.get("step")
            n_steps = data.get("n_steps")
            
            def update():
                if msg:
                    self.micro_status_var.set(msg)
                if step is not None and n_steps is not None:
                    prog = (step / n_steps) * 100
                    self.micro_progress.config(value=prog)
            self.root.after(0, update)
            if msg:
                logging.info(f"[Microstructure] {msg}")

        try:
            m_cfg = MicrostructureConfig(
                target_porosity=float(self.micro_vars["target_porosity"].get()),
                target_active_fraction=float(self.micro_vars["target_active_fraction"].get()),
                target_cbd_fraction=float(self.micro_vars["target_cbd_fraction"].get()),
                calendaring_ratio=float(self.micro_vars["calendaring_ratio"].get()),
                random_seed=int(self.micro_vars["random_seed"].get()),
                width_um=float(self.micro_vars["width_um"].get()),
                thickness_um=float(self.entries["electrode.thickness_um"].get()),
                resolution_um_per_px=float(self.micro_vars["resolution_um_per_px"].get())
            )
            
            generator = MicrostructureGenerator(m_cfg)
            phase_map = generator.generate(progress_callback=micro_progress_cb)
            
            micro_progress_cb({"message": "Analyzing structure..."})
            analyzer = MicrostructureAnalyzer(phase_map)
            transport_map = analyzer.analyze()
            stats = analyzer.get_summary_statistics(transport_map, m_cfg)
            
            # Store results in GUI instance
            self.last_phase_map = phase_map
            self.last_micro_config = m_cfg
            self.last_transport_map = transport_map
            self.last_micro_stats = stats

            micro_progress_cb({"message": "Rendering visualization..."})
            temp_img = "outputs/gui_micro_preview.png"
            os.makedirs("outputs", exist_ok=True)
            visualize_microstructure(phase_map, transport_map, save_path=temp_img)
            
            self.root.after(0, lambda: self._update_micro_ui(temp_img, stats))
            self.root.after(0, lambda: self.micro_status_var.set("Generation Completed"))
            logging.info("Microstructure generation successful.")
        except Exception as e:
            error_msg = f"Microstructure generation failed: {e}"
            logging.error(error_msg)
            self.root.after(0, lambda: self.micro_status_var.set("Error occurred"))
            self.root.after(0, lambda: messagebox.showerror("Generation Error", error_msg))

    def _update_micro_ui(self, img_path, stats):
        self.micro_info_text.config(state='normal')
        self.micro_info_text.delete('1.0', tk.END)
        for k, v in stats.items():
            self.micro_info_text.insert(tk.END, f"{k}: {v}\n")
        self.micro_info_text.config(state='disabled')
        
        from PIL import Image, ImageTk
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img.thumbnail((600, 600))
            self.micro_photo = ImageTk.PhotoImage(img)
            self.micro_img_label.config(image=self.micro_photo, text="")

    def _build_transport2d_tab(self):
        main_f = ttk.Frame(self.tab_trans2d, padding="10")
        main_f.pack(fill=tk.BOTH, expand=True)

        left_f = ttk.Frame(main_f, width=300)
        left_f.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_f, text="2D Ion Transport Solver", font=("", 12, "bold")).pack(pady=10)
        
        # パラメータ入力
        self.trans_vars = {}
        
        # 構造の再利用設定
        micro_reuse_f = ttk.LabelFrame(left_f, text="Microstructure Source", padding="5")
        micro_reuse_f.pack(fill=tk.X, pady=5)
        
        self.trans_micro_source = tk.StringVar(value="reuse")
        ttk.Radiobutton(micro_reuse_f, text="Use generated Microstructure", 
                        variable=self.trans_micro_source, value="reuse").pack(anchor=tk.W)
        ttk.Radiobutton(micro_reuse_f, text="Regenerate before run", 
                        variable=self.trans_micro_source, value="regenerate").pack(anchor=tk.W)

        fields = [
            ("Time Step (s)", "dt", 1.0),
            ("Final Time (s)", "t_final", 600.0),
            ("Initial Conc. (mol/m3)", "c_initial", 1000.0),
            ("C-rate (for flux)", "c_rate", 1.0),
            ("BC Separator", "bc_sep", "constant_concentration"),
        ]
        
        for label, key, default in fields:
            f = ttk.Frame(left_f)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=label, width=20).pack(side=tk.LEFT)
            if key == "bc_sep":
                var = tk.StringVar(value=default)
                combo = ttk.Combobox(f, textvariable=var, values=["constant_concentration", "constant_flux"], width=15)
                combo.pack(side=tk.LEFT)
            else:
                var = tk.StringVar(value=str(default))
                ttk.Entry(f, textvariable=var, width=10).pack(side=tk.LEFT)
            self.trans_vars[key] = var

        ttk.Button(left_f, text="⚡ Run 2D Transport", command=self.run_trans2d_thread).pack(fill=tk.X, pady=20)
        
        # 出力ディレクトリ情報
        self.trans_out_label = ttk.Label(left_f, text="Output: outputs/gui_trans2d", foreground="blue", wraplength=250)
        self.trans_out_label.pack(fill=tk.X, pady=2)
        ttk.Button(left_f, text="📂 Open Output Folder", command=self._open_trans_out_dir).pack(fill=tk.X, pady=2)

        # 進捗表示エリア
        progress_f = ttk.LabelFrame(left_f, text="Execution Status", padding="5")
        progress_f.pack(fill=tk.X, pady=5)
        
        self.trans_status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_f, textvariable=self.trans_status_var, wraplength=250).pack(fill=tk.X)
        
        self.trans_progress = ttk.Progressbar(progress_f, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.trans_progress.pack(fill=tk.X, pady=5)
        
        self.trans_stats_var = tk.StringVar(value="")
        ttk.Label(progress_f, textvariable=self.trans_stats_var).pack(fill=tk.X)

        self.trans_info_text = tk.Text(left_f, height=10, width=40, state='disabled')
        self.trans_info_text.pack(fill=tk.X, pady=10)

        # 右側: プレビュー表示
        right_f = ttk.LabelFrame(main_f, text="Results Preview")
        right_f.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # プレビュー切り替えボタン
        btn_f = ttk.Frame(right_f)
        btn_f.pack(fill=tk.X)
        self.preview_mode = tk.StringVar(value="ce_final")
        ttk.Radiobutton(btn_f, text="Concentration (2D)", variable=self.preview_mode, value="ce_final", command=self._refresh_trans_preview).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(btn_f, text="Potential (2D)", variable=self.preview_mode, value="phi_e", command=self._refresh_trans_preview).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(btn_f, text="Profiles (1D)", variable=self.preview_mode, value="profiles", command=self._refresh_trans_preview).pack(side=tk.LEFT, padx=5)

        self.trans_img_label = ttk.Label(right_f, text="Results will appear here")
        self.trans_img_label.pack(fill=tk.BOTH, expand=True)

    def run_trans2d_thread(self):
        threading.Thread(target=self._run_trans2d, daemon=True).start()

    def _run_trans2d(self):
        from iondynamics.solver2d import TransportSolverConfig, TransportSolver2D, save_transport_results
        from iondynamics.microstructure import MicrostructureConfig, MicrostructureGenerator
        
        logging.info("Starting 2D Transport Simulation...")
        self.root.after(0, lambda: self.trans_status_var.set("Initializing..."))
        self.root.after(0, lambda: self.trans_progress.config(value=0))
        
        def progress_cb(data):
            msg = data.get("message", "")
            step = data.get("step")
            n_steps = data.get("n_steps")
            eta = data.get("eta")
            
            def update():
                if msg:
                    self.trans_status_var.set(msg)
                if step is not None and n_steps is not None:
                    prog = (step / n_steps) * 100
                    self.trans_progress.config(value=prog)
                    stats_text = f"Step {step}/{n_steps}"
                    if eta is not None:
                        stats_text += f" | ETA: {eta:.1f}s"
                    self.trans_stats_var.set(stats_text)
                
            self.root.after(0, update)
            if msg:
                logging.info(f"[Transport2D] {msg}")

        try:
            # 1. 構造の決定 (再利用 or 新規生成)
            reuse = (self.trans_micro_source.get() == "reuse")
            if reuse and self.last_phase_map is not None:
                logging.info("Reusing previously generated microstructure.")
                phase_map = self.last_phase_map
                m_cfg = self.last_micro_config
                micro_log = "Reused existing structure"
            else:
                if reuse:
                    logging.info("No generated microstructure found. Generating new one...")
                
                m_cfg = MicrostructureConfig(
                    target_porosity=float(self.micro_vars["target_porosity"].get()),
                    target_active_fraction=float(self.micro_vars["target_active_fraction"].get()),
                    target_cbd_fraction=float(self.micro_vars["target_cbd_fraction"].get()),
                    calendaring_ratio=float(self.micro_vars["calendaring_ratio"].get()),
                    random_seed=int(self.micro_vars["random_seed"].get()),
                    width_um=float(self.micro_vars["width_um"].get()),
                    thickness_um=float(self.entries["electrode.thickness_um"].get()),
                    resolution_um_per_px=float(self.micro_vars["resolution_um_per_px"].get())
                )
                progress_cb({"message": "Generating microstructure..."})
                gen = MicrostructureGenerator(m_cfg)
                phase_map = gen.generate(progress_callback=progress_cb)
                micro_log = "Newly generated structure"
            
            # 2. 輸送設定
            t_cfg = TransportSolverConfig(
                case_name="gui_trans2d",
                dt=float(self.trans_vars["dt"].get()),
                t_final=float(self.trans_vars["t_final"].get()),
                c_initial=float(self.trans_vars["c_initial"].get()),
                applied_current_density=float(self.trans_vars["c_rate"].get()) * 20.0,
                bc_separator=self.trans_vars["bc_sep"].get()
            )
            
            solver = TransportSolver2D(t_cfg, phase_map.grid)
            solver.prepare_property_map(phase_map)
            
            res_steady = solver.solve_steady_potential(progress_callback=progress_cb)
            res_transient = solver.solve_transient_concentration(progress_callback=progress_cb)
            
            # 3. 保存と可視化
            progress_cb({"message": "Saving results and generating plots..."})
            self.trans_out_dir = "outputs/gui_trans2d"
            os.makedirs(self.trans_out_dir, exist_ok=True)
            
            # 保存
            save_transport_results(self.trans_out_dir, t_cfg, res_steady, res_transient, solver.prop_map)
            
            # 可視化
            from iondynamics.visualize2d import visualize_transport_results
            visualize_transport_results(res_steady, res_transient, solver.prop_map, 
                                        {"width_um": m_cfg.width_um, "thickness_um": m_cfg.thickness_um},
                                        self.trans_out_dir,
                                        phase_map=phase_map)
            
            stats = {**res_steady.kpis, **res_transient.kpis}
            self.root.after(0, lambda: self._update_trans_ui(stats))
            self.root.after(0, lambda: self.trans_status_var.set("Simulation Completed"))
            
            # 詳細ログ
            logging.info(f"2D Transport Simulation successful.")
            logging.info(f" - Microstructure: {micro_log}")
            logging.info(f" - Grid: {phase_map.grid.nx} x {phase_map.grid.nz}")
            logging.info(f" - Physical: {m_cfg.width_um} x {m_cfg.thickness_um} um (res: {m_cfg.resolution_um_per_px})")
            logging.info(f" - Output: {os.path.abspath(self.trans_out_dir)}")
            
            self.root.after(0, lambda: self.trans_out_label.config(text=f"Output: {self.trans_out_dir}"))

        except Exception as e:
            import traceback
            error_msg = f"2D Transport Simulation failed: {e}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            self.root.after(0, lambda: self.trans_status_var.set("Error occurred"))
            self.root.after(0, lambda: messagebox.showerror("Simulation Error", error_msg))

    def _update_trans_ui(self, stats):
        self.trans_info_text.config(state='normal')
        self.trans_info_text.delete('1.0', tk.END)
        for k, v in stats.items():
            self.trans_info_text.insert(tk.END, f"{k}: {v}\n")
        self.trans_info_text.config(state='disabled')
        self._refresh_trans_preview()

    def _refresh_trans_preview(self):
        if not hasattr(self, 'trans_out_dir'): return
        
        mode = self.preview_mode.get()
        mapping = {
            "ce_final": "ce_final_2d.png",
            "phi_e": "phi_e_steady.png",
            "profiles": "ce_profiles_1d.png"
        }
        img_path = os.path.join(self.trans_out_dir, mapping.get(mode, "ce_final_2d.png"))
        
        from PIL import Image, ImageTk
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img.thumbnail((600, 600))
            self.trans_photo = ImageTk.PhotoImage(img)
            self.trans_img_label.config(image=self.trans_photo, text="")

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
