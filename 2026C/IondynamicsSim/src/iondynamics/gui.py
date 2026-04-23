import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import subprocess
import threading
import logging
from .config import load_config, SimConfig
from .cli import execute_simulation, execute_animation

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
        self.root.title("IondynamicsSim GUI")
        self.root.geometry("800x600")

        self.config_path = "configs/default.yaml"
        try:
            self.cfg = load_config(self.config_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default config: {e}")
            self.root.destroy()
            return

        self.last_run_dir = None
        self._create_widgets()
        self._setup_logging()

    def _create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左側: パラメータ入力
        input_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.entries = {}
        
        # Meta
        self._add_entry(input_frame, "Case Name", "meta.case_name", self.cfg.meta.case_name)
        
        # Electrode
        self._add_entry(input_frame, "Electrode Thickness (um)", "electrode.thickness_um", self.cfg.electrode.thickness_um)
        self._add_entry(input_frame, "Porosity", "electrode.porosity", self.cfg.electrode.porosity)
        self._add_entry(input_frame, "Particle Radius (um)", "electrode.particle_radius_um", self.cfg.electrode.particle_radius_um)
        
        # Operation
        self._add_entry(input_frame, "C-rate", "operation.c_rate", self.cfg.operation.c_rate)
        self._add_entry(input_frame, "Cutoff Voltage (V)", "operation.cutoff_voltage_V", self.cfg.operation.cutoff_voltage_V)
        
        # Particles
        self._add_entry(input_frame, "Particle Count", "particles.count", self.cfg.particles.count)

        # 右側: 操作とログ
        right_frame = ttk.Frame(main_frame, padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ボタン
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X)

        self.run_btn = ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation_thread)
        self.run_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.anim_btn = ttk.Button(btn_frame, text="Generate Animation", command=self.run_animation_thread)
        self.anim_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # 結果アクセスボタン（初期は無効）
        self.result_frame = ttk.LabelFrame(right_frame, text="Last Results", padding="10")
        self.result_frame.pack(fill=tk.X, pady=10)
        
        self.open_dir_btn = ttk.Button(self.result_frame, text="Open Results Directory", command=self.open_dir, state=tk.DISABLED)
        self.open_dir_btn.pack(fill=tk.X, pady=2)
        
        self.open_csv_btn = ttk.Button(self.result_frame, text="Open CSV (Results)", command=lambda: self.open_file("results.csv"), state=tk.DISABLED)
        self.open_csv_btn.pack(fill=tk.X, pady=2)

        self.open_img_btn = ttk.Button(self.result_frame, text="Open Voltage Plot", command=lambda: self.open_file("voltage_time.png"), state=tk.DISABLED)
        self.open_img_btn.pack(fill=tk.X, pady=2)

        self.open_anim_btn = ttk.Button(self.result_frame, text="Open Animation", command=lambda: self.open_file("animation.mp4"), state=tk.DISABLED)
        self.open_anim_btn.pack(fill=tk.X, pady=2)

        # ログ
        log_frame = ttk.LabelFrame(right_frame, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=15, state='disabled', wrap='word')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _add_entry(self, parent, label_text, attr_path, default_val):
        row = len(self.entries)
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
        var = tk.StringVar(value=str(default_val))
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.entries[attr_path] = var

    def _setup_logging(self):
        handler = TextHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def update_config(self):
        try:
            self.cfg.meta.case_name = self.entries["meta.case_name"].get()
            self.cfg.electrode.thickness_um = float(self.entries["electrode.thickness_um"].get())
            self.cfg.electrode.porosity = float(self.entries["electrode.porosity"].get())
            self.cfg.electrode.particle_radius_um = float(self.entries["electrode.particle_radius_um"].get())
            self.cfg.operation.c_rate = float(self.entries["operation.c_rate"].get())
            self.cfg.operation.cutoff_voltage_V = float(self.entries["operation.cutoff_voltage_V"].get())
            self.cfg.particles.count = int(self.entries["particles.count"].get())
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return False

    def run_simulation_thread(self):
        if not self.update_config():
            return
        
        self.run_btn.config(state=tk.DISABLED)
        self.anim_btn.config(state=tk.DISABLED)
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
            self.root.after(0, lambda: self.anim_btn.config(state=tk.NORMAL))

    def run_animation_thread(self):
        if not self.update_config():
            return
            
        self.run_btn.config(state=tk.DISABLED)
        self.anim_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_anim, daemon=True).start()

    def _run_anim(self):
        logging.info("Starting animation generation (this may take a while)...")
        try:
            if not self.last_run_dir:
                self.last_run_dir = execute_simulation(self.cfg)
                self.root.after(0, self._enable_result_buttons)
            
            out_path = os.path.join(self.last_run_dir, "animation.mp4")
            execute_animation(self.cfg, out_path)
            logging.info(f"Animation saved to {out_path}")
            self.root.after(0, lambda: self.open_anim_btn.config(state=tk.NORMAL))
        except Exception as e:
            logging.error(f"Animation failed: {e}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.anim_btn.config(state=tk.NORMAL))

    def _enable_result_buttons(self):
        self.open_dir_btn.config(state=tk.NORMAL)
        self.open_csv_btn.config(state=tk.NORMAL)
        self.open_img_btn.config(state=tk.NORMAL)
        # animation.mp4 が存在する場合のみ有効化したいが、ここでは一括有効化
        if os.path.exists(os.path.join(self.last_run_dir, "animation.mp4")):
            self.open_anim_btn.config(state=tk.NORMAL)

    def open_dir(self):
        if self.last_run_dir:
            os.startfile(os.path.abspath(self.last_run_dir))

    def open_file(self, filename):
        if self.last_run_dir:
            path = os.path.join(self.last_run_dir, filename)
            if os.path.exists(path):
                os.startfile(os.path.abspath(path))
            else:
                messagebox.showwarning("File not found", f"File {filename} not found in {self.last_run_dir}")

def main():
    root = tk.Tk()
    app = IondynamicsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
