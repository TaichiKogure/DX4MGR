import os
import sys
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

PLAN7_DIR = Path(__file__).resolve().parents[1]
if str(PLAN7_DIR) not in sys.path:
    sys.path.insert(0, str(PLAN7_DIR))

import gui_sim  # noqa: E402


class SimulationGUIProgressTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.periodic_patch = patch.object(gui_sim.SimulationGUI, "periodic_check", lambda _self: None)
        self.periodic_patch.start()
        self.app = gui_sim.SimulationGUI(self.root)

    def tearDown(self):
        if getattr(self.app, "plot_refresh_job", None) is not None:
            self.root.after_cancel(self.app.plot_refresh_job)
        self.periodic_patch.stop()
        self.root.update_idletasks()
        self.root.destroy()

    def test_refresh_running_elapsed_labels_uses_wall_clock(self):
        self.app.start_times[0] = time.perf_counter() - 2.0
        self.app.scenario_states[0] = "running"
        self.app.progress_values[0] = 40.0

        self.app._refresh_running_elapsed_labels()

        self.assertGreaterEqual(self.app.elapsed_times[0], 1.5)
        self.assertIn("40.0%", self.app.progress_labels[0].cget("text"))

    def test_schedule_plot_refresh_coalesces_partial_updates(self):
        render_calls = []

        def fake_update_comparison_plots(data):
            render_calls.append(sorted(data.keys()))

        self.app.update_comparison_plots = fake_update_comparison_plots
        self.app.pending_comparison_plot = {"Scenario A": (None, None, None, None)}
        self.app._schedule_plot_refresh("comparison")

        first_job = self.app.plot_refresh_job
        self.app.pending_comparison_plot = {
            "Scenario A": (None, None, None, None),
            "Scenario B": (None, None, None, None)
        }
        self.app._schedule_plot_refresh("comparison")

        self.assertEqual(self.app.plot_refresh_job, first_job)

        wait_deadline = time.time() + 3.0
        while time.time() < wait_deadline and not render_calls:
            self.root.update()
            time.sleep(0.05)

        self.assertEqual(len(render_calls), 1)
        self.assertEqual(render_calls[0], ["Scenario A", "Scenario B"])


if __name__ == "__main__":
    unittest.main()