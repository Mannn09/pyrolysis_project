import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class IOManager:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # 1. Setup Paths using Pathlib
        self.base_path = Path.cwd()
        self.results_root = self.base_path / "results"
        self.docs_dir = self.base_path / "documentation"
        
        # 2. Find the next Run ID
        run_id = self._get_next_run_id()
        
        # 3. Define the specific run folder
        self.run_dir = self.results_root / f"run_{run_id}"
        
        # Create folders
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = []
        
        print("\n" + "="*60)
        print(f"FOLDER CREATED: {self.run_dir}")
        print(f"RUN ID: {run_id}")
        print("="*60 + "\n")

    def _get_next_run_id(self):
        """Checks the results folder and increments the number."""
        if not self.results_root.exists():
            return 1
        folders = list(self.results_root.glob("run_*"))
        if not folders:
            return 1
        ids = []
        for f in folders:
            try:
                num = int(f.name.split('_')[-1])
                ids.append(num)
            except:
                continue
        return max(ids) + 1 if ids else 1

    def save_iteration_data(self, T, P, W, current_min):
        """Saves highly detailed plots with labels and legends."""
        try:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            # Physical extent in millimeters [left, right, bottom, top]
            extent_mm = [0, self.cfg.R_MAX * 1000, 0, self.cfg.Z_MAX * 1000]

            # --- 1. TEMPERATURE PLOT ---
            im1 = ax[0].imshow(T.T - 273.15, origin='lower', extent=extent_mm, 
                               cmap='hot', vmin=25, vmax=650, aspect='auto')
            ax[0].set_title(f"Temperature Profile ({current_min:.1f} min)", fontweight='bold')
            ax[0].set_xlabel("Radius (mm)")
            ax[0].set_ylabel("Height (mm)")
            plt.colorbar(im1, ax=ax[0], label="Temperature (°C)")

            # --- 2. PRESSURE PLOT ---
            P_over_kpa = (P.T - self.cfg.P_ATM) / 1000.0
            im2 = ax[1].imshow(P_over_kpa, origin='lower', extent=extent_mm, 
                               cmap='jet', aspect='auto')
            ax[1].set_title(f"Gas Overpressure ({current_min:.1f} min)", fontweight='bold')
            ax[1].set_xlabel("Radius (mm)")
            ax[1].set_ylabel("Height (mm)")
            plt.colorbar(im2, ax=ax[1], label="Gauge Pressure (kPa)")

            plt.tight_layout()
            
            # FIX: Using os.path.join to prevent the TypeError
            filename = f"step_{int(round(current_min))}min.png"
            save_path = os.path.join(self.run_dir, filename)
            
            plt.savefig(save_path, dpi=150)
            plt.close()

            # Record history
            self.history.append({
                'Time_min': round(current_min, 2),
                'Avg_Temp_C': round(np.mean(T)-273.15, 1),
                'Max_P_kPa': round(np.max(P_over_kpa), 3)
            })
        except Exception as e:
            print(f"Error during save: {e}")

    def finalize(self):
        """Save CSV results."""
        if self.history:
            df = pd.DataFrame(self.history)
            # FIX: Using os.path.join here as well
            csv_path = os.path.join(self.run_dir, "simulation_log.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n[DONE] Data saved in: {self.run_dir}")