import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

class IOManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_path = Path.cwd()
        self.results_root = self.base_path / "results"
        self.run_id = self._get_next_run_id()
        self.run_dir = self.results_root / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = []
        self.residual_hist = []

        # --- UPDATED GLOBAL SCALING LIMITS ---
        # Temperature: Fixed from 40C to 800C per your request
        self.T_min, self.T_max = 40.0, 800.0
        # Pressure: Fixed at 0.2 kPa for clear gradient analysis
        self.P_min_kpa, self.P_max_kpa = 0.0, 0.05
        
        # Initialize the comprehensive run report
        self._generate_run_report()
        
        print(f"\n[INIT] Run {self.run_id} started.")
        print(f"[INFO] Report generated with T_max = {self.T_max}°C")

    def _get_next_run_id(self):
        if not self.results_root.exists(): return 1
        folders = list(self.results_root.glob("run_*"))
        ids = [int(f.name.split('_')[-1]) for f in folders if f.name.split('_')[-1].isdigit()]
        return max(ids) + 1 if ids else 1

    def _generate_run_report(self):
        """
        Creates a categorized Run Report containing all initial settings.
        """
        report_path = self.run_dir / "run_report.txt"
        with open(report_path, "w") as f:
            f.write("====================================================\n")
            f.write(f"      PYROLYSIS SIMULATION REPORT - RUN {self.run_id}\n")
            f.write("====================================================\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("1. PHYSICAL PROPERTIES\n")
            f.write(f"   K_EFF (Thermal Conductivity):  {self.cfg.K_EFF} W/m*K\n")
            f.write(f"   RHO_S_INIT (Bulk Density):     {self.cfg.RHO_S_INIT} kg/m^3\n")
            f.write(f"   CP_EFF (Specific Heat):        {self.cfg.CP_EFF} J/kg*K\n")
            f.write(f"   POROSITY (Epsilon):            {self.cfg.POROSITY}\n")
            f.write(f"   PERMEABILITY (K):              {self.cfg.PERMEABILITY} m^2\n\n")

            f.write("2. GAS & REFERENCE PROPERTIES\n")
            f.write(f"   T_INIT (Starting Temp):        {self.cfg.T_INIT} K ({self.cfg.T_INIT-273.15} C)\n")
            f.write(f"   P_ATM (Outlet Pressure):       {self.cfg.P_ATM} Pa\n")
            f.write(f"   MU_GAS (Viscosity):            {self.cfg.MU_GAS} Pa*s\n")
            f.write(f"   MW_GAS_AVG (Molecular Weight): {self.cfg.MW_GAS_AVG} kg/mol\n\n")

            f.write("3. KINETIC PARAMETERS\n")
            f.write("   Format: [A, Ea, n, deltaH, f, eta]\n")
            for key, val in self.cfg.KINETICS_PARAMS.items():
                f.write(f"   - {key}: {val}\n")
            f.write("\n")

            f.write("4. GEOMETRY & NUMERICAL SETUP\n")
            f.write(f"   Dimensions (R x Z):            {self.cfg.R_MAX*1000}mm x {self.cfg.Z_MAX*1000}mm\n")
            f.write(f"   Grid Resolution (NR x NZ):     {self.cfg.NR} x {self.cfg.NZ}\n")
            f.write(f"   Time Step (DT):                {self.cfg.DT} s\n\n")

            f.write("5. SIMULATION SETTINGS\n")
            f.write(f"   TOTAL_MIN:                     {self.cfg.TOTAL_MIN} min\n")
            f.write(f"   HEATING_RATE (Beta):           {self.cfg.HEATING_RATE} C/min\n\n")

            f.write("6. JSON-SPEC BOUNDARY CONDITIONS\n")
            f.write("   - r=0: Symmetry (Zero Heat/Pressure Flux)\n")
            f.write("   - r=R: Heated Wall (Dirichlet: T_init + Beta*t), Impermeable to gas\n")
            f.write("   - z=0: Insulated Base, Impermeable to gas\n")
            f.write("   - z=L: Insulated Top, Gas Outlet (Fixed P_atm)\n")
            f.write("====================================================\n")

    def _compute_bc_residuals(self, T, P, T_target):
        res_sym = np.mean(np.abs(T[1, :] - T[0, :]))
        res_insul = np.mean(np.abs(T[:, 1] - T[:, 0]))
        res_wall = np.mean(np.abs(T[-1, :] - T_target))
        res_p_seal = (np.mean(np.abs(P[-1, :] - P[-2, :])) + np.mean(np.abs(P[:, 1] - P[:, 0]))) / 2
        return {'res_symmetry': res_sym, 'res_insulation': res_insul, 'res_wall_heating': res_wall, 'res_pressure_seal': res_p_seal}

    def save_iteration_data(self, T, P, alpha, q_rxn, current_min):
        try:
            T_target_k = self.cfg.T_INIT + (self.cfg.HEATING_RATE / 60.0) * (current_min * 60)
            residuals = self._compute_bc_residuals(T, P, T_target_k)
            residuals['Time_min'] = current_min
            self.residual_hist.append(residuals)

            fig, ax = plt.subplots(1, 3, figsize=(20, 6))
            ext = [0, self.cfg.R_MAX * 1000, 0, self.cfg.Z_MAX * 1000]

            # 1. Temperature (Fixed Scale: 40 - 800 C)
            im1 = ax[0].imshow(T.T - 273.15, origin='lower', extent=ext, cmap='hot', 
                               aspect='auto', vmin=self.T_min, vmax=self.T_max)
            ax[0].set_title(f"Temperature Profile (Minute {int(current_min)})", fontweight='bold')
            ax[0].set_xlabel("R (mm)"); ax[0].set_ylabel("Z (mm)")
            plt.colorbar(im1, ax=ax[0], label="Temp (°C)")

            # 2. Pressure (Fixed Scale: 0 - 0.2 kPa)
            over_kpa = (P.T - self.cfg.P_ATM) / 1000.0
            im2 = ax[1].imshow(over_kpa, origin='lower', extent=ext, cmap='jet', 
                               aspect='auto', vmin=self.P_min_kpa, vmax=self.P_max_kpa)
            ax[1].set_title(f"Overpressure (Minute {int(current_min)})", fontweight='bold')
            ax[1].set_xlabel("R (mm)"); ax[1].set_ylabel("Z (mm)")
            plt.colorbar(im2, ax=ax[1], label="Pressure (kPa)")

            # 3. Reaction Progress (Fixed Scale: 0 - 1)
            avg_alpha = (alpha['alpha_1'] + alpha['alpha_2'] + alpha['alpha_3'] + alpha['alpha_4']) / 4
            im3 = ax[2].imshow(avg_alpha.T, origin='lower', extent=ext, cmap='viridis', 
                               aspect='auto', vmin=0.0, vmax=1.0)
            ax[2].set_title(f"Avg Reaction Progress (Minute {int(current_min)})", fontweight='bold')
            ax[2].set_xlabel("R (mm)"); ax[2].set_ylabel("Z (mm)")
            plt.colorbar(im3, ax=ax[2], label="Fraction (α)")

            plt.tight_layout()
            plt.savefig(self.run_dir / f"state_{int(current_min)}min.png", dpi=130)
            plt.close()

            self.history.append({
                'Time_min': current_min,
                'Avg_T_C': np.mean(T) - 273.15,
                'Max_P_kPa': np.max(over_kpa),
                'Conversion': 1.0 - np.mean(avg_alpha)
            })
        except Exception as e:
            print(f"Error in saving Minute {current_min}: {e}")

    def finalize(self):
        if not self.residual_hist: return
        res_df = pd.DataFrame(self.residual_hist)
        pd.DataFrame(self.history).to_csv(self.run_dir / "simulation_log.csv", index=False)
        res_df.to_csv(self.run_dir / "residuals_log.csv", index=False)
        print(f"[DONE] Report and logs saved in {self.run_dir}")