import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# 1. ARCHITECTURE & ENFORCEMENT
# =============================================================================
class WeakCorrectionNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=6, hidden_layers=10, hidden_neurons=92):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, output_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.net(x)

class AdaBoostPINN:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weak_learners = []
        self.alphas = []
        self.R, self.L = cfg.R_MAX, cfg.Z_MAX
        self.t_end = cfg.TOTAL_MIN * 60.0
        self.beta = cfg.HEATING_RATE / 60.0

    def get_mask(self, r, z, t):
        r_h, z_h, t_h = r/self.R, z/self.L, t/self.t_end
        # Hard BC Mask: t_hat * r_hat^2 * (1-r_hat) * z_hat^2 * (1-z_hat)^2
        mask = t_h * (r_h**2) * (1 - r_h) * (z_h**2) * (1 - z_h)**2
        return torch.clamp(mask, 0.0, 1.0) * 200.0

    def get_base_fields(self, r, z, t):
        # T_base satisfies IC at t=0, Symmetry at r=0, Dirichlet at r=R
        T_base = self.cfg.T_INIT + self.beta * t * (r / self.R)**2
        p_base = torch.full_like(r, self.cfg.P_ATM)
        alpha_base = torch.ones((r.shape[0], 4)).to(self.device)
        return T_base, p_base, alpha_base

    def predict(self, r, z, t):
        r_h, z_h, t_h = r/self.R, z/self.L, t/self.t_end
        coords = torch.cat([r_h, z_h, t_h], dim=1)
        T_b, p_b, a_b = self.get_base_fields(r, z, t)
        mask = self.get_mask(r, z, t)
        
        T_c, p_c, a_c = 0, 0, 0
        for i, model in enumerate(self.weak_learners):
            out = model(coords)
            T_c += self.alphas[i] * mask * torch.tanh(out[:, 0:1]) * 100.0
            p_c += self.alphas[i] * mask * torch.tanh(out[:, 1:2]) * 500.0
            a_c += self.alphas[i] * mask * torch.tanh(out[:, 2:6])
            
        return T_b + T_c, p_b + p_c, torch.clamp(a_b + a_c, 0, 1)

# =============================================================================
# 2. TRAINING ENGINE
# =============================================================================
class AdaBoostTrainer:
    def __init__(self, pinn, fvm_data_path):
        self.pinn = pinn
        self.cfg = pinn.cfg
        self.device = pinn.device
        self.N_coll = 10000
        self.run_dir = Path(fvm_data_path) / "pinn_training"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Logs & Hist
        self.iter_log = open(self.run_dir / "iteration_log.tsv", "w")
        self.iter_log.write("Global_Iter\tRound\tEpoch\tLoss_PDE\tLoss_Data\tLR\tVal_RMS\n")
        self.history = {'train_loss': [], 'val_rms': []}

        # Points
        self.r = (torch.rand(self.N_coll, 1)*pinn.R).to(self.device).requires_grad_(True)
        self.z = (torch.rand(self.N_coll, 1)*pinn.L).to(self.device).requires_grad_(True)
        self.t = (torch.rand(self.N_coll, 1)*pinn.t_end).to(self.device).requires_grad_(True)
        self.sample_w = torch.ones(self.N_coll, 1).to(self.device) / self.N_coll
        
        # Val points (2000)
        self.r_v = torch.rand(2000, 1).to(self.device) * pinn.R
        self.z_v = torch.rand(2000, 1).to(self.device) * pinn.L
        self.t_v = torch.rand(2000, 1).to(self.device) * pinn.t_end

        # Load FVM Data for learning
        self.data_loss = torch.tensor(0.0).to(self.device)
        csv_p = Path(fvm_data_path) / "simulation_log.csv"
        if csv_p.exists():
            df = pd.read_csv(csv_p)
            self.fvm_t = torch.tensor(df['Time_min'].values[:,None]*60).float().to(self.device)
            self.fvm_T = torch.tensor(df['Avg_T_C'].values[:,None]+273.15).float().to(self.device)
            print(f"[PINN] Training with {len(df)} FVM points.")

    def compute_residuals(self, T, p, alpha, r, z, t):
        # Gradients
        dT = torch.autograd.grad(T.sum(), [r, z, t], create_graph=True)
        dT_dr, dT_dz, dT_dt = dT[0], dT[1], dT[2]
        dT_dr2 = torch.autograd.grad(dT_dr.sum(), r, create_graph=True)[0]
        dT_dz2 = torch.autograd.grad(dT_dz.sum(), z, create_graph=True)[0]
        
        dp = torch.autograd.grad(p.sum(), [r, z], create_graph=True)
        dp_dr, dp_dz = torch.clamp(dp[0], -1e7, 1e7), torch.clamp(dp[1], -1e7, 1e7)
        v_r = torch.clamp(-(self.cfg.PERMEABILITY/self.cfg.MU_GAS)*dp_dr, -5, 5)
        v_z = torch.clamp(-(self.cfg.PERMEABILITY/self.cfg.MU_GAS)*dp_dz, -5, 5)

        # Physics
        laplacian = dT_dr2 + (1.0/(r+1e-6))*dT_dr + dT_dz2
        res_e = (self.cfg.RHO_S_INIT*self.cfg.CP_EFF*(dT_dt + v_r*dT_dr + v_z*dT_dz)) - (self.cfg.K_EFF*laplacian)
        
        c = p / (self.cfg.R_GAS * torch.clamp(T, 250, 2000))
        res_m = torch.autograd.grad(c.sum(), t, create_graph=True)[0] # Simplified molar
        
        return (res_e**2 / 1e11) + (res_m**2 / 1e6)

    def train_round(self, rnd):
        model = WeakCorrectionNN().to(self.device)
        self.pinn.weak_learners.append(model)
        self.pinn.alphas.append(1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=10)

        for epoch in range(20):
            optimizer.zero_grad()
            T, p, a = self.pinn.predict(self.r, self.z, self.t)
            L_pde = self.compute_residuals(T, p, a, self.r, self.z, self.t)
            
            # Data Loss (Against FVM Averages)
            T_pred_bulk = T.mean() # Simplified mapping
            L_data = torch.mean((T_pred_bulk - self.fvm_T.mean())**2) if hasattr(self, 'fvm_T') else torch.tensor(0.0)
            
            loss = torch.mean(self.sample_w * L_pde) + L_data
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Val RMS
            with torch.no_grad():
                Tv, pv, av = self.pinn.predict(self.r_v, self.z_v, self.t_v)
                rms = torch.sqrt(torch.mean(Tv**2)).item()
                self.history['val_rms'].append(rms)
                self.history['train_loss'].append(loss.item())
            
            self.iter_log.write(f"{rnd*20+epoch}\t{rnd}\t{epoch}\t{loss.item():.2e}\t{L_data.item():.2e}\t{optimizer.param_groups[0]['lr']:.2e}\t{rms:.2f}\n")
            scheduler.step(loss)

        # AdaBoost.R2 Update
        with torch.no_grad():
            L_norm = torch.clamp(L_pde / (torch.quantile(L_pde, 0.95)+1e-10), 0, 1)
            err = torch.sum(self.sample_w * L_norm)
            beta = err / (1.0 - err + 1e-10)
            self.pinn.alphas[-1] = torch.log(1.0/(beta+1e-10)).item()
            self.sample_w *= torch.sqrt(beta**(1.0 - L_norm))
            self.sample_w = torch.clamp(self.sample_w, 1e-6, 1e2)
            self.sample_w /= self.sample_w.sum()

    def finalize(self):
        self.iter_log.close()
        # Loss Semilogy
        plt.figure()
        plt.semilogy(self.history['train_loss'])
        plt.title("Training Loss"); plt.savefig(self.run_dir/"loss.png"); plt.close()
        
        # Residual Histogram
        plt.figure()
        T, p, a = self.pinn.predict(self.r, self.z, self.t)
        res = self.compute_residuals(T, p, a, self.r, self.z, self.t).detach().cpu().numpy()
        plt.hist(np.log10(res + 1e-12), bins=50)
        plt.title("Log10 Residuals"); plt.savefig(self.run_dir/"res_hist.png"); plt.close()

        # Timestep Predictions
        for i, ts in enumerate(np.linspace(0, self.pinn.t_end, 10)):
            r_g = torch.linspace(0, self.pinn.R, 50).to(self.device)
            z_g = torch.linspace(0, self.pinn.L, 50).to(self.device)
            RR, ZZ = torch.meshgrid(r_g, z_g, indexing='ij')
            TT = torch.full_like(RR, ts)
            with torch.no_grad():
                Tp, pp, ap = self.pinn.predict(RR.reshape(-1,1), ZZ.reshape(-1,1), TT.reshape(-1,1))
            plt.imshow(Tp.reshape(50,50).cpu().T - 273.15, origin='lower', cmap='hot', vmin=40, vmax=800)
            plt.title(f"PINN @ {ts/60:.1f} min"); plt.savefig(self.run_dir/f"pred_{i}.png"); plt.close()

        # Sound
        os.system("powershell -c (New-Object Media.SoundPlayer 'C:\Windows\Media\notify.wav').PlaySync()")

def run_pinn_training(cfg, fvm_data_path):
    pinn = AdaBoostPINN(cfg)
    trainer = AdaBoostTrainer(pinn, fvm_data_path)
    for r in range(10):
        print(f"Round {r}..."); trainer.train_round(r)
    trainer.finalize()
    torch.save({'alphas': pinn.alphas, 'models': [m.state_dict() for m in pinn.weak_learners], 'cfg': cfg}, 
               os.path.join(fvm_data_path, "pinn_ensemble.pth"))