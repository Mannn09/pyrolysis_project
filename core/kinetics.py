import numpy as np

class KineticsEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def compute_step(self, alpha, T, dt, cell_volumes):
        """
        Calculates reaction progress, heat generation, and molar gas production.
        Matches JSON Spec: alpha_1, alpha_2, alpha_3, alpha_4
        """
        q_rxn = np.zeros_like(T)
        dn_moles_per_cell = np.zeros_like(T)
        
        rho_bulk = self.cfg.RHO_S_INIT 
        epsilon = self.cfg.POROSITY
        
        for i in range(1, 5):
            idx = f'alpha_{i}'
            # JSON params: [A, Ea, n, deltaH, f, eta]
            params = self.cfg.KINETICS_PARAMS[idx]
            A, Ea, n, dH, f, eta = params[0], params[1], params[2], params[3], params[4], params[5]
            
            # 1. Reaction Rate r_i [mol/(g*s)]
            # alpha starts at 1.0 and is consumed
            r_i = A * (np.maximum(alpha[idx], 1e-8)**n) * np.exp(-Ea / (self.cfg.R_GAS * T))
            
            # 2. Reaction Heat Source [W/m^3]
            q_rxn += dH * rho_bulk * r_i
            
            # 3. Molar Generation [mol per cell per step]
            # (1-epsilon) * rho_bulk * f * eta * r_i * V_cell * dt
            dn_moles_per_cell += (1.0 - epsilon) * rho_bulk * (f * eta * r_i) * cell_volumes * dt
            
            # 4. Update Reaction Progress (Consumption)
            alpha[idx] = np.maximum(alpha[idx] - r_i * dt, 0.0)
            
        return alpha, dn_moles_per_cell, q_rxn