import numpy as np

class KineticsEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def compute_step(self, W, T, dt):
        S_gas = np.zeros_like(T)
        for comp, (A, Ea, n) in self.cfg.KINETICS_PARAMS.items():
            rate = A * np.exp(-Ea / (self.cfg.R_GAS * T)) * (W[comp]**n)
            dm_dt = rate * self.cfg.RHO_S_INIT
            W[comp] = np.maximum(W[comp] - (rate * dt), 0)
            S_gas += dm_dt
        return W, S_gas