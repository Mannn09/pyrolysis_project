import numpy as np

class Mesh:
    def __init__(self, cfg):
        self.dr = cfg.R_MAX / cfg.NR
        self.dz = cfg.Z_MAX / cfg.NZ
        
        # Face locations
        self.r_f = np.linspace(0, cfg.R_MAX, cfg.NR + 1)
        self.z_f = np.linspace(0, cfg.Z_MAX, cfg.NZ + 1)
        
        # Cell Centers
        self.r_c = 0.5 * (self.r_f[:-1] + self.r_f[1:])
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        
        # Metrics: Volumes and Face Areas
        self.V = np.zeros((cfg.NR, cfg.NZ))
        self.Af_w = np.zeros((cfg.NR, cfg.NZ)) # West (Inner Radial)
        self.Af_e = np.zeros((cfg.NR, cfg.NZ)) # East (Outer Radial)
        self.Af_s = np.zeros((cfg.NR, cfg.NZ)) # South (Bottom)
        self.Af_n = np.zeros((cfg.NR, cfg.NZ)) # North (Top)

        for i in range(cfg.NR):
            for j in range(cfg.NZ):
                self.V[i,j] = np.pi * (self.r_f[i+1]**2 - self.r_f[i]**2) * self.dz
                self.Af_w[i,j] = 2 * np.pi * self.r_f[i] * self.dz
                self.Af_e[i,j] = 2 * np.pi * self.r_f[i+1] * self.dz
                area_z = np.pi * (self.r_f[i+1]**2 - self.r_f[i]**2)
                self.Af_s[i,j] = area_z
                self.Af_n[i,j] = area_z