import numpy as np

class Mesh:
    def __init__(self, cfg):
        # JSON Spec: r from 0 to R, z from 0 to L
        self.R = cfg.R_MAX
        self.L = cfg.Z_MAX 
        
        self.dr = self.R / cfg.NR
        self.dz = self.L / cfg.NZ
        
        # 1. Face locations (JSON: Domain boundaries)
        self.r_f = np.linspace(0, self.R, cfg.NR + 1)
        self.z_f = np.linspace(0, self.L, cfg.NZ + 1)
        
        # 2. Cell Centers (JSON: Independent variables r, z)
        self.r_c = 0.5 * (self.r_f[:-1] + self.r_f[1:])
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        
        # 3. Geometry Metrics (For Axisymmetric Expanded Operators)
        self.V = np.zeros((cfg.NR, cfg.NZ))
        self.Af_w = np.zeros((cfg.NR, cfg.NZ)) # Inner Radial Face (West)
        self.Af_e = np.zeros((cfg.NR, cfg.NZ)) # Outer Radial Face (East)
        self.Af_s = np.zeros((cfg.NR, cfg.NZ)) # Bottom Axial Face (South)
        self.Af_n = np.zeros((cfg.NR, cfg.NZ)) # Top Axial Face (North)

        for i in range(cfg.NR):
            for j in range(cfg.NZ):
                # Volumetric calculation for a cylindrical ring
                # V = pi * (r_out^2 - r_in^2) * dz
                self.V[i,j] = np.pi * (self.r_f[i+1]**2 - self.r_f[i]**2) * self.dz
                
                # Radial Areas (2 * pi * r * dz)
                # JSON Note: r=0 (Af_w[0,j]) will be 0, satisfying the symmetry condition dT/dr=0
                self.Af_w[i,j] = 2.0 * np.pi * self.r_f[i] * self.dz
                self.Af_e[i,j] = 2.0 * np.pi * self.r_f[i+1] * self.dz
                
                # Axial Areas (pi * (r_out^2 - r_in^2))
                area_z = np.pi * (self.r_f[i+1]**2 - self.r_f[i]**2)
                self.Af_s[i,j] = area_z
                self.Af_n[i,j] = area_z

        # 4. Radial Midpoints for Source Terms
        # Useful for evaluating (1/r) terms at the cell center if needed
        self.R_grid, self.Z_grid = np.meshgrid(self.r_c, self.z_c, indexing='ij')