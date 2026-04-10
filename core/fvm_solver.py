from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np

class PackedBedSolver:
    def __init__(self, cfg, mesh):
        self.cfg, self.mesh = cfg, mesh
        self.N = cfg.NR * cfg.NZ

    def solve_pressure(self, P_old, T, S_gas_total):
        """ Implicit FVM for Darcy Pressure: d(rho_g)/dt + div(rho_g*u) = S_gas """
        A = sparse.lil_matrix((self.N, self.N))
        B = np.zeros(self.N)
        
        # Diffusivity for Pressure (Darcy)
        coeff_D = (self.cfg.PERMEABILITY / self.cfg.MU_GAS) * (self.cfg.M_GAS / (self.cfg.R_GAS * T))
        trans_const = (self.cfg.POROSITY * self.cfg.M_GAS) / (self.cfg.R_GAS * T * self.cfg.DT)

        for i in range(self.cfg.NR):
            for j in range(self.cfg.NZ):
                idx = i * self.cfg.NZ + j
                
                # Top Boundary (Open to Atm)
                if j == self.cfg.NZ - 1:
                    A[idx, idx] = 1.0
                    B[idx] = self.cfg.P_ATM
                    continue

                aw, ae, asub, an = 0, 0, 0, 0
                if i > 0: aw = coeff_D[i,j] * self.mesh.Af_w[i,j] * P_old[i,j] / self.mesh.dr
                if i < self.cfg.NR-1: ae = coeff_D[i,j] * self.mesh.Af_e[i,j] * P_old[i,j] / self.mesh.dr
                if j > 0: asub = coeff_D[i,j] * self.mesh.Af_s[i,j] * P_old[i,j] / self.mesh.dz
                # North neighbor is j+1
                an = coeff_D[i,j] * self.mesh.Af_n[i,j] * P_old[i,j] / self.mesh.dz

                ap0 = trans_const[i,j] * self.mesh.V[i,j]
                ap = aw + ae + asub + an + ap0
                
                A[idx, idx] = ap
                if i > 0: A[idx, idx - self.cfg.NZ] = -aw
                if i < self.cfg.NR-1: A[idx, idx + self.cfg.NZ] = -ae
                if j > 0: A[idx, idx - 1] = -asub
                if j < self.cfg.NZ - 1: A[idx, idx + 1] = -an
                
                B[idx] = ap0 * P_old[i,j] + S_gas_total[i,j] * self.mesh.V[i,j]

        return spsolve(A.tocsr(), B).reshape((self.cfg.NR, self.cfg.NZ))

    def solve_heat(self, T_old, P, T_amb):
        """ Implicit FVM for Energy: Conduction + Advection """
        A = sparse.lil_matrix((self.N, self.N))
        B = np.zeros(self.N)
        
        # Calculate Darcy Velocities for Advection
        # u = -(K/mu) * grad(P)
        ur = np.zeros_like(P); uz = np.zeros_like(P)
        ur[1:-1, :] = -(self.cfg.PERMEABILITY/self.cfg.MU_GAS) * (P[2:,:] - P[:-2,:])/(2*self.mesh.dr)
        uz[:, 1:-1] = -(self.cfg.PERMEABILITY/self.cfg.MU_GAS) * (P[:,2:] - P[:,:-2])/(2*self.mesh.dz)
        
        rho_cp_v_dt = (self.cfg.RHO_S_INIT * self.cfg.CP_EFF * self.mesh.V) / self.cfg.DT
        
        for i in range(self.cfg.NR):
            for j in range(self.cfg.NZ):
                idx = i * self.cfg.NZ + j
                
                aw, ae, asub, an = 0, 0, 0, 0
                # Radial Conduction
                if i > 0: aw = self.cfg.K_EFF * self.mesh.Af_w[i,j] / self.mesh.dr
                if i < self.cfg.NR-1: ae = self.cfg.K_EFF * self.mesh.Af_e[i,j] / self.mesh.dr
                else: B[idx] += self.cfg.H_CONV * self.mesh.Af_e[i,j] * T_amb
                
                # Axial Conduction
                if j > 0: asub = self.cfg.K_EFF * self.mesh.Af_s[i,j] / self.mesh.dz
                else: B[idx] += self.cfg.H_CONV * self.mesh.Af_s[i,j] * T_amb
                
                if j < self.cfg.NZ-1: an = self.cfg.K_EFF * self.mesh.Af_n[i,j] / self.mesh.dz

                # Advection term (Simplified Upwind)
                # rho_g * Cp_g * u * Area * T
                rho_g = (P[i,j]*self.cfg.M_GAS)/(self.cfg.R_GAS*T_old[i,j])
                adv = rho_g * 1000.0 # Using 1000 as Cp for gas
                if ur[i,j] > 0: ae += adv * ur[i,j] * self.mesh.Af_e[i,j]
                else: aw += abs(adv * ur[i,j] * self.mesh.Af_w[i,j])
                
                ap0 = rho_cp_v_dt[i,j]
                ap = aw + ae + asub + an + ap0
                if i == self.cfg.NR-1: ap += self.cfg.H_CONV * self.mesh.Af_e[i,j]
                if j == 0: ap += self.cfg.H_CONV * self.mesh.Af_s[i,j]

                A[idx, idx] = ap
                if i > 0: A[idx, idx - self.cfg.NZ] = -aw
                if i < self.cfg.NR-1: A[idx, idx + self.cfg.NZ] = -ae
                if j > 0: A[idx, idx - 1] = -asub
                if j < self.cfg.NZ - 1: A[idx, idx + 1] = -an
                
                B[idx] += ap0 * T_old[i,j]

        return spsolve(A.tocsr(), B).reshape((self.cfg.NR, self.cfg.NZ))