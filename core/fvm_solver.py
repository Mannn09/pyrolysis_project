from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np

class PackedBedSolver:
    def __init__(self, cfg, mesh):
        self.cfg, self.mesh = cfg, mesh
        self.N = cfg.NR * cfg.NZ

    def solve_pressure(self, P_old, T, dn_generated):
        """ 
        Implicit FVM for Pressure based on Molar Continuity PDE.
        BCs (JSON): r=0 (Sym), r=R (Impermeable), z=0 (Impermeable), z=L (Dirichlet Outlet)
        """
        A = sparse.lil_matrix((self.N, self.N))
        B = np.zeros(self.N)
        
        molar_mobility = (self.cfg.PERMEABILITY * P_old) / (self.cfg.MU_GAS * self.cfg.R_GAS * T)
        molar_trans_const = self.cfg.POROSITY / (self.cfg.R_GAS * T * self.cfg.DT)

        for i in range(self.cfg.NR):
            for j in range(self.cfg.NZ):
                idx = i * self.cfg.NZ + j
                
                # --- BC: Top (z=L) Dirichlet Outlet (JSON Spec) ---
                if j == self.cfg.NZ - 1:
                    A[idx, idx] = 1.0
                    B[idx] = self.cfg.P_ATM # p0 in JSON
                    continue

                aw, ae, asub, an = 0, 0, 0, 0
                
                # Radial Flux (Darcy)
                if i > 0: 
                    aw = molar_mobility[i,j] * self.mesh.Af_w[i,j] / self.mesh.dr
                # BC: r=R (i=NR-1) is Impermeable Wall (ae remains 0)

                if i < self.cfg.NR - 1: 
                    ae = molar_mobility[i,j] * self.mesh.Af_e[i,j] / self.mesh.dr

                # Axial Flux (Darcy)
                if j > 0: 
                    asub = molar_mobility[i,j] * self.mesh.Af_s[i,j] / self.mesh.dz
                # BC: z=0 (j=0) is Impermeable Wall (asub remains 0 in flow logic)

                an = molar_mobility[i,j] * self.mesh.Af_n[i,j] / self.mesh.dz

                ap0 = molar_trans_const[i, j] * self.mesh.V[i, j]
                ap = aw + ae + asub + an + ap0
                
                A[idx, idx] = ap
                if i > 0:              A[idx, idx - self.cfg.NZ] = -aw
                if i < self.cfg.NR - 1: A[idx, idx + self.cfg.NZ] = -ae
                if j > 0:              A[idx, idx - 1] = -asub
                if j < self.cfg.NZ - 1: A[idx, idx + 1] = -an
                
                moles_source_rate = dn_generated[i, j] / self.cfg.DT
                B[idx] = (ap0 * P_old[i, j]) + moles_source_rate

        return spsolve(A.tocsr(), B).reshape((self.cfg.NR, self.cfg.NZ))

    def solve_heat(self, T_old, P, T_amb, q_rxn):
        """ 
        Implicit FVM for Energy Balance.
        Includes Advection + Conduction + Reaction Heat Source.
        BCs (JSON): r=0 (Sym), r=R (Dirichlet Ramp), z=0 (Insulated), z=L (Insulated)
        """
        A = sparse.lil_matrix((self.N, self.N))
        B = np.zeros(self.N)
        
        # Calculate Velocities from Darcy Law components (JSON Spec)
        ur = np.zeros_like(P); uz = np.zeros_like(P)
        ur[1:-1, :] = -(self.cfg.PERMEABILITY/self.cfg.MU_GAS) * (P[2:,:] - P[:-2,:])/(2*self.mesh.dr)
        uz[:, 1:-1] = -(self.cfg.PERMEABILITY/self.cfg.MU_GAS) * (P[:,2:] - P[:,:-2])/(2*self.mesh.dz)
        
        rho_cp_v_dt = (self.cfg.RHO_S_INIT * self.cfg.CP_EFF * self.mesh.V) / self.cfg.DT
        
        for i in range(self.cfg.NR):
            for j in range(self.cfg.NZ):
                idx = i * self.cfg.NZ + j
                
                # --- BC: Side (r=R) Dirichlet Boundary Heating (JSON Spec) ---
                if i == self.cfg.NR - 1:
                    A[idx, idx] = 1.0
                    B[idx] = T_amb # T0 + beta*t
                    continue

                aw, ae, asub, an = 0, 0, 0, 0
                
                # Radial Conduction
                if i > 0: aw = self.cfg.K_EFF * self.mesh.Af_w[i,j] / self.mesh.dr
                if i < self.cfg.NR - 1: ae = self.cfg.K_EFF * self.mesh.Af_e[i,j] / self.mesh.dr

                # Axial Conduction
                # BC: z=0 and z=L are Neumann Insulated (asub and an remain 0 at boundaries)
                if j > 0: asub = self.cfg.K_EFF * self.mesh.Af_s[i,j] / self.mesh.dz
                if j < self.cfg.NZ - 1: an = self.cfg.K_EFF * self.mesh.Af_n[i,j] / self.mesh.dz

                # Advection (Upwind scheme based on Darcy Velocity)
                rho_g = (P[i,j] * self.cfg.MW_GAS_AVG) / (self.cfg.R_GAS * T_old[i,j])
                adv_cap = rho_g * 1000.0 # Assumed Cp_gas
                
                if ur[i,j] > 0: ae += adv_cap * ur[i,j] * self.mesh.Af_e[i,j]
                else: aw += abs(adv_cap * ur[i,j] * self.mesh.Af_w[i,j])
                
                if uz[i,j] > 0: an += adv_cap * uz[i,j] * self.mesh.Af_n[i,j]
                else: asub += abs(adv_cap * uz[i,j] * self.mesh.Af_s[i,j])

                ap0 = rho_cp_v_dt[i,j]
                ap = aw + ae + asub + an + ap0
                
                A[idx, idx] = ap
                if i > 0:              A[idx, idx - self.cfg.NZ] = -aw
                if i < self.cfg.NR - 1: A[idx, idx + self.cfg.NZ] = -ae
                if j > 0:              A[idx, idx - 1] = -asub
                if j < self.cfg.NZ - 1: A[idx, idx + 1] = -an
                
                # Source: Accumulation + Reaction Heat Source (q_rxn = sum(dH * rho * r))
                B[idx] = (ap0 * T_old[i,j]) + (q_rxn[i,j] * self.mesh.V[i,j])

        return spsolve(A.tocsr(), B).reshape((self.cfg.NR, self.cfg.NZ))