import numpy as np

class Config:
    # --- 1. PHYSICAL PROPERTIES (Solid & Porous Medium) ---
    # JSON: (rhoCp)_eff and k_eff
    RHO_S_INIT = 550.0    # rho_bulk [g/m^3] or [kg/m^3] (must match Ea/R units)
    CP_EFF = 1900.0       # Cp_eff [J/kg*K]
    K_EFF = 0.07          # k_eff [W/m*K]
    
    # JSON: Porous medium parameters
    POROSITY = 0.1        # epsilon (porosity)
    PERMEABILITY = 1e-13  # K (Darcy permeability)

    # --- 2. FLUID & GAS PROPERTIES ---
    # JSON: mu (viscosity) and R (gas constant)
    MU_GAS = 3e-5        # mu (dynamic viscosity) [Pa*s]
    R_GAS = 8.314         # universal gas constant [J/mol*K]
    MW_GAS_AVG = 0.030    # Average molecular weight for results tracking [kg/mol]

    # --- 3. REFERENCE CONDITIONS ---
    # JSON: T0 = 40 degC, p0 = 1 atm
    T_INIT = 40.0 + 273.15 # T0 [K]
    P_ATM = 101325.0       # p0 [Pa]

    # --- 4. REACTION PARAMETERS (JSON Spec alpha_1 to alpha_4) ---
    # Format: [A, Ea, n, deltaH_rxn, f, eta]
    # A: Pre-exponential, Ea: Activation Energy, n: Reaction Order
    # deltaH_rxn: Heat of Reaction, f: gas fraction, eta: stoichiometric coeff
    # Ensure these keys match alpha_1, alpha_2, etc.
    KINETICS_PARAMS = {
        'alpha_1': [5.13e6,  40000,  2.8,   2260.0, 0.0665, 0.056], # Moisture (Latent Heat)
        'alpha_2': [9.67e9,  126310, 2.3,  282.5,  0.4082, 0.101], # Hemicellulose (Net Exothermic)
        'alpha_3': [3.50e12, 168610, 1.38,  -490.0,  0.2515, 0.057], # Cellulose (Net Endothermic)
        'alpha_4': [2.59e5,  87210,  1.51, -650.0,  0.2538, 0.202]  # Lignin (Strongly Exothermic)
    }
    # Initial Reaction Progress (0.0 = Unreacted, 1.0 = Fully Reacted)
    ALPHA_INIT = 0.0

    # --- 5. GEOMETRY & MESH (JSON Spec R and L) ---
    R_MAX = 0.040         # R [m]
    Z_MAX = 0.010         # L [m]
    NR, NZ = 80, 20

    # --- 6. SIMULATION & BOUNDARY HEATING ---
    # JSON: beta (heating ramp)
    DT = 0.5              # Time step [s]
    TOTAL_MIN = 40        # Total simulation time
    HEATING_RATE = 20.0   # beta (heating ramp) [degC/min]
    
    # Heat Transfer Coefficient (for external convection if used, 
    # though JSON implies Dirichlet for r=R)
    H_CONV = 8.5