import numpy as np

class Config:
    # Physical Properties (Solid)
    K_EFF = 0.08
    RHO_S_INIT = 400.0
    CP_EFF = 1500.0
    R_GAS = 8.314
    
    # Gas Properties
    M_GAS = 0.030       # kg/mol
    MU_GAS = 3e-5       # Pa*s
    PERMEABILITY = 1e-11 
    POROSITY = 0.7
    P_ATM = 101325.0

    # Kinetics (Rice Husk)
    W0 = {'M': 0.10, 'HC': 0.20, 'C': 0.35, 'L': 0.25}
    KINETICS_PARAMS = {
        'M':  [5.13e6, 40000, 2.8],
        'HC': [9.67e9, 126310, 2.3],
        'C':  [3.50e12, 168610, 1.38],
        'L':  [2.59e5, 87210, 1.51]
    }

    # Geometry & Mesh
    R_MAX, Z_MAX = 0.010, 0.030
    NR, NZ = 20, 40

    # Simulation
    DT = 0.1            # Small DT for pressure-velocity coupling stability
    TOTAL_MIN = 10
    HEATING_RATE = 60.0
    T_INIT = 25.0 + 273.15
    H_CONV = 60.0