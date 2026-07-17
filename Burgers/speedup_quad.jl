include("BurgersMP.jl")
include("BurgersExperiments.jl")



Nx = 50        # Spatial size
p = 4           # Order of the method, i.e, 2: IMR, 3:SDIRK3, 4:SDIRK4
maxcorr = 1     # Number of corrections
corr_id = "JAC" # EXP, EIN, JAC
run_time_mp_quad(p, Nx; maxcorr = maxcorr, corr_id = corr_id)