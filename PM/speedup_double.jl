include("PMMP.jl")
include("PMExperiments.jl")



Nx = 100        # Spatial size
p = 3           # Order of the method, i.e, 2: IMR, 3:SDIRK3, 4:SDIRK4
maxcorr = 2      # Number of corrections
corr_id = "EIN" # EXP, EIN, JAC
run_time_mp_double(p, Nx; maxcorr = maxcorr, corr_id = corr_id)