include("PMMP.jl")
using DataFrames
using XLSX
using Statistics
using LinearAlgebra
using Serialization

# === path/headless setup ===
const BASEDIR = @__DIR__
ENV["GKSwstype"] = "100"
# ===========================


function run_experiment_FP(p::Int, Nx::Int, T::Type)

    # --- Experiment Parameters ---
    Nt_values = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    tFinal = 0.5

    # Generate experimental arrays natively in precision T
    u, x, dx, Dxx_high, Dxx_low = setup_problem(Nx, T)

    # Reference solution file path
    ref_file = joinpath(BASEDIR, "Solutions/Ref_sol_$(Nx).jls")

    if isfile(ref_file)
        println("Loading reference solution from $(ref_file)...")
        u_ref = deserialize(ref_file)
    else
        println("Reference solution not found. Computing now (this may take a while)...")
        u_ref_init, _, _, _, _ = setup_problem(Nx, Float128)
        Nt_ref = 500000
        dt_ref = Float128(tFinal) / Float128(Nt_ref)
        u_ref = rk4(u_ref_init; Nt = Nt_ref, dt = dt_ref)

        if !isdir(joinpath(BASEDIR, "Solutions"))
            mkpath(joinpath(BASEDIR, "Solutions"))
        end
        serialize(ref_file, u_ref)
        println("Reference solution saved to $(ref_file)")
    end

    dts = Float64[]
    errors_fp = Float64[]
    file_name_fp = " "

    for Nt in Nt_values
        dt = T(tFinal) / T(Nt)
        push!(dts, Float64(dt))

        u_fp = nothing
        if p == 2
            u_fp, = IMR_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
            file_name_fp = joinpath(BASEDIR, "Errors/IMR_fp_Nx$(Nx)_$(T).jls")
        elseif p == 3
            u_fp, = SDIRK3_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
            file_name_fp = joinpath(BASEDIR, "Errors/SDIRK3_fp_Nx$(Nx)_$(T).jls")
        elseif p == 4
            u_fp, = SDIRK4_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
            file_name_fp = joinpath(BASEDIR, "Errors/SDIRK4_fp_Nx$(Nx)_$(T).jls")
        else
            u_fp, = IMR_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
            file_name_fp = joinpath(BASEDIR, "Errors/IMR_fp_Nx$(Nx)_$(T).jls")
        end

        err = norm(Float128.(Array(u_fp[end, :])) .- u_ref, Inf)
        push!(errors_fp, Float64(err))
    end

    if !isdir(joinpath(BASEDIR, "Errors"))
        mkpath(joinpath(BASEDIR, "Errors"))
    end
    serialize(file_name_fp, errors_fp)

    print("Slopes: ")
    println(diff(log10.(errors_fp)) ./ diff(log10.(dts)))
end


function run_experiment_MP(p::Int, Nx::Int; maxcorr::Int = 0, corr_id::String = "JAC", T::Type, Tr::Type)

    if !(corr_id in ["JAC", "EIN", "EXP"])
        @warn "Invalid corr_id '$corr_id' provided. Defaulting to 'JAC'."
        corr_id = "JAC"
    end

    Nt_values = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    tFinal = 0.5

    u, x, dx, Dxx_high, Dxx_low = setup_problem(Nx, T, Tr)

    ref_file = joinpath(BASEDIR, "Solutions/Ref_sol_$(Nx).jls")

    if isfile(ref_file)
        println("Loading reference solution from $(ref_file)...")
        u_ref = deserialize(ref_file)
    else
        println("Reference solution not found. Computing now (this may take a while)...")
        u_ref_init, _, _, _, _ = setup_problem(Nx, Float128)
        Nt_ref = 500000
        dt_ref = Float128(tFinal) / Float128(Nt_ref)
        u_ref = rk4(u_ref_init; Nt = Nt_ref, dt = dt_ref)

        if !isdir(joinpath(BASEDIR, "Solutions"))
            mkpath(joinpath(BASEDIR, "Solutions"))
        end
        serialize(ref_file, u_ref)
        println("Reference solution saved to $(ref_file)")
    end

    dts = Float64[]
    errors_mp = Float64[]
    file_name_mp = " "

    for Nt in Nt_values
        dt = T(tFinal) / T(Nt)
        push!(dts, Float64(dt))

        u_mp = nothing
        if p == 2
            u_mp, = IMR_mp(u, Dxx_high, Dxx_low; Nx = Nx, Nt = Nt, dt = dt, maxcorr = maxcorr, corr_id = corr_id)
            if maxcorr == 0
                file_name_mp = joinpath(BASEDIR, "Errors/IMR_mp_Nx$(Nx)_$(T)_$(Tr).jls")
            else
                file_name_mp = joinpath(BASEDIR, "Errors/IMR_mp_Nx$(Nx)_$(T)_$(Tr)_$(maxcorr)_$(corr_id).jls")
            end
        elseif p == 3
            u_mp, = SDIRK3_mp(u, Dxx_high, Dxx_low; Nx = Nx, Nt = Nt, dt = dt, maxcorr = maxcorr, corr_id = corr_id)
            if maxcorr == 0
                file_name_mp = joinpath(BASEDIR, "Errors/SDIRK3_mp_Nx$(Nx)_$(T)_$(Tr).jls")
            else
                file_name_mp = joinpath(BASEDIR, "Errors/SDIRK3_mp_Nx$(Nx)_$(T)_$(Tr)_$(maxcorr)_$(corr_id).jls")
            end
        elseif p == 4
            u_mp, = SDIRK4_mp(u, Dxx_high, Dxx_low; Nx = Nx, Nt = Nt, dt = dt, maxcorr = maxcorr, corr_id = corr_id)
            if maxcorr == 0
                file_name_mp = joinpath(BASEDIR, "Errors/SDIRK4_mp_Nx$(Nx)_$(T)_$(Tr).jls")
            else
                file_name_mp = joinpath(BASEDIR, "Errors/SDIRK4_mp_Nx$(Nx)_$(T)_$(Tr)_$(maxcorr)_$(corr_id).jls")
            end
        else
            u_mp, = IMR_mp(u, Dxx_high, Dxx_low; Nx = Nx, Nt = Nt, dt = dt, maxcorr = maxcorr, corr_id = corr_id)
            if maxcorr == 0
                file_name_mp = joinpath(BASEDIR, "Errors/IMR_mp_Nx$(Nx)_$(T)_$(Tr).jls")
            else
                file_name_mp = joinpath(BASEDIR, "Errors/IMR_mp_Nx$(Nx)_$(T)_$(Tr)_$(maxcorr)_$(corr_id).jls")
            end
        end

        err = norm(Float128.(Array(u_mp[end, :])) .- u_ref, Inf)
        push!(errors_mp, Float64(err))
    end

    if !isdir(joinpath(BASEDIR, "Errors"))
        mkpath(joinpath(BASEDIR, "Errors"))
    end
    serialize(file_name_mp, errors_mp)

    print("Slopes: ")
    println(diff(log10.(errors_mp)) ./ diff(log10.(dts)))
end


# plot_time 
#
# Each combo is cached to its own .jls under Times/. Existing .xlsx
# files are read as a backward-compat fallback and promoted to .jls
# on first hit. The .xlsx summary is rewritten at the end from all
# cached data.


function plot_time(p::Int, Nx::Int; maxcorr::Int = 0, corr_id::String = "JAC")
    if !(corr_id in ["JAC", "EIN", "EXP"])
        @warn "Invalid corr_id '$corr_id' provided. Defaulting to 'JAC'."
        corr_id = "JAC"
    end

    if p == 2
        method = "IMR"
    elseif p == 3
        method = "SDIRK3"
    elseif p == 4
        method = "SDIRK4"
    else
        method = "IMR"
    end

    Nt_values = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    tFinal = 0.5
    NUM_RUNS = 101
    Nt_ref = 500000

    if !isdir(joinpath(BASEDIR, "Solutions")); mkpath(joinpath(BASEDIR, "Solutions")); end
    if !isdir(joinpath(BASEDIR, "Times"));     mkpath(joinpath(BASEDIR, "Times"));     end
    if !isdir(joinpath(BASEDIR, "Results"));   mkpath(joinpath(BASEDIR, "Results"));   end

    println("--- Starting plot_time benchmark (Porous Medium) ---")
    println("Method: $(method), Nx: $(Nx), MaxCorr: $(maxcorr), CorrID: $(corr_id)")

    combos = [
        (Float128, Float128, "128/128", :black),
        (Float128, Float64,  "128/64",  :red),
        (Float128, Float32,  "128/32",  :green),
        (Float128, Float16,  "128/16",  :purple),
        (Float64,  Float64,  "64/64",   :blue),
        (Float64,  Float32,  "64/32",   :brown),
        (Float64,  Float16,  "64/16",   :magenta),
        (Float32,  Float32,  "32/32",   :cyan),
        (Float32,  Float16,  "32/16",   :olive),
        (Float16,  Float16,  "16/16",   :orange),
    ]

    old_names = Dict(
        "128/128" => "Float128_Float128",
        "128/64"  => "Float128_Float64",
        "128/32"  => "Float128_Float32",
        "128/16"  => "Float128_Float16",
        "64/64"   => "Float64_Float64",
        "64/32"   => "Float64_Float32",
        "64/16"   => "Float64_Float16",
        "32/32"   => "Float32_Float32",
        "32/16"   => "Float32_Float16",
        "16/16"   => "Float16_Float16",
    )

    excel_filename = maxcorr == 0 ?
        joinpath(BASEDIR, "Times/PlotTime_$(method)_Nx$(Nx).xlsx") :
        joinpath(BASEDIR, "Times/PlotTime_$(method)_Nx$(Nx)_$(maxcorr)_$(corr_id).xlsx")

    jls_filename(lbl, HighT, LowT) = begin
        short = replace(lbl, "/" => "_")
        # Pure precision or maxcorr=0 shares the same base file
        if HighT == LowT || maxcorr == 0
            joinpath(BASEDIR, "Times/PlotTime_$(method)_Nx$(Nx)_$(short).jls")
        else
            joinpath(BASEDIR, "Times/PlotTime_$(method)_Nx$(Nx)_$(maxcorr)_$(corr_id)_$(short).jls")
        end
    end

    all_results = Dict{String, NamedTuple}()

    # Stage 1: per-combo .jls
    for (HighT, LowT, lbl, col) in combos
        fname = jls_filename(lbl, HighT, LowT)
        if isfile(fname)
            try
                data = deserialize(fname)
                all_results[lbl] = (times = data.times, errors = data.errors, color = col)
                println("  Loaded $lbl from $fname")
            catch e
                @warn "Failed to read $fname; will rerun. ($e)"
            end
        end
    end

    # Stage 2: .xlsx fallback (for backward compatibility); promote to .jls if found
    if length(all_results) < length(combos) && isfile(excel_filename)
        println("\nFound .xlsx cache: $excel_filename — checking for missing combos there.")
        try
            xf = XLSX.readxlsx(excel_filename)
            available = XLSX.sheetnames(xf)

            for (HighT, LowT, lbl, col) in combos
                haskey(all_results, lbl) && continue
                short_name = replace(lbl, "/" => "_")
                long_name  = old_names[lbl]
                sheet_name = if short_name in available
                    short_name
                elseif long_name in available
                    long_name
                else
                    continue
                end

                df = DataFrame(XLSX.gettable(xf[sheet_name]))
                times_vec  = Float64[]
                errors_vec = Float64[]
                for row in eachrow(df)
                    t = row.Time
                    e = row.Error
                    push!(times_vec,  t isa Number ? Float64(t) : NaN)
                    push!(errors_vec, e isa Number ? Float64(e) : NaN)
                end
                all_results[lbl] = (times = times_vec, errors = errors_vec, color = col)
                serialize(jls_filename(lbl, HighT, LowT), (times = times_vec, errors = errors_vec))
                println("  Loaded $lbl from .xlsx and promoted to .jls")
            end
        catch e
            if e isa SystemError && occursin("opening file", e.prefix)
                @warn "Cache .xlsx '$excel_filename' could not be opened (likely open in Excel)."
            else
                @warn "Failed to read .xlsx cache ($e). Will rerun missing combos."
            end
        end
    end

    # Stage 3: run any combos still missing
    if length(all_results) < length(combos)
        ref_file = joinpath(BASEDIR, "Solutions/Ref_sol_$(Nx).jls")
        if isfile(ref_file)
            println("  Loading reference solution from $ref_file...")
            u_ref = deserialize(ref_file)
        else
            println("  Computing reference solution (Float128, Nt=$Nt_ref)...")
            u_ref_init, _, _, _, _ = setup_problem(Nx, Float128, Float128)
            dt_ref_val = Float128(tFinal) / Float128(Nt_ref)
            u_ref = rk4(u_ref_init; Nt=Nt_ref, dt=dt_ref_val)
            serialize(ref_file, u_ref)
        end
        u_ref_final = u_ref isa Matrix ? Vector{Float128}(u_ref[end, :]) : Vector{Float128}(u_ref)

        for (HighT, LowT, lbl, col) in combos
            haskey(all_results, lbl) && continue
            println("\n  >>> Precision combo: $lbl")

            local u, Dxx_high, Dxx_low
            try
                u, _, _, Dxx_high, Dxx_low = setup_problem(Nx, HighT, LowT)
            catch e
                @warn "setup_problem failed for $lbl: $e"
                continue
            end

            is_mixed = (HighT != LowT)
            times_vec  = Float64[]
            errors_vec = Float64[]

            for Nt in Nt_values
                print("    Nt = $Nt ... ")
                dt = HighT(tFinal) / HighT(Nt)

                times = zeros(NUM_RUNS)
                u_final = nothing
                failed = false

                for r in 1:NUM_RUNS
                    t_elapsed = @elapsed begin
                        try
                            local u_sol
                            if is_mixed
                                if p == 2
                                    u_sol, = IMR_mp(u, Dxx_high, Dxx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                                elseif p == 3
                                    u_sol, = SDIRK3_mp(u, Dxx_high, Dxx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                                elseif p == 4
                                    u_sol, = SDIRK4_mp(u, Dxx_high, Dxx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                                else
                                    u_sol, = IMR_mp(u, Dxx_high, Dxx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                                end
                            else
                                if p == 2
                                    u_sol, = IMR_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
                                elseif p == 3
                                    u_sol, = SDIRK3_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
                                elseif p == 4
                                    u_sol, = SDIRK4_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
                                else
                                    u_sol, = IMR_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
                                end
                            end

                            if r == NUM_RUNS
                                u_final = u_sol isa AbstractMatrix ? u_sol[end, :] : u_sol
                            end
                        catch e
                            @warn "run failed" exception=(e, catch_backtrace())
                            failed = true
                        end
                    end
                    times[r] = t_elapsed
                    if failed; break; end
                end

                if failed || u_final === nothing
                    println("FAILED (likely overflow in low precision)")
                    push!(times_vec,  NaN)
                    push!(errors_vec, NaN)
                    continue
                end

                t_mean = mean(times[2:end])
                err    = norm(Float128.(u_final) .- u_ref_final, Inf)
                push!(times_vec,  t_mean)
                push!(errors_vec, Float64(err))
                @printf("time=%.4es, err=%.3e\n", t_mean, Float64(err))
            end

            all_results[lbl] = (times = times_vec, errors = errors_vec, color = col)
            serialize(jls_filename(lbl, HighT, LowT), (times = times_vec, errors = errors_vec))
            println("  Saved $lbl to $(jls_filename(lbl, HighT, LowT))")
        end

        sheets = Pair{String, DataFrame}[]
        for (HighT, LowT, lbl, col) in combos
            if !haskey(all_results, lbl); continue; end
            r = all_results[lbl]
            df = DataFrame(Nt = Nt_values, Time = r.times, Error = r.errors)
            sheet_name = replace(lbl, "/" => "_")
            push!(sheets, sheet_name => df)
        end
        if !isempty(sheets)
            try
                XLSX.writetable(excel_filename, sheets...; overwrite=true)
                println("\nSaved Excel summary: $excel_filename")
            catch e
                @warn "Failed to write .xlsx (file may be open in Excel). .jls files are still saved. ($e)"
            end
        end
    else
        println("\nAll combos loaded from cache; no benchmarks run.")
    end

    p_main = plot(
        xscale = :log10,
        yscale = :log10,
        xlabel = L"\mathrm{Runtime\ (s)}",
        ylabel = L"\mathrm{Error}",
        guidefontsize = 20,
        tickfontsize = 12,
        legend = :bottomleft,
        legendfontsize = 10,
        grid = true,
        gridlinewidth = 1,
        gridalpha = 0.35,
        gridstyle = :solid,
        minorgrid = false,
        size = (800, 700),
        framestyle = :box,
        foreground_color_border = :black,
        foreground_color_axis   = :black,
        foreground_color_text   = :black,
        foreground_color_guide  = :black,
        yticks = 10.0 .^ (-15:2:-1),
        xticks = 10.0 .^ (-5:1:2),
        left_margin   = 8Plots.mm,
        right_margin  = 4Plots.mm,
        bottom_margin = 6Plots.mm,
        top_margin    = 4Plots.mm,
    )

    for (HighT, LowT, lbl, col) in combos
        if !haskey(all_results, lbl); continue; end
        r = all_results[lbl]

        valid = .!isnan.(r.times) .& .!isnan.(r.errors)
        t_plot = r.times[valid]
        e_plot = r.errors[valid]

        if isempty(t_plot); continue; end

        plot!(p_main, t_plot, e_plot,
            label = lbl,
            color = col,
            linestyle = :solid,
            linewidth = 2,
            marker = :circle,
            markercolor = col,
            markerstrokecolor = col,
            markersize = 5
        )
    end

    if maxcorr == 0
        savefig(p_main, joinpath(BASEDIR, "Results/Figure_PlotTime_$(method)_Nx$(Nx).png"))
    else
        savefig(p_main, joinpath(BASEDIR, "Results/Figure_PlotTime_$(method)_Nx$(Nx)_$(maxcorr)_$(corr_id).png"))
    end

    display(p_main)
    readline()
end





# run_time_mp_quad / run_time_mp_double
#
# One .jls per (Nx, Nt, combo). Pure-precision baselines (128/128, 64/64)
# are shared across all maxcorr/corr_id calls. Speedups are recomputed
# from cached times so they stay consistent within a call.


function _bench_combo_cached(p::Int,
                              u, Dxx_high, Dxx_low,  # Dxx_low = nothing for _fp baseline
                              Nx::Int, Nt::Int, dt,
                              maxcorr::Int, corr_id::String,
                              NUM_RUNS::Int, u_ref_final,
                              jls_path::String, combo_label::String)

    # ---- Try cache ----
    if isfile(jls_path)
        try
            data = deserialize(jls_path)
            println("    Loaded $combo_label from $jls_path")
            return data.time, data.error
        catch e
            @warn "Failed to read $jls_path; will rerun. ($e)"
        end
    end

    # ---- Run benchmark ----
    is_mixed = (Dxx_low !== nothing)
    times = zeros(NUM_RUNS)
    u_final = nothing

    for r in 1:NUM_RUNS
        times[r] = @elapsed begin
            local u_sol
            if is_mixed
                if p == 2
                    u_sol, = IMR_mp(u, Dxx_high, Dxx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                elseif p == 3
                    u_sol, = SDIRK3_mp(u, Dxx_high, Dxx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                elseif p == 4
                    u_sol, = SDIRK4_mp(u, Dxx_high, Dxx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                else
                    u_sol, = IMR_mp(u, Dxx_high, Dxx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                end
            else
                if p == 2
                    u_sol, = IMR_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
                elseif p == 3
                    u_sol, = SDIRK3_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
                elseif p == 4
                    u_sol, = SDIRK4_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
                else
                    u_sol, = IMR_fp(u, Dxx_high; Nx=Nx, Nt=Nt, dt=dt)
                end
            end
            if r == NUM_RUNS
                u_final = u_sol isa AbstractMatrix ? u_sol[end, :] : u_sol
            end
        end
    end

    t = mean(times[2:end])
    e = Float64(norm(Float128.(u_final) .- u_ref_final, Inf))

    try
        serialize(jls_path, (time = t, error = e))
        println("    Saved $combo_label to $jls_path")
    catch e
        @warn "Failed to write $jls_path ($e). Result will be in Excel only."
    end

    return t, e
end


function run_time_mp_quad(p::Int, Nx::Int; maxcorr::Int = 0, corr_id::String = "JAC")
    if p == 2
        method = "IMR"
    elseif p == 3
        method = "SDIRK3"
    elseif p == 4
        method = "SDIRK4"
    else
        method = "IMR"
    end

    Nt_values = [50, 500, 5000]
    tFinal = 0.5
    NUM_RUNS = 101
    Nt_ref = 500000

    if !isdir(joinpath(BASEDIR, "Solutions")); mkpath(joinpath(BASEDIR, "Solutions")); end
    if !isdir(joinpath(BASEDIR, "Times"));     mkpath(joinpath(BASEDIR, "Times"));     end

    println("--- run_time_mp_quad (Porous Medium) ---")
    println("Method: $(method), Nx: $(Nx), MaxCorr: $(maxcorr), CorrID: $(corr_id)")
    println("Combos: 128/128 (baseline, _fp), 128/32 (_mp), 128/16 (_mp)")

    ref_file = joinpath(BASEDIR, "Solutions/Ref_sol_$(Nx).jls")
    if isfile(ref_file)
        println("  Loading reference solution from $ref_file...")
        u_ref = deserialize(ref_file)
    else
        println("  Computing reference (Float128, Nt=$Nt_ref)...")
        u_ref_init, _, _, _, _ = setup_problem(Nx, Float128, Float128)
        dt_ref_val = Float128(tFinal) / Float128(Nt_ref)
        u_ref = rk4(u_ref_init; Nt=Nt_ref, dt=dt_ref_val)
        serialize(ref_file, u_ref)
    end
    u_ref_final = u_ref isa Matrix ? Vector{Float128}(u_ref[end, :]) : Vector{Float128}(u_ref)

    u, _, dx_val, Dxx_high, Dxx_low_32 = setup_problem(Nx, Float128, Float32)
    _, _, _,      _,         Dxx_low_16 = setup_problem(Nx, Float128, Float16)

    Nx_col = Int[]
    Nt_col = Int[]
    dx_col = Float64[]
    dt_col = Float64[]

    time_128_128_col  = Float64[]
    time_128_32_col   = Float64[]
    time_128_16_col   = Float64[]
    error_128_128_col = Float64[]
    error_128_32_col  = Float64[]
    error_128_16_col  = Float64[]
    sp_128_32_col     = Float64[]
    sp_128_16_col     = Float64[]

    for Nt in Nt_values
        println("  Running Nt = $Nt ...")
        push!(Nx_col, Nx)
        push!(Nt_col, Nt)
        dt = Float128(tFinal) / Float128(Nt)
        push!(dt_col, Float64(dt))
        push!(dx_col, Float64(dx_val))

        # Build per-combo cache paths for this Nt
        jls_128_128 = joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_128_128.jls")
        jls_128_32  = maxcorr == 0 ?
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_128_32.jls") :
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_$(maxcorr)_$(corr_id)_128_32.jls")
        jls_128_16  = maxcorr == 0 ?
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_128_16.jls") :
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_$(maxcorr)_$(corr_id)_128_16.jls")

        # ---- 128/128 (pure, _fp) ----
        t_128128, e_128128 = _bench_combo_cached(
            p, u, Dxx_high, nothing,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_128_128, "128/128")

        # ---- 128/32 (mp) ----
        t_12832, e_12832 = _bench_combo_cached(
            p, u, Dxx_high, Dxx_low_32,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_128_32, "128/32")

        # ---- 128/16 (mp) ----
        t_12816, e_12816 = _bench_combo_cached(
            p, u, Dxx_high, Dxx_low_16,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_128_16, "128/16")

        sp_32 = t_128128 / t_12832
        sp_16 = t_128128 / t_12816

        push!(time_128_128_col,  t_128128);  push!(error_128_128_col, e_128128)
        push!(time_128_32_col,   t_12832);   push!(error_128_32_col,  e_12832)
        push!(time_128_16_col,   t_12816);   push!(error_128_16_col,  e_12816)
        push!(sp_128_32_col,     sp_32)
        push!(sp_128_16_col,     sp_16)
    end

    results_df = DataFrame(
        Nx             = Nx_col,
        Nt             = Nt_col,
        dx             = dx_col,
        dt             = dt_col,
        Time_128_128   = time_128_128_col,
        Error_128_128  = error_128_128_col,
        Time_128_32    = time_128_32_col,
        Error_128_32   = error_128_32_col,
        Speedup_128_32 = sp_128_32_col,
        Time_128_16    = time_128_16_col,
        Error_128_16   = error_128_16_col,
        Speedup_128_16 = sp_128_16_col,
    )

    if maxcorr == 0
        filename = joinpath(BASEDIR, "Times/$(method)_mp_quad_Nx$(Nx).xlsx")
    else
        filename = joinpath(BASEDIR, "Times/$(method)_mp_quad_Nx$(Nx)_$(maxcorr)_$(corr_id).xlsx")
    end
    try
        XLSX.writetable(filename, "Results" => results_df, overwrite=true)
        println("\nSaved results to: $filename")
    catch e
        @warn "Failed to write .xlsx ($e). .jls files are still saved."
    end
end


function run_time_mp_double(p::Int, Nx::Int; maxcorr::Int = 0, corr_id::String = "JAC")
    if p == 2
        method = "IMR"
    elseif p == 3
        method = "SDIRK3"
    elseif p == 4
        method = "SDIRK4"
    else
        method = "IMR"
    end

    Nt_values = [50, 500, 5000]
    tFinal = 0.5
    NUM_RUNS = 101
    Nt_ref = 500000

    if !isdir(joinpath(BASEDIR, "Solutions")); mkpath(joinpath(BASEDIR, "Solutions")); end
    if !isdir(joinpath(BASEDIR, "Times"));     mkpath(joinpath(BASEDIR, "Times"));     end

    println("--- run_time_mp_double (Porous Medium) ---")
    println("Method: $(method), Nx: $(Nx), MaxCorr: $(maxcorr), CorrID: $(corr_id)")
    println("Combos: 64/64 (baseline, _fp), 64/32 (_mp), 64/16 (_mp)")

    ref_file = joinpath(BASEDIR, "Solutions/Ref_sol_$(Nx).jls")
    if isfile(ref_file)
        println("  Loading reference solution from $ref_file...")
        u_ref = deserialize(ref_file)
    else
        println("  Computing reference (Float128, Nt=$Nt_ref)...")
        u_ref_init, _, _, _, _ = setup_problem(Nx, Float128, Float128)
        dt_ref_val = Float128(tFinal) / Float128(Nt_ref)
        u_ref = rk4(u_ref_init; Nt=Nt_ref, dt=dt_ref_val)
        serialize(ref_file, u_ref)
    end
    u_ref_final = u_ref isa Matrix ? Vector{Float128}(u_ref[end, :]) : Vector{Float128}(u_ref)

    u, _, dx_val, Dxx_high, Dxx_low_32 = setup_problem(Nx, Float64, Float32)
    _, _, _,      _,         Dxx_low_16 = setup_problem(Nx, Float64, Float16)

    Nx_col = Int[]
    Nt_col = Int[]
    dx_col = Float64[]
    dt_col = Float64[]

    time_64_64_col  = Float64[]
    time_64_32_col  = Float64[]
    time_64_16_col  = Float64[]
    error_64_64_col = Float64[]
    error_64_32_col = Float64[]
    error_64_16_col = Float64[]
    sp_64_32_col    = Float64[]
    sp_64_16_col    = Float64[]

    for Nt in Nt_values
        println("  Running Nt = $Nt ...")
        push!(Nx_col, Nx)
        push!(Nt_col, Nt)
        dt = Float64(tFinal) / Float64(Nt)
        push!(dt_col, dt)
        push!(dx_col, Float64(dx_val))

        # Build per-combo cache paths for this Nt
        jls_64_64 = joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_64_64.jls")
        jls_64_32 = maxcorr == 0 ?
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_64_32.jls") :
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_$(maxcorr)_$(corr_id)_64_32.jls")
        jls_64_16 = maxcorr == 0 ?
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_64_16.jls") :
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_$(maxcorr)_$(corr_id)_64_16.jls")

        # ---- 64/64 (pure, _fp) ----
        t_6464, e_6464 = _bench_combo_cached(
            p, u, Dxx_high, nothing,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_64_64, "64/64")

        # ---- 64/32 (mp) ----
        t_6432, e_6432 = _bench_combo_cached(
            p, u, Dxx_high, Dxx_low_32,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_64_32, "64/32")

        # ---- 64/16 (mp) ----
        t_6416, e_6416 = _bench_combo_cached(
            p, u, Dxx_high, Dxx_low_16,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_64_16, "64/16")

        sp_32 = t_6464 / t_6432
        sp_16 = t_6464 / t_6416

        push!(time_64_64_col,  t_6464);  push!(error_64_64_col, e_6464)
        push!(time_64_32_col,  t_6432);  push!(error_64_32_col, e_6432)
        push!(time_64_16_col,  t_6416);  push!(error_64_16_col, e_6416)
        push!(sp_64_32_col,    sp_32)
        push!(sp_64_16_col,    sp_16)
    end

    results_df = DataFrame(
        Nx            = Nx_col,
        Nt            = Nt_col,
        dx            = dx_col,
        dt            = dt_col,
        Time_64_64    = time_64_64_col,
        Error_64_64   = error_64_64_col,
        Time_64_32    = time_64_32_col,
        Error_64_32   = error_64_32_col,
        Speedup_64_32 = sp_64_32_col,
        Time_64_16    = time_64_16_col,
        Error_64_16   = error_64_16_col,
        Speedup_64_16 = sp_64_16_col,
    )

    if maxcorr == 0
        filename = joinpath(BASEDIR, "Times/$(method)_mp_double_Nx$(Nx).xlsx")
    else
        filename = joinpath(BASEDIR, "Times/$(method)_mp_double_Nx$(Nx)_$(maxcorr)_$(corr_id).xlsx")
    end
    try
        XLSX.writetable(filename, "Results" => results_df, overwrite=true)
        println("\nSaved results to: $filename")
    catch e
        @warn "Failed to write .xlsx ($e). .jls files are still saved."
    end
end