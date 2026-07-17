# Path and headless plotting setup.
const BASEDIR = @__DIR__
ENV["GKSwstype"] = "100"

function run_experiment_FP(p::Int, Nx::Int, T::Type)

    Nt_values = [7, 15, 30, 70, 150, 300, 700, 1500, 3000, 7000]
    tFinal = 0.7

    u, x, dx, Dx_high, Dx_low = setup_problem(Nx, T)

    ref_file = joinpath(BASEDIR, "Solutions/Ref_sol_$(Nx).jls")

    if isfile(ref_file)
        println("Loading reference solution from $(ref_file)...")
        u_ref = deserialize(ref_file)
    else
        println("Reference solution not found. Computing now (this may take a while)...")

        # Reference always uses Float128 for a clean absolute-error comparison.
        u_ref_init, _, _, _, _ = setup_problem(Nx, Float128)
        Nt_ref = 700000
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
            u_fp, = IMR_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
            file_name_fp = joinpath(BASEDIR, "Errors/IMR_fp_Nx$(Nx)_$(T).jls")
        elseif p == 3
            u_fp, = SDIRK3_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
            file_name_fp = joinpath(BASEDIR, "Errors/SDIRK3_fp_Nx$(Nx)_$(T).jls")
        elseif p == 4
            u_fp, = SDIRK4_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
            file_name_fp = joinpath(BASEDIR, "Errors/SDIRK4_fp_Nx$(Nx)_$(T).jls")
        else
            u_fp, = IMR_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
            file_name_fp = joinpath(BASEDIR, "Errors/IMR_fp_Nx$(Nx)_$(T).jls")
        end

        # cast to Float128 before differencing against the reference
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

    Nt_values = [7, 15, 30, 70, 150, 300, 700, 1500, 3000, 7000]
    tFinal = 0.7

    u, x, dx, Dx_high, Dx_low = setup_problem(Nx, T, Tr)

    ref_file = joinpath(BASEDIR, "Solutions/Ref_sol_$(Nx).jls")

    if isfile(ref_file)
        println("Loading reference solution from $(ref_file)...")
        u_ref = deserialize(ref_file)
    else
        println("Reference solution not found. Computing now (this may take a while)...")

        # Reference always uses Float128 for a clean absolute-error comparison.
        u_ref_init, _, _, _, _ = setup_problem(Nx, Float128)
        Nt_ref = 700000
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
            u_mp, = IMR_mp(u, Dx_high, Dx_low; Nx = Nx, Nt = Nt, dt = dt, maxcorr = maxcorr, corr_id = corr_id)
            if maxcorr == 0
                file_name_mp = joinpath(BASEDIR, "Errors/IMR_mp_Nx$(Nx)_$(T)_$(Tr).jls")
            else
                file_name_mp = joinpath(BASEDIR, "Errors/IMR_mp_Nx$(Nx)_$(T)_$(Tr)_$(maxcorr)_$(corr_id).jls")
            end
        elseif p == 3
            u_mp, = SDIRK3_mp(u, Dx_high, Dx_low; Nx = Nx, Nt = Nt, dt = dt, maxcorr = maxcorr, corr_id = corr_id)
            if maxcorr == 0
                file_name_mp = joinpath(BASEDIR, "Errors/SDIRK3_mp_Nx$(Nx)_$(T)_$(Tr).jls")
            else
                file_name_mp = joinpath(BASEDIR, "Errors/SDIRK3_mp_Nx$(Nx)_$(T)_$(Tr)_$(maxcorr)_$(corr_id).jls")
            end
        elseif p == 4
            u_mp, = SDIRK4_mp(u, Dx_high, Dx_low; Nx = Nx, Nt = Nt, dt = dt, maxcorr = maxcorr, corr_id = corr_id)
            if maxcorr == 0
                file_name_mp = joinpath(BASEDIR, "Errors/SDIRK4_mp_Nx$(Nx)_$(T)_$(Tr).jls")
            else
                file_name_mp = joinpath(BASEDIR, "Errors/SDIRK4_mp_Nx$(Nx)_$(T)_$(Tr)_$(maxcorr)_$(corr_id).jls")
            end
        else
            u_mp, = IMR_mp(u, Dx_high, Dx_low; Nx = Nx, Nt = Nt, dt = dt, maxcorr = maxcorr, corr_id = corr_id)
            if maxcorr == 0
                file_name_mp = joinpath(BASEDIR, "Errors/IMR_mp_Nx$(Nx)_$(T)_$(Tr).jls")
            else
                file_name_mp = joinpath(BASEDIR, "Errors/IMR_mp_Nx$(Nx)_$(T)_$(Tr)_$(maxcorr)_$(corr_id).jls")
            end
        end

        # cast to Float128 before differencing against the reference
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

function plot_all_methods(Nx::Int, maxcorr::Int = 0, corr_id::String = "JAC")
    tFinal = 0.7
    Nt_values = [7, 15, 30, 70, 150, 300, 700, 1500, 3000, 7000]

    method_to_p = Dict("IMR" => 2, "SDIRK3" => 3, "SDIRK4" => 4)

    curves = []

    for (method, col) in [("IMR", :blue), ("SDIRK3", :red), ("SDIRK4", :green)]
        # Float128 baseline + mixed variants
        push!(curves, (method, Float128, nothing, "$method-Float128/Float128", col, :square))
        push!(curves, (method, Float128, Float64, "$method-Float128/Float64", col, :circle))
        push!(curves, (method, Float128, Float32, "$method-Float128/Float32", col, :utriangle))
        # Float64 baseline + mixed variants
        push!(curves, (method, Float64,  nothing, "$method-Float64/Float64", col, :diamond))
        push!(curves, (method, Float64,  Float32, "$method-Float64/Float32", col, :hexagon))
    end

    function get_data(method, Nx, HighT, LowT)
        p_val = method_to_p[method]

        if isnothing(LowT) || LowT == HighT
            # pure precision
            fname = joinpath(BASEDIR, "Errors/$(method)_fp_Nx$(Nx)_$(HighT).jls")
            is_mixed = false
            if !isfile(fname)
                println("Data not found: $fname. Generating FP experiment...")
                run_experiment_FP(p_val, Nx, HighT)
            end
        else
            # mixed precision
            if maxcorr == 0
                fname = joinpath(BASEDIR, "Errors/$(method)_mp_Nx$(Nx)_$(HighT)_$(LowT).jls")
            else
                fname = joinpath(BASEDIR, "Errors/$(method)_mp_Nx$(Nx)_$(HighT)_$(LowT)_$(maxcorr)_$(corr_id).jls")
            end
            is_mixed = true
            if !isfile(fname)
                println("Data not found: $fname. Generating MP experiment...")
                run_experiment_MP(p_val, Nx; maxcorr=maxcorr, corr_id=corr_id, T=HighT, Tr=LowT)
            end
        end

        if isfile(fname)
            errs = deserialize(fname)
            return errs, is_mixed
        else
            @warn "Failed to generate or find file: $fname"
            return fill(NaN, length(Nt_values)), is_mixed
        end
    end

    p_main = plot(
        xscale = :log10,
        yscale = :log10,
        xlabel = L"\Delta t",
        ylabel = L"\mathrm{Error}",
        guidefontsize = 20,
        tickfontsize = 12,
        legend = false,
        grid = false,
        minorgrid = false,
        size = (800, 600),
        yticks = 10.0 .^ (-34:2:10),
        xticks = 10.0 .^ (-5:1:0),
        left_margin = 5Plots.mm
    )

    p_legend = plot(
        xlims = (0, 1),
        ylims = (0, 1),
        showaxis = false,
        grid = false,
        framestyle = :none,
        legend = (0.22, 0.88),
        legendfontsize = 12,
        margin = 0Plots.mm,
        size = (450, 450),
        dpi = 500
    )

    for (method, HighT, LowT, lbl, col, shp) in curves
        errors, is_mixed = get_data(method, Nx, HighT, LowT)
        dts = [Float64(Float128(tFinal) / Float128(nt)) for nt in Nt_values]

        if is_mixed
            ln_style = :dot
            mk_color = :white
        else
            ln_style = :dash
            mk_color = col
        end

        plot!(p_main, dts, errors,
            label = "",
            color = col,
            linestyle = ln_style,
            linewidth = 2,
            marker = shp,
            markercolor = mk_color,
            markerstrokecolor = col,
            markersize = 5
        )

        plot!(p_legend, [NaN], [NaN],
            label = lbl,
            color = col,
            linestyle = ln_style,
            linewidth = 2,
            marker = shp,
            markercolor = mk_color,
            markerstrokecolor = col,
            markersize = 5
        )
    end

    if !isdir(joinpath(BASEDIR, "Results"))
        mkpath(joinpath(BASEDIR, "Results"))
    end

    if maxcorr == 0
        savefig(p_main, joinpath(BASEDIR, "Results/Figure_AllMethods_Nx$(Nx).png"))
        savefig(p_legend, joinpath(BASEDIR, "Results/Figure_AllMethods_Legend.png"))
    else
        savefig(p_main, joinpath(BASEDIR, "Results/Figure_AllMethods_Nx$(Nx)_$(maxcorr)_$(corr_id).png"))
        savefig(p_legend, joinpath(BASEDIR, "Results/Figure_AllMethods_Legend_$(maxcorr)_$(corr_id).png"))
    end

    display(p_main)
    readline()
    display(p_legend)
    readline()
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

    Nt_values = [7, 15, 30, 70, 150, 300, 700, 1500, 3000, 7000]
    tFinal = 0.7
    NUM_RUNS = 101  # 1 compilation run + 100 measured runs
    Nt_ref = 700000

    if !isdir(joinpath(BASEDIR, "Solutions")); mkpath(joinpath(BASEDIR, "Solutions")); end
    if !isdir(joinpath(BASEDIR, "Times"));     mkpath(joinpath(BASEDIR, "Times"));     end
    if !isdir(joinpath(BASEDIR, "Results"));   mkpath(joinpath(BASEDIR, "Results"));   end

    println("--- Starting plot_time benchmark ---")
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

    # filename helpers
    excel_filename = maxcorr == 0 ?
        joinpath(BASEDIR, "Times/PlotTime_$(method)_Nx$(Nx).xlsx") :
        joinpath(BASEDIR, "Times/PlotTime_$(method)_Nx$(Nx)_$(maxcorr)_$(corr_id).xlsx")

    jls_filename(lbl, HighT, LowT) = begin
        short = replace(lbl, "/" => "_")
        # pure precision and no-correction runs share the same base file
        if HighT == LowT || maxcorr == 0
            joinpath(BASEDIR, "Times/PlotTime_$(method)_Nx$(Nx)_$(short).jls")
        else
            joinpath(BASEDIR, "Times/PlotTime_$(method)_Nx$(Nx)_$(maxcorr)_$(corr_id)_$(short).jls")
        end
    end

    all_results = Dict{String, NamedTuple}()

    # Stage 1: load per-combo .jls where present
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

    # Stage 2: backward-compat fallback via .xlsx (promote to .jls on hit)
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
                # promote to .jls
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

    # Stage 3: benchmark any combos still missing
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

            local u, Dx_high, Dx_low
            try
                u, _, _, Dx_high, Dx_low = setup_problem(Nx, HighT, LowT)
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
                                    u_sol, = IMR_mp(u, Dx_high, Dx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                                elseif p == 3
                                    u_sol, = SDIRK3_mp(u, Dx_high, Dx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                                elseif p == 4
                                    u_sol, = SDIRK4_mp(u, Dx_high, Dx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                                else
                                    u_sol, = IMR_mp(u, Dx_high, Dx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                                end
                            else
                                if p == 2
                                    u_sol, = IMR_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
                                elseif p == 3
                                    u_sol, = SDIRK3_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
                                elseif p == 4
                                    u_sol, = SDIRK4_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
                                else
                                    u_sol, = IMR_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
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
            # save right away so a later crash doesn't lose this combo
            serialize(jls_filename(lbl, HighT, LowT), (times = times_vec, errors = errors_vec))
            println("  Saved $lbl to $(jls_filename(lbl, HighT, LowT))")
        end

        # rewrite Excel summary
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
                              u, Dx_high, Dx_low,  # Dx_low = nothing → fp baseline
                              Nx::Int, Nt::Int, dt,
                              maxcorr::Int, corr_id::String,
                              NUM_RUNS::Int, u_ref_final,
                              jls_path::String, combo_label::String)

    # try cache first
    if isfile(jls_path)
        try
            data = deserialize(jls_path)
            println("    Loaded $combo_label from $jls_path")
            return data.time, data.error
        catch e
            @warn "Failed to read $jls_path; will rerun. ($e)"
        end
    end

    # otherwise, run the benchmark
    is_mixed = (Dx_low !== nothing)
    times = zeros(NUM_RUNS)
    u_final = nothing

    for r in 1:NUM_RUNS
        times[r] = @elapsed begin
            local u_sol
            if is_mixed
                if p == 2
                    u_sol, = IMR_mp(u, Dx_high, Dx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                elseif p == 3
                    u_sol, = SDIRK3_mp(u, Dx_high, Dx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                elseif p == 4
                    u_sol, = SDIRK4_mp(u, Dx_high, Dx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                else
                    u_sol, = IMR_mp(u, Dx_high, Dx_low; Nx=Nx, Nt=Nt, dt=dt, maxcorr=maxcorr, corr_id=corr_id)
                end
            else
                if p == 2
                    u_sol, = IMR_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
                elseif p == 3
                    u_sol, = SDIRK3_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
                elseif p == 4
                    u_sol, = SDIRK4_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
                else
                    u_sol, = IMR_fp(u, Dx_high; Nx=Nx, Nt=Nt, dt=dt)
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

    Nt_values = [70, 700, 7000]
    tFinal = 0.7
    NUM_RUNS = 101
    Nt_ref = 700000

    if !isdir(joinpath(BASEDIR, "Solutions")); mkpath(joinpath(BASEDIR, "Solutions")); end
    if !isdir(joinpath(BASEDIR, "Times"));     mkpath(joinpath(BASEDIR, "Times"));     end

    println("--- run_time_mp_quad ---")
    println("Method: $(method), Nx: $(Nx), MaxCorr: $(maxcorr), CorrID: $(corr_id)")
    println("Combos: 128/128 (baseline, fp), 128/32 (mp), 128/16 (_mp)")

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

    u, _, dx_val, Dx_high, Dx_low_32 = setup_problem(Nx, Float128, Float32)
    _, _, _,      _,       Dx_low_16 = setup_problem(Nx, Float128, Float16)

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

        # ---- 128/128 (pure, fp) ----
        t_128128, e_128128 = _bench_combo_cached(
            p, u, Dx_high, nothing,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_128_128, "128/128")

        # ---- 128/32 (mp) ----
        t_12832, e_12832 = _bench_combo_cached(
            p, u, Dx_high, Dx_low_32,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_128_32, "128/32")

        # ---- 128/16 (mp) ----
        t_12816, e_12816 = _bench_combo_cached(
            p, u, Dx_high, Dx_low_16,
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

    # write .xlsx
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

    Nt_values = [70, 700, 7000]
    tFinal = 0.7
    NUM_RUNS = 101
    Nt_ref = 700000

    if !isdir(joinpath(BASEDIR, "Solutions")); mkpath(joinpath(BASEDIR, "Solutions")); end
    if !isdir(joinpath(BASEDIR, "Times"));     mkpath(joinpath(BASEDIR, "Times"));     end

    println("--- run_time_mp_double ---")
    println("Method: $(method), Nx: $(Nx), MaxCorr: $(maxcorr), CorrID: $(corr_id)")
    println("Combos: 64/64 (baseline, fp), 64/32 (mp), 64/16 (_mp)")

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

    u, _, dx_val, Dx_high, Dx_low_32 = setup_problem(Nx, Float64, Float32)
    _, _, _,      _,       Dx_low_16 = setup_problem(Nx, Float64, Float16)

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
        push!(dt_col, Float64(dt))
        push!(dx_col, Float64(dx_val))

        # Build per-combo cache paths for this Nt
        jls_64_64 = joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_64_64.jls")
        jls_64_32 = maxcorr == 0 ?
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_64_32.jls") :
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_$(maxcorr)_$(corr_id)_64_32.jls")
        jls_64_16 = maxcorr == 0 ?
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_64_16.jls") :
            joinpath(BASEDIR, "Times/RunTime_$(method)_Nx$(Nx)_Nt$(Nt)_$(maxcorr)_$(corr_id)_64_16.jls")

        # ---- 64/64 (pure, fp) ----
        t_6464, e_6464 = _bench_combo_cached(
            p, u, Dx_high, nothing,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_64_64, "64/64")

        # ---- 64/32 (mp) ----
        t_6432, e_6432 = _bench_combo_cached(
            p, u, Dx_high, Dx_low_32,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_64_32, "64/32")

        # ---- 64/16 (mp) ----
        t_6416, e_6416 = _bench_combo_cached(
            p, u, Dx_high, Dx_low_16,
            Nx, Nt, dt, maxcorr, corr_id,
            NUM_RUNS, u_ref_final,
            jls_64_16, "64/16")

        sp_32 = t_6464 / t_6432
        sp_16 = t_6464 / t_6416

        push!(time_64_64_col,  t_6464);  push!(error_64_64_col, e_6464)
        push!(time_64_32_col,  t_6432);  push!(error_64_32_col,  e_6432)
        push!(time_64_16_col,  t_6416);  push!(error_64_16_col,  e_6416)
        push!(sp_64_32_col,    sp_32)
        push!(sp_64_16_col,    sp_16)
    end

    # write .xlsx
    results_df = DataFrame(
        Nx             = Nx_col,
        Nt             = Nt_col,
        dx             = dx_col,
        dt             = dt_col,
        Time_64_64     = time_64_64_col,
        Error_64_64    = error_64_64_col,
        Time_64_32     = time_64_32_col,
        Error_64_32    = error_64_32_col,
        Speedup_64_32  = sp_64_32_col,
        Time_64_16     = time_64_16_col,
        Error_64_16    = error_64_16_col,
        Speedup_64_16  = sp_64_16_col,
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