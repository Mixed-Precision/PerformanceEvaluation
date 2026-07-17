using Statistics
using LinearAlgebra
using FFTW
using Dates
using TimerOutputs
using DelimitedFiles 
using Printf        
using XLSX          
using DataFrames
using Quadmath
using Plots
using LaTeXStrings
using Serialization

# Custom vectorized LU with partial pivoting
function lu_decompose!(A::AbstractMatrix{T}, piv::AbstractVector{<:Integer}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n "lu_decompose!: A must be square"
    @assert length(piv) == n "lu_decompose!: piv must have length n"

    piv .= 1:n

    for k in 1:n
        # Find pivot
        col_k = @view A[k:n, k]
        idx = argmax(abs.(col_k))
        prow = k + idx - 1

        pmax_val = A[prow, k]
        if pmax_val == zero(T)
            return A, piv, true  # singular
        end

        # Swap rows
        if prow != k
            row_k = A[k, :]
            A[k, :] .= A[prow, :]
            A[prow, :] .= row_k

            tmp_p = piv[k]
            piv[k] = piv[prow]
            piv[prow] = tmp_p
        end

        # Rank-1 Schur complement update
        if k < n
            akk_inv = one(T) / A[k, k]
            @views A[k+1:n, k] .*= akk_inv
            @views A[k+1:n, k+1:n] .-= A[k+1:n, k] .* A[k:k, k+1:n]
        end
    end

    return A, piv, false
end

# Forward/backward substitution via column AXPY
function lu_solve!(A::AbstractMatrix{T}, piv::AbstractVector{<:Integer},
                   b::AbstractVector{T}) where {T}
    n = size(A, 1)

    # Apply row permutation
    bp = b[piv] 

    # Forward solve (L * y = bp)
    for j in 1:n-1
        bp_j = bp[j]
        @views bp[j+1:n] .-= A[j+1:n, j] .* bp_j
    end

    # Back solve (U * x = y)
    for j in n:-1:1
        bp_j = bp[j] / A[j, j]
        bp[j] = bp_j
        if j > 1
            @views bp[1:j-1] .-= A[1:j-1, j] .* bp_j
        end
    end

    b .= bp
    return b
end

function lu_solve(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    Ac  = copy(A)
    piv = Vector{Int}(undef, size(A, 1))
    _, _, singular = lu_decompose!(Ac, piv)
    x = copy(b)
    if singular
        fill!(x, T(NaN))
        return x
    end
    lu_solve!(Ac, piv, x)
    return x
end

# Spectral derivative operators 
function build_fourier_matrix(N::Int; T = Float64)
    if iseven(N)
        k = vcat(0:N÷2, -N÷2+1:-1)
    else
        k = vcat(0:N÷2, -N÷2:-1)
    end
    k =  T.(k)
    ii = Complex{T}(0, 1)
    D_operator = Complex{T}.(ii .* k)
    ω = Complex{T}(exp(-T(2)*T(pi) * Complex{T}(im) / T(N)))
    F = [ω^(T(m-1)*T(n-1)) for m in 1:N, n in 1:N] ./ sqrt(T(N))
    F_inv = F'
    D = real.(F_inv  * Diagonal(D_operator) * F)
    return Matrix{T}(D)
end

function build_fourier_matrix_Dxx(N::Int; T = Float64)
    if iseven(N)
        k = vcat(0:N÷2, -N÷2+1:-1)
    else
        k = vcat(0:N÷2, -N÷2:-1)
    end
    k =  T.(k)
    D2_operator = Complex{T}.(-(k.^2))
    ω = Complex{T}(exp(-T(2)*T(pi) * Complex{T}(im) / T(N)))
    F = [ω^((m-1)*(n-1)) for m in 1:N, n in 1:N] ./ sqrt(T(N))
    F_inv = F'
    D2 = real.(F_inv * Diagonal(D2_operator) * F)
    return Matrix{T}(D2)
end

# Initialize domain and operators
function setup_problem(Nx::Int, T = Float64, Q = Float32)
    x_linrange = LinRange(-T(pi), T(pi), Nx + 1)[1:end-1]
    x  = collect(x_linrange)
    dx = x[2] - x[1]
    u  = Vector{T}(undef, Nx)
    u .= T(0.5) .* cos.(x) .+ T(0.5)
    Dxx_high = build_fourier_matrix_Dxx(Nx; T = T)
    Dxx_low  = build_fourier_matrix_Dxx(Nx; T = Q)
    return u, x, dx, Dxx_high, Dxx_low
end

# Low-precision Newton solver
function newton_solve_low(f::Function, x0::AbstractVector{T}, Dxx_low::AbstractMatrix{Tr};
                          Nx::Int, dt_low::Tr, alpha_low::Tr,
                          maxiter::Int = 20) where {T, Tr}
    tol   = Tr(10 * eps(Tr))
    x     = copy(x0)
    x_low = Tr.(x)
    Id    = Matrix{Tr}(I, Nx, Nx)
    J     = similar(Id)
    fx    = similar(x_low)
    b     = similar(x)
    piv   = Vector{Int}(undef, Nx)

    for iter in 1:maxiter
        x_low .= Tr.(x)
        fx    .= f(x_low)
        res    = norm(fx, Inf)
        if res < tol
            return x, res
        end
        J .= Id .- Tr(3.0)*alpha_low*dt_low .* (Dxx_low .* ((x_low.^2)'))

        if !all(isfinite, fx) || !all(isfinite, x_low) || !all(isfinite, J)
            return x, res
        end

        _, _, singular = lu_decompose!(J, piv)
        if singular
            return x, res
        end
        rhs = copy(fx)
        lu_solve!(J, piv, rhs)

        b .= T.(rhs)
        x .= x .- b
    end
    return x, norm(fx, Inf)
end

# Full-precision Newton solver
function newton_solve_high(f::Function, x0::AbstractVector{T}, Dxx_high::AbstractMatrix{T};
                           Nx::Int, dt::T, alpha::T,
                           maxiter::Int = 20) where {T}
    tol = T(10 * eps(T))
    x   = copy(x0)
    Id  = Matrix{T}(I, Nx, Nx)
    J   = similar(Id)
    fx  = similar(x)
    b   = similar(x)
    piv = Vector{Int}(undef, Nx)

    for iter in 1:maxiter
        fx .= f(x)
        res = norm(fx, Inf)
        if res < tol
            return x, res
        end
        J .= Id .- T(3.0) * alpha*dt .* (Dxx_high .* ((x.^2)'))

        if !all(isfinite, fx) || !all(isfinite, x) || !all(isfinite, J)
            println("Warning: NaN or Inf detected in full-precision Newton run.")
            return x, res
        end

        _, _, singular = lu_decompose!(J, piv)
        if singular
            return x, res
        end
        rhs = copy(fx)
        lu_solve!(J, piv, rhs)

        b .= rhs
        x .= x .- b
    end
    return x, norm(fx, Inf)
end

# High-precision explicit correction steps
function correction(yexp::Vector{T}, y0::Vector{T}, Dxx_high::Matrix{T},
                    Phi::Matrix{T}; Nx::Int, dt::T, alpha::T,
                    maxcorr::Int = 1,
                    corr_id::String = "JAC") where {T}
    f0 = similar(y0)
    y  = copy(y0)
    for i in 1:maxcorr
        f0 .= Dxx_high*(y0.^3)
        if corr_id == "EXP"
            # Shortcut for EXP: Phi = I avoids matvec
            y .= yexp .+ alpha*dt .* f0
        else
            y .= y0 .+ Phi*(yexp .+ alpha*dt .* f0 .- y0)
        end
        y0 .= y
    end
    return y
end

# Reference RK4 solver
function rk4(uinit::Vector{T}; Nt::Int, dt::T) where {T}
    u = copy(uinit)
    N = length(u)
    D = build_fourier_matrix_Dxx(N; T = T)
    for i in 1:Nt
        k1 = D*(u.^3)
        k2 = D*(u .+ T(0.5) * dt .* k1).^3
        k3 = D*(u .+ T(0.5) * dt .* k2).^3
        k4 = D*(u .+ dt .* k3).^3
        u .= u .+ (dt / T(6)) .* (k1 .+ T(2).*k2 .+ T(2).*k3 .+ k4)
    end
    return u
end

# Build correction matrix based on selected scheme
function _build_phi(corr_id::String, Dxx_high::Matrix{T}, u::Vector{T},
                    Nx::Int, alpha::T, dt::T) where {T}
    if corr_id == "EXP"
        return Matrix{T}(undef, 0, 0) 
    elseif corr_id == "EIN"
        Id = Matrix{T}(I, Nx, Nx)
        return inv(Id .- alpha*dt .* Dxx_high)
    else
        # JAC fallback
        Id = Matrix{T}(I, Nx, Nx)
        J0 = T(3.0) .* (Dxx_high .* ((u.^2)'))
        return inv(Id .- alpha*dt .* J0)
    end
end

# IMR: Mixed precision
function IMR_mp(uinit::Vector{T}, Dxx_high::Matrix{T}, Dxx_low::Matrix{Tr};
                Nx::Int, Nt::Int, dt::T, maxcorr::Int = 0,
                corr_id::String = "JAC") where {T, Tr}
    u    = copy(uinit)
    ulow = Tr.(u)
    ut   = Matrix{T}(undef, Nt+1, Nx)
    yt   = Matrix{T}(undef, Nt,   Nx)
    ut[1, :] .= u

    dt_low    = Tr(dt)
    alpha_low = Tr(0.5)
    alpha     = T(0.5)

    Phi = maxcorr > 0 ? _build_phi(corr_id, Dxx_high, u, Nx, alpha, dt) :
                        Matrix{T}(undef, 0, 0)

    for i in 1:Nt
        ulow .= Tr.(u)
        f = y -> y .- ulow .- alpha_low*dt_low .* (Dxx_low*(y.^3))
        y, res = newton_solve_low(f, u, Dxx_low; Nx, dt_low, alpha_low)

        if maxcorr > 0
            y = correction(u, y, Dxx_high, Phi; Nx, dt, alpha, maxcorr, corr_id)
        end

        u .= u .+ dt .* (Dxx_high*(y.^3))
        ut[i+1, :] .= u
        yt[i,   :] .= y
    end
    return ut, yt
end

# IMR: Full precision
function IMR_fp(uinit::Vector{T}, Dxx_high::Matrix{T};
                Nx::Int, Nt::Int, dt::T) where {T}
    u  = copy(uinit)
    ut = Matrix{T}(undef, Nt+1, Nx)
    yt = Matrix{T}(undef, Nt,   Nx)
    ut[1, :] .= u
    alpha = T(0.5)

    for i in 1:Nt
        f = y -> y .- u .- alpha*dt .* (Dxx_high*(y.^3))
        y, res = newton_solve_high(f, u, Dxx_high; Nx, dt, alpha)

        u .= u .+ dt .* (Dxx_high*(y.^3))
        ut[i+1, :] .= u
        yt[i,   :] .= y
    end
    return ut, yt
end

# SDIRK3: Mixed precision
function SDIRK3_mp(uinit::Vector{T}, Dxx_high::Matrix{T}, Dxx_low::Matrix{Tr};
                   Nx::Int, Nt::Int, dt::T, maxcorr::Int = 0,
                   corr_id::String = "JAC") where {T, Tr}
    u   = copy(uinit)
    ut  = Matrix{T}(undef, Nt+1, Nx)
    yt1 = Matrix{T}(undef, Nt,   Nx)
    yt2 = Matrix{T}(undef, Nt,   Nx)
    ulow = Tr.(u)
    ut[1, :] .= u
    dt_low    = Tr(dt)
    y1 = similar(u); y2 = similar(u)
    F1 = similar(u); F2 = similar(u)
    alpha     = (sqrt(T(3)) + T(3))/T(6)
    alpha_low = Tr(alpha)

    Phi = maxcorr > 0 ? _build_phi(corr_id, Dxx_high, u, Nx, alpha, dt) :
                        Matrix{T}(undef, 0, 0)

    for i in 1:Nt
        ulow .= Tr.(u)

        # Implicit stage 1
        f = y -> y .- ulow .- alpha_low*dt_low .* (Dxx_low*(y.^3))
        y1, res = newton_solve_low(f, u, Dxx_low; Nx, dt_low, alpha_low)
        if maxcorr > 0
            y1 = correction(u, y1, Dxx_high, Phi; Nx, dt, alpha, maxcorr, corr_id)
        end
        F1 .= Dxx_high*(y1.^3)

        # Implicit stage 2
        f = y -> y .- ulow .- (Tr(1) - Tr(2)*alpha_low)*dt_low .* Tr.(F1) .-
                  alpha_low*dt_low .* (Dxx_low*(y.^3))
        y2, res = newton_solve_low(f, u, Dxx_low; Nx, dt_low, alpha_low)
        if maxcorr > 0
            yexp = u .+ (T(1) - T(2)*alpha)*dt .* F1
            y2 = correction(yexp, y2, Dxx_high, Phi; Nx, dt, alpha, maxcorr, corr_id)
        end
        F2 .= Dxx_high*(y2.^3)

        # Explicit final stage
        u .= u .+ dt*(T(0.5)) .* F1 .+ dt*(T(0.5)) .* F2
        ut[i+1, :] .= u
        yt1[i, :]  .= y1
        yt2[i, :]  .= y2
    end
    return ut, yt1, yt2
end

# SDIRK3: Full precision
function SDIRK3_fp(uinit::Vector{T}, Dxx_high::Matrix{T};
                   Nx::Int, Nt::Int, dt::T) where {T}
    u   = copy(uinit)
    ut  = Matrix{T}(undef, Nt+1, Nx)
    yt1 = Matrix{T}(undef, Nt,   Nx)
    yt2 = Matrix{T}(undef, Nt,   Nx)
    ut[1, :] .= u
    y1 = similar(u); y2 = similar(u)
    F1 = similar(u); F2 = similar(u)
    alpha = (sqrt(T(3)) + T(3))/T(6)

    for i in 1:Nt
        f = y -> y .- u .- alpha*dt .* (Dxx_high*(y.^3))
        y1, res = newton_solve_high(f, u, Dxx_high; Nx, dt, alpha)
        F1 .= Dxx_high*(y1.^3)

        f = y -> y .- u .- (T(1) - T(2)*alpha)*dt .* F1 .-
                  alpha*dt .* (Dxx_high*(y.^3))
        y2, res = newton_solve_high(f, u, Dxx_high; Nx, dt, alpha)
        F2 .= Dxx_high*(y2.^3)

        u .= u .+ dt*(T(0.5)) .* F1 .+ dt*(T(0.5)) .* F2
        ut[i+1, :] .= u
        yt1[i, :]  .= y1
        yt2[i, :]  .= y2
    end
    return ut, yt1, yt2
end

# SDIRK4: Mixed precision
function SDIRK4_mp(uinit::Vector{T}, Dxx_high::Matrix{T}, Dxx_low::Matrix{Tr};
                   Nx::Int, Nt::Int, dt::T, maxcorr::Int = 0,
                   corr_id::String = "JAC") where {T, Tr}
    u   = copy(uinit)
    ut  = Matrix{T}(undef, Nt+1, Nx)
    yt1 = Matrix{T}(undef, Nt,   Nx)
    yt2 = Matrix{T}(undef, Nt,   Nx)
    yt3 = Matrix{T}(undef, Nt,   Nx)
    ut[1, :] .= u
    ulow = Tr.(u)
    dt_low = Tr(dt)
    y1 = similar(u); y2 = similar(u); y3 = similar(u)
    F1 = similar(u); F2 = similar(u); F3 = similar(u)
    gamma     = T(2) * cos(T(pi) / T(18)) / sqrt(T(3))
    alpha     = T(0.5)*(T(1) + gamma)
    alpha_low = Tr(alpha)
    gamma_low = Tr(gamma)

    Phi = maxcorr > 0 ? _build_phi(corr_id, Dxx_high, u, Nx, alpha, dt) :
                        Matrix{T}(undef, 0, 0)

    for i in 1:Nt
        ulow .= Tr.(u)

        # Implicit stage 1
        f = y -> y .- ulow .- alpha_low*dt_low .* (Dxx_low*(y.^3))
        y1, res = newton_solve_low(f, u, Dxx_low; Nx, dt_low, alpha_low)
        if maxcorr > 0
            y1 = correction(u, y1, Dxx_high, Phi; Nx, dt, alpha, maxcorr, corr_id)
        end
        F1 .= Dxx_high*(y1.^3)

        # Implicit stage 2
        f = y -> y .- ulow .+ gamma_low*Tr(0.5)*dt_low .* Tr.(F1) .-
                  alpha_low*dt_low .* (Dxx_low*(y.^3))
        y2, res = newton_solve_low(f, u, Dxx_low; Nx, dt_low, alpha_low)
        if maxcorr > 0
            yexp = u .- gamma*T(0.5)*dt .* F1
            y2 = correction(yexp, y2, Dxx_high, Phi; Nx, dt, alpha, maxcorr, corr_id)
        end
        F2 .= Dxx_high*(y2.^3)

        # Implicit stage 3
        f = y -> y .- ulow .- (Tr(1) + gamma_low)*dt_low .* Tr.(F1) .+
                  (Tr(1) + Tr(2)*gamma_low)*dt_low .* Tr.(F2) .-
                  alpha_low*dt_low .* (Dxx_low*(y.^3))
        y3, res = newton_solve_low(f, u, Dxx_low; Nx, dt_low, alpha_low)
        if maxcorr > 0
            yexp = u .+ (T(1) + gamma)*dt .* F1 .- (T(1) + T(2)*gamma)*dt .* F2
            y3 = correction(yexp, y3, Dxx_high, Phi; Nx, dt, alpha, maxcorr, corr_id)
        end
        F3 .= Dxx_high*(y3.^3)

        # Explicit final stage
        u .= u .+ dt .* ((T(1)/T(6))*(T(1)/(gamma^2)) .* F1 .+
                          (T(1) - T(1)/(T(3)*gamma^2)) .* F2 .+
                          (T(1)/T(6))*(T(1)/(gamma^2)) .* F3)
        ut[i+1, :] .= u
        yt1[i, :]  .= y1
        yt2[i, :]  .= y2
        yt3[i, :]  .= y3
    end
    return ut, yt1, yt2, yt3
end

# SDIRK4: Full precision
function SDIRK4_fp(uinit::Vector{T}, Dxx_high::Matrix{T};
                   Nx::Int, Nt::Int, dt::T) where {T}
    u   = copy(uinit)
    ut  = Matrix{T}(undef, Nt+1, Nx)
    yt1 = Matrix{T}(undef, Nt,   Nx)
    yt2 = Matrix{T}(undef, Nt,   Nx)
    yt3 = Matrix{T}(undef, Nt,   Nx)
    ut[1, :] .= u
    y1 = similar(u); y2 = similar(u); y3 = similar(u)
    F1 = similar(u); F2 = similar(u); F3 = similar(u)
    gamma = T(2) * cos(T(pi) / T(18)) / sqrt(T(3))
    alpha = T(0.5)*(T(1) + gamma)

    for i in 1:Nt
        f = y -> y .- u .- alpha*dt .* (Dxx_high*(y.^3))
        y1, res = newton_solve_high(f, u, Dxx_high; Nx, dt, alpha)
        F1 .= Dxx_high*(y1.^3)

        f = y -> y .- u .+ gamma*T(0.5)*dt .* F1 .-
                  alpha*dt .* (Dxx_high*(y.^3))
        y2, res = newton_solve_high(f, u, Dxx_high; Nx, dt, alpha)
        F2 .= Dxx_high*(y2.^3)

        f = y -> y .- u .- (T(1) + gamma)*dt .* F1 .+
                  (T(1) + T(2)*gamma)*dt .* F2 .-
                  alpha*dt .* (Dxx_high*(y.^3))
        y3, res = newton_solve_high(f, u, Dxx_high; Nx, dt, alpha)
        F3 .= Dxx_high*(y3.^3)

        u .= u .+ dt .* ((T(1)/T(6))*(T(1)/(gamma^2)) .* F1 .+
                          (T(1) - T(1)/(T(3)*gamma^2)) .* F2 .+
                          (T(1)/T(6))*(T(1)/(gamma^2)) .* F3)
        ut[i+1, :] .= u
        yt1[i, :]  .= y1
        yt2[i, :]  .= y2
        yt3[i, :]  .= y3
    end
    return ut, yt1, yt2, yt3
end