module StatlibBackend

using LinearAlgebra
using Statistics
using Random
using Base.Threads

const ERR_OK = 0
const ERR_BAD_INPUT = 1
const ERR_LINALG = 2
const ERR_EXCEPTION = 3
const CONFORMAL_MODE_FAST = 0
const CONFORMAL_MODE_REFERENCE = 1

struct PlainSCMProblem
    X0::Matrix{Float64}
    X0_raw::Matrix{Float64}
    Y0_raw::Matrix{Float64}
    x1::Vector{Float64}
    x1_raw::Vector{Float64}
    y1_raw::Vector{Float64}
    control_pre_sum::Vector{Float64}
    treated_pre_sum::Vector{Float64}
    treated_post_sum::Vector{Float64}
    gram::Matrix{Float64}
    q_base::Vector{Float64}
    control_idx::Vector{Int}
    treated_idx::Vector{Int}
    n0::Int
    n1::Int
    t0::Int
    tpost::Int
end

@inline function _fill_omit_pre_col!(outX::AbstractMatrix{Float64},
                                    outY::AbstractMatrix{Float64},
                                    X::AbstractMatrix{Float64},
                                    y::AbstractMatrix{Float64},
                                    omit_idx::Int)
    n, t0 = size(X)
    tpost = size(y, 2)
    col_out = 1
    @inbounds for col in 1:t0
        if col == omit_idx
            continue
        end
        outX[:, col_out] .= @view X[:, col]
        col_out += 1
    end
    @views outY[:, 1] .= X[:, omit_idx]
    for col in 1:tpost
        @views outY[:, col + 1] .= y[:, col]
    end
    return
end

@inline function _fill_omit_row!(outX::AbstractMatrix{Float64},
                                outY::AbstractMatrix{Float64},
                                outTrt::AbstractVector{Float64},
                                X::AbstractMatrix{Float64},
                                y::AbstractMatrix{Float64},
                                trt::AbstractVector{Float64},
                                omit_row::Int)
    n = size(X, 1)
    out_row = 1
    @inbounds for row in 1:n
        if row == omit_row
            continue
        end
        outX[out_row, :] .= @view X[row, :]
        outY[out_row, :] .= @view y[row, :]
        outTrt[out_row] = trt[row]
        out_row += 1
    end
    return
end

function fit_ols_dense!(n::Int32, p::Int32, Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble},
                        coefptr::Ptr{Cdouble}, sigma2ptr::Ptr{Cdouble},
                        dfresidptr::Ptr{Cint}, rssptr::Ptr{Cdouble},
                        errptr::Ptr{UInt8}, errlen::Int32)::Cint
    try
        if n <= 0 || p <= 0 || n < p
            _set_err!(errptr, errlen, "invalid dimensions: require n > 0, p > 0, n >= p")
            return ERR_BAD_INPUT
        end

        X = unsafe_wrap(Array, Xptr, (Int(n), Int(p)); own = false)
        y = unsafe_wrap(Array, yptr, Int(n); own = false)

        F = qr(X)
        β = F \ y
        r = y .- X * β
        df = Int(n - p)
        rss = sum(abs2, r)
        sigma2 = df > 0 ? rss / df : NaN

        unsafe_copyto!(coefptr, pointer(β), Int(p))
        unsafe_store!(sigma2ptr, sigma2)
        unsafe_store!(dfresidptr, Cint(df))
        unsafe_store!(rssptr, rss)

        return ERR_OK
    catch e
        _set_err!(errptr, errlen, sprint(showerror, e))
        return ERR_EXCEPTION
    end
end

"""
Compute ridge coefficients using LOOCV-selected lambda via one SVD pass.

Inputs are raw pointers to column-major X and y, with caller-provided lambda grid.
This implementation minimizes repeated allocations by reusing work buffers per lambda.
"""
function fit_ridge_loocv_dense!(n::Int32, p::Int32,
                                Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble},
                                nlambda::Int32, lambdasptr::Ptr{Cdouble},
                                coefptr::Ptr{Cdouble},
                                bestlambdaptr::Ptr{Cdouble}, bestmseptr::Ptr{Cdouble},
                                errptr::Ptr{UInt8}, errlen::Int32)::Cint
    try
        if n <= 1 || p <= 0
            _set_err!(errptr, errlen, "invalid dimensions: require n > 1, p > 0")
            return ERR_BAD_INPUT
        end
        if nlambda <= 0
            _set_err!(errptr, errlen, "nlambda must be positive")
            return ERR_BAD_INPUT
        end

        ni = Int(n)
        pi = Int(p)
        li = Int(nlambda)

        X = unsafe_wrap(Array, Xptr, (ni, pi); own = false)
        y = unsafe_wrap(Array, yptr, ni; own = false)
        lambdas = unsafe_wrap(Array, lambdasptr, li; own = false)

        # Economy SVD once; all candidate lambdas reuse this decomposition.
        F = svd(X)
        U = F.U
        s = F.S
        V = F.V

        rnk = length(s)
        if rnk == 0
            _set_err!(errptr, errlen, "X has zero numerical rank")
            return ERR_LINALG
        end

        s2 = similar(s)
        @inbounds @simd for j in eachindex(s)
            s2[j] = s[j] * s[j]
        end

        Uy = transpose(U) * y
        U2 = U .^ 2

        d = similar(s)
        tmp_r = similar(s)
        yhat = Vector{Float64}(undef, ni)
        hdiag = Vector{Float64}(undef, ni)

        best_idx = 1
        best_mse = Inf

        @inbounds for k in 1:li
            λ = lambdas[k]
            if !(isfinite(λ) && λ >= 0.0)
                continue
            end

            @simd for j in 1:rnk
                dj = s2[j] / (s2[j] + λ)
                d[j] = dj
                tmp_r[j] = dj * Uy[j]
            end

            mul!(yhat, U, tmp_r)

            fill!(hdiag, 0.0)
            for j in 1:rnk
                w = d[j]
                @simd for i in 1:ni
                    hdiag[i] += U2[i, j] * w
                end
            end

            sse = 0.0
            for i in 1:ni
                denom = 1.0 - hdiag[i]
                if abs(denom) < 1e-12
                    sse = Inf
                    break
                end
                e = (y[i] - yhat[i]) / denom
                sse += e * e
            end

            mse = sse / ni
            if mse < best_mse
                best_mse = mse
                best_idx = k
            end
        end

        λbest = lambdas[best_idx]
        @inbounds @simd for j in 1:rnk
            tmp_r[j] = (s[j] / (s2[j] + λbest)) * Uy[j]
        end

        β = Vector{Float64}(undef, pi)
        mul!(β, V, tmp_r)

        unsafe_copyto!(coefptr, pointer(β), pi)
        unsafe_store!(bestlambdaptr, λbest)
        unsafe_store!(bestmseptr, best_mse)

        return ERR_OK
    catch e
        _set_err!(errptr, errlen, sprint(showerror, e))
        return ERR_EXCEPTION
    end
end

function _solve_square_system(A::AbstractMatrix{Float64}, b::AbstractVector{Float64})
    try
        return A \ b
    catch
        return pinv(Matrix(A)) * b
    end
end

function _quantile(values::AbstractVector{Float64}, p::Float64)
    x = sort(values)
    n = length(x)
    if n == 0
        return NaN
    elseif n == 1
        return x[1]
    end

    r = clamp((n - 1) * p + 1, 1, n)
    lo = Int(floor(r))
    hi = Int(ceil(r))
    if lo == hi
        return x[lo]
    end
    w = r - lo
    x[lo] + w * (x[hi] - x[lo])
end

@inline function _maxabsdiff(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    out = 0.0
    @inbounds @simd for i in eachindex(a, b)
        d = abs(a[i] - b[i])
        out = ifelse(d > out, d, out)
    end
    out
end

function _create_lambda_grid(lambda_max::Float64, lambda_min_ratio::Float64, n_lambda::Int)
    if lambda_max <= 0 || !isfinite(lambda_max) || n_lambda <= 0
        return Float64[]
    end
    scale = lambda_min_ratio^(1 / n_lambda)
    return lambda_max .* (scale .^ (collect(0:n_lambda) .- 1))
end

function _ridge_adjustment_cached(Xc::AbstractMatrix{Float64},
                                 eig::Eigen,
                                 δ::AbstractVector{Float64},
                                 λ::Float64)
    if length(δ) != size(Xc, 2)
        error("delta has wrong length")
    end

    c = eig.values .+ λ
    coeff = eig.vectors * ((eig.vectors' * δ) ./ c)
    Xc * coeff
end

function _ridge_adjustment(Xc::AbstractMatrix{Float64},
                           x1::AbstractVector{Float64},
                           syn::AbstractVector{Float64},
                           λ::Float64)
    δ = x1 .- transpose(Xc) * syn
    G = transpose(Xc) * Xc
    A = Matrix(G)
    @inbounds for j in 1:size(A, 1)
        A[j, j] += λ
    end
    γ = _solve_square_system(A, δ)
    return Xc * γ
end

function _solve_simplex_qp(donors::AbstractMatrix{Float64},
                           target::AbstractVector{Float64};
                           tol::Float64 = 1e-8,
                           maxiter::Int = 20_000,
                           init::Union{AbstractVector{Float64}, Nothing} = nothing)
    n0 = size(donors, 1)
    if n0 == 1
        return ones(Float64, 1)
    end

    P = Matrix(donors * transpose(donors))
    q = -(donors * target)
    return _solve_simplex_qp_warm(P, q; tol = tol, maxiter = maxiter, init = init)
end

function _solve_simplex_qp_cached(P::AbstractMatrix{Float64},
                                  q::AbstractVector{Float64};
                                  tol::Float64 = 1e-8,
                                  maxiter::Int = 20_000,
                                  init::Union{AbstractVector{Float64}, Nothing} = nothing)
    n0 = size(P, 1)
    if n0 == 0
        error("donors matrix must have at least one control unit")
    end
    if n0 == 1
        return ones(Float64, 1)
    end

    w = if init === nothing
        start = 1
        best = 0.5 * P[1, 1] + q[1]
        @inbounds for j in 2:n0
            cand = 0.5 * P[j, j] + q[j]
            if cand < best
                best = cand
                start = j
            end
        end
        w0 = zeros(Float64, n0)
        w0[start] = 1.0
        w0
    else
        _project_simplex!(copy(init))
    end

    z = copy(w)
    t = 1.0

    # FISTA step size uses a conservative upper bound for spectral norm.
    # Computing opnorm repeatedly in inference loops can dominate runtime.
    L = 0.0
    for j in 1:n0
        colsum = 0.0
        @inbounds @simd for i in 1:n0
            colsum += abs(P[i, j])
        end
        if colsum > L
            L = colsum
        end
    end
    L = max(L, 1.0)
    step = 1.0 / L

    grad = similar(q)
    w_new = similar(w)
    trial = similar(w)
    last_obj = Inf

    for _ in 1:maxiter
        mul!(grad, P, z)
        @. grad = grad + q
        @. w_new = z - step * grad
        _project_simplex!(w_new)

        mul!(trial, P, w_new)
        obj = 0.5 * dot(w_new, trial) + dot(q, w_new)

        if !isfinite(obj)
            error("non-finite objective in simplex solve")
        end

        if obj > last_obj + 1e-12
            copyto!(z, w)
            t = 1.0
            continue
        end

        @inbounds if _maxabsdiff(w_new, w) <= tol
            return w_new
        end

        t_new = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t))
        @. trial = w_new + ((t - 1.0) / t_new) * (w_new - w)
        copyto!(z, trial)
        copyto!(w, w_new)
        t = t_new
        last_obj = obj
    end

    error("simplex QP did not converge")
end

@inline function _best_simplex_vertex(P::AbstractMatrix{Float64},
                                      q::AbstractVector{Float64})
    n0 = size(P, 1)
    best_idx = 1
    best_obj = 0.5 * P[1, 1] + q[1]
    @inbounds for j in 2:n0
        cand = 0.5 * P[j, j] + q[j]
        if cand < best_obj
            best_obj = cand
            best_idx = j
        end
    end
    best_idx
end

function _initial_simplex_support(P::AbstractMatrix{Float64},
                                  q::AbstractVector{Float64},
                                  init::Union{AbstractVector{Float64}, Nothing},
                                  active_tol::Float64)
    n0 = size(P, 1)
    if init === nothing || length(init) != n0
        return [_best_simplex_vertex(P, q)]
    end

    start = _project_simplex!(copy(init))
    support = findall(x -> x > active_tol, start)
    if isempty(support)
        return [_best_simplex_vertex(P, q)]
    end
    sort!(support)
    support
end

function _solve_simplex_qp_support(P::AbstractMatrix{Float64},
                                   q::AbstractVector{Float64},
                                   support::AbstractVector{Int})
    s = length(support)
    if s == 0
        error("support must not be empty")
    end
    if s == 1
        idx = support[1]
        return ones(Float64, 1), -(P[idx, idx] + q[idx])
    end

    K = Matrix{Float64}(undef, s + 1, s + 1)
    rhs = Vector{Float64}(undef, s + 1)

    @inbounds for col in 1:s
        j = support[col]
        rhs[col] = -q[j]
        for row in 1:s
            K[row, col] = P[support[row], j]
        end
        K[col, s + 1] = 1.0
        K[s + 1, col] = 1.0
    end
    K[s + 1, s + 1] = 0.0
    rhs[s + 1] = 1.0

    sol = _solve_square_system(K, rhs)
    sol[1:s], sol[s + 1]
end

function _solve_simplex_qp_active_set(P::AbstractMatrix{Float64},
                                      q::AbstractVector{Float64};
                                      tol::Float64 = 1e-8,
                                      maxiter::Int = max(50, 10 * size(P, 1)),
                                      init::Union{AbstractVector{Float64}, Nothing} = nothing)
    n0 = size(P, 1)
    if n0 == 0
        error("donors matrix must have at least one control unit")
    end
    if n0 == 1
        return ones(Float64, 1)
    end

    active_tol = max(10.0 * tol, 1e-10)
    reduced_tol = max(100.0 * tol, 1e-8)
    support = _initial_simplex_support(P, q, init, active_tol)
    active = falses(n0)
    active[support] .= true

    weights = zeros(Float64, n0)
    grad = similar(q)

    for _ in 1:maxiter
        ws, nu = _solve_simplex_qp_support(P, q, support)
        if !isfinite(nu) || any(!isfinite, ws)
            error("non-finite active-set simplex iterate")
        end

        keep = findall(x -> x > active_tol, ws)
        if length(keep) != length(support)
            if isempty(keep)
                best_local = argmax(ws)
                support = [support[best_local]]
            else
                support = support[keep]
            end
            fill!(active, false)
            active[support] .= true
            continue
        end

        fill!(weights, 0.0)
        @inbounds for (k, idx) in pairs(support)
            weights[idx] = ws[k]
        end

        weight_sum = sum(weights)
        if !isfinite(weight_sum) || weight_sum <= 0.0
            error("invalid active-set simplex weights")
        end
        if abs(weight_sum - 1.0) > reduced_tol
            @. weights = weights / weight_sum
        end

        mul!(grad, P, weights)
        @. grad = grad + q

        add_idx = 0
        worst_violation = 0.0
        @inbounds for j in 1:n0
            if active[j]
                continue
            end
            viol = grad[j] + nu
            if viol < worst_violation
                worst_violation = viol
                add_idx = j
            end
        end

        if add_idx == 0
            return weights
        end

        push!(support, add_idx)
        sort!(support)
        fill!(active, false)
        active[support] .= true
    end

    error("active-set simplex QP did not converge")
end

function _solve_simplex_qp_warm(P::AbstractMatrix{Float64},
                                q::AbstractVector{Float64};
                                tol::Float64 = 1e-8,
                                maxiter::Int = 20_000,
                                init::Union{AbstractVector{Float64}, Nothing} = nothing)
    try
        _solve_simplex_qp_active_set(
            P, q;
            tol = tol,
            maxiter = max(50, 10 * size(P, 1)),
            init = init
        )
    catch
        _solve_simplex_qp_cached(P, q; tol = tol, maxiter = maxiter, init = init)
    end
end

function _project_simplex!(x::AbstractVector{Float64})
    n = length(x)
    if n == 0
        return x
    end

    u = sort!(copy(x), rev = true)
    ρ = 0
    cssv = 0.0
    idx = 0.0
    @inbounds for j in 1:n
        cssv += u[j]
        t = (cssv - 1.0) / j
        if u[j] > t
            ρ = j
            idx = t
        end
    end

    if ρ == 0
        fill!(x, 1.0 / n)
        return x
    end

    θ = idx
    s = 0.0
    @inbounds for j in 1:n
        v = x[j] - θ
        v = ifelse(v > 0.0, v, 0.0)
        x[j] = v
        s += v
    end

    if !isfinite(s) || s <= 0.0
        fill!(x, 1.0 / n)
        return x
    end
    @. x = x / s
    return x
end

function _project_simplex(v::AbstractVector{Float64})
    w = copy(v)
    _project_simplex!(w)
    return w
end

function _lambda_errors(Xc::AbstractMatrix{Float64},
                        x1::AbstractVector{Float64},
                        lambdas::AbstractVector{Float64},
                        holdout_length::Int,
                        scm::Bool)
    t0 = size(Xc, 2)
    nsplit = t0 - holdout_length
    if nsplit <= 0
        error("holdout_length must be smaller than the number of pre-treatment periods")
    end

    n0 = size(Xc, 1)
    errors = Matrix{Float64}(undef, nsplit, length(lambdas))

    for i in 1:nsplit
        keep = trues(t0)
        keep[i:(i + holdout_length - 1)] .= false

        Xfit = Xc[:, keep]
        xfit = x1[keep]
        Xval = Xc[:, .!keep]
        xval = x1[.!keep]

        syn = scm ? _solve_simplex_qp(Xfit, xfit) : fill(1.0 / n0, n0)
        δ = xfit .- transpose(Xfit) * syn
        eig = eigen(Symmetric(Matrix(transpose(Xfit) * Xfit)))
        for j in eachindex(lambdas)
            λ = lambdas[j]
            aug = syn .+ _ridge_adjustment_cached(Xfit, eig, δ, λ)
            resid = xval .- transpose(Xval) * aug
            errors[i, j] = sum(abs2, resid)
        end
    end

    means = vec(mean(errors; dims = 1))
    ses = similar(means)
    @inbounds for j in eachindex(means)
        ses[j] = nsplit > 1 ? std(view(errors, :, j)) / sqrt(nsplit) : 0.0
    end

    return means, ses
end

function _choose_lambda(lambdas::AbstractVector{Float64},
                        lambda_errors::AbstractVector{Float64},
                        lambda_errors_se::AbstractVector{Float64},
                        min1se::Bool)
    idx = argmin(lambda_errors)
    λmin = lambdas[idx]
    if !min1se
        return λmin
    end

    cutoff = lambda_errors[idx] + lambda_errors_se[idx]
    admissible = lambdas[lambda_errors .<= cutoff]
    return maximum(admissible)
end

function _matrix_sqrt_psd(V::AbstractMatrix{Float64})
    if size(V, 1) != size(V, 2)
        error("V must be square")
    end
    sym = Symmetric((V + transpose(V)) / 2)
    decomp = eigen(sym)
    vals = max.(decomp.values, 0.0)
    return decomp.vectors * (Diagonal(sqrt.(vals)) * transpose(decomp.vectors))
end

@inline function _can_use_plain_scm_fastpath(ridge::Bool,
                                            scm::Bool,
                                            V::Union{AbstractMatrix{Float64}, Nothing},
                                            fixedeff::Bool)
    scm && !ridge && V === nothing && !fixedeff
end

function _prepare_plain_scm_problem(X::AbstractMatrix{Float64},
                                   y::AbstractMatrix{Float64},
                                   trt::AbstractVector{Float64};
                                   trt_mask::Union{AbstractVector{Bool}, Nothing} = nothing)
    if size(y, 1) != size(X, 1)
        error("X and y must have matching row counts")
    end

    treated = trt_mask === nothing ? trt .> 0.5 : trt_mask
    control = .!treated
    control_idx = findall(control)
    treated_idx = findall(treated)
    n0 = length(control_idx)
    n1 = length(treated_idx)
    if n0 == 0
        error("No control units found")
    end
    if n1 == 0
        error("No treated units found")
    end

    X0_raw = Matrix(X[control_idx, :])
    Y0_raw = Matrix(y[control_idx, :])
    control_pre_sum = vec(sum(X0_raw; dims = 1))
    control_pre_mean = control_pre_sum ./ n0

    x1_raw = vec(mean(X[treated_idx, :]; dims = 1))
    y1_raw = vec(mean(y[treated_idx, :]; dims = 1))
    treated_pre_sum = vec(sum(X[treated_idx, :]; dims = 1))
    treated_post_sum = vec(sum(y[treated_idx, :]; dims = 1))

    X0 = X0_raw .- transpose(control_pre_mean)
    x1 = x1_raw .- control_pre_mean
    gram = Matrix(X0 * transpose(X0))
    q_base = -(X0 * x1)

    PlainSCMProblem(
        X0,
        X0_raw,
        Y0_raw,
        x1,
        x1_raw,
        y1_raw,
        control_pre_sum,
        treated_pre_sum,
        treated_post_sum,
        gram,
        q_base,
        control_idx,
        treated_idx,
        n0,
        n1,
        size(X, 2),
        size(y, 2)
    )
end

function _solve_plain_scm(problem::PlainSCMProblem;
                          gram::AbstractMatrix{Float64} = problem.gram,
                          q::AbstractVector{Float64} = problem.q_base,
                          init_weights::Union{AbstractVector{Float64}, Nothing} = nothing)
    syn = _solve_simplex_qp_warm(gram, q; init = init_weights)
    mhat = vcat(vec(transpose(syn) * problem.X0), vec(transpose(syn) * problem.Y0_raw))
    (
        weights = reshape(syn, :, 1),
        syn = reshape(syn, :, 1),
        lambda = NaN,
        mhat = mhat,
        l2_imbalance = NaN,
        scaled_l2_imbalance = NaN,
        t0 = problem.t0,
        tpost = problem.tpost,
        control_idx = problem.control_idx,
        treated_idx = problem.treated_idx
    )
end

function _plain_scm_counterfactual(problem::PlainSCMProblem,
                                   weights::AbstractVector{Float64})
    out = Vector{Float64}(undef, problem.t0 + problem.tpost)
    mul!(@view(out[1:problem.t0]), transpose(problem.X0_raw), weights)
    mul!(@view(out[(problem.t0 + 1):end]), transpose(problem.Y0_raw), weights)
    out
end

function _plain_scm_att(problem::PlainSCMProblem,
                        weights::AbstractVector{Float64},
                        counter::AbstractVector{Float64})
    out = similar(counter)
    @inbounds for j in 1:problem.t0
        out[j] = problem.x1_raw[j] - counter[j]
    end
    @inbounds for j in 1:problem.tpost
        out[problem.t0 + j] = problem.y1_raw[j] - counter[problem.t0 + j]
    end
    out
end

@inline function _fill_plain_scm_omit_pre_system!(P::AbstractMatrix{Float64},
                                                 q::AbstractVector{Float64},
                                                 problem::PlainSCMProblem,
                                                 omit_idx::Int)
    copyto!(P, problem.gram)
    d = @view(problem.X0[:, omit_idx])
    x1i = problem.x1[omit_idx]
    @inbounds for j in 1:problem.n0
        dj = d[j]
        q[j] = problem.q_base[j] + dj * x1i
        for i in 1:problem.n0
            P[i, j] -= d[i] * dj
        end
    end
    return
end

function _jackknife_plus_row_plain_scm!(X::AbstractMatrix{Float64},
                                       y::AbstractMatrix{Float64},
                                       trt::AbstractVector{Float64};
                                       conservative::Bool,
                                       alpha::Float64)
    problem = _prepare_plain_scm_problem(X, y, trt)
    t0 = problem.t0
    tpost = problem.tpost
    total = t0 + tpost

    base_fit = _solve_plain_scm(problem)
    base_weights = vec(base_fit.weights)
    base_counter = _plain_scm_counterfactual(problem, base_weights)
    base_att = _plain_scm_att(problem, base_weights, base_counter)

    jack_ests = Array{Float64}(undef, 4, tpost + 1, t0)
    held_out = Vector{Float64}(undef, t0)

    if Threads.nthreads() > 1 && t0 > 8
        nthreads = Threads.nthreads()
        gram_pool = [Matrix{Float64}(undef, problem.n0, problem.n0) for _ in 1:nthreads]
        q_pool = [Vector{Float64}(undef, problem.n0) for _ in 1:nthreads]
        post_pool = [Vector{Float64}(undef, tpost) for _ in 1:nthreads]

        Threads.@threads for i in 1:t0
            tid = threadid()
            gram_i = gram_pool[tid]
            q_i = q_pool[tid]
            post_i = post_pool[tid]
            _fill_plain_scm_omit_pre_system!(gram_i, q_i, problem, i)
            weights_i = _solve_simplex_qp_warm(gram_i, q_i; init = base_weights)
            mul!(post_i, transpose(problem.Y0_raw), weights_i)

            held = problem.x1_raw[i] - dot(weights_i, @view(problem.X0_raw[:, i]))
            held_out[i] = held

            est_mean = 0.0
            @inbounds for j in 1:tpost
                v = problem.y1_raw[j] - post_i[j]
                est_mean += v
                jack_ests[1, j, i] = v + abs(held)
                jack_ests[2, j, i] = v - abs(held)
                jack_ests[3, j, i] = v + held
                jack_ests[4, j, i] = v
            end
            est_mean /= tpost
            jack_ests[1, tpost + 1, i] = est_mean + abs(held)
            jack_ests[2, tpost + 1, i] = est_mean - abs(held)
            jack_ests[3, tpost + 1, i] = est_mean + held
            jack_ests[4, tpost + 1, i] = est_mean
        end
    else
        gram_i = Matrix{Float64}(undef, problem.n0, problem.n0)
        q_i = Vector{Float64}(undef, problem.n0)
        post_i = Vector{Float64}(undef, tpost)
        for i in 1:t0
            _fill_plain_scm_omit_pre_system!(gram_i, q_i, problem, i)
            weights_i = _solve_simplex_qp_warm(gram_i, q_i; init = base_weights)
            mul!(post_i, transpose(problem.Y0_raw), weights_i)

            held = problem.x1_raw[i] - dot(weights_i, @view(problem.X0_raw[:, i]))
            held_out[i] = held

            est_mean = 0.0
            @inbounds for j in 1:tpost
                v = problem.y1_raw[j] - post_i[j]
                est_mean += v
                jack_ests[1, j, i] = v + abs(held)
                jack_ests[2, j, i] = v - abs(held)
                jack_ests[3, j, i] = v + held
                jack_ests[4, j, i] = v
            end
            est_mean /= tpost
            jack_ests[1, tpost + 1, i] = est_mean + abs(held)
            jack_ests[2, tpost + 1, i] = est_mean - abs(held)
            jack_ests[3, tpost + 1, i] = est_mean + held
            jack_ests[4, tpost + 1, i] = est_mean
        end
    end

    lb = fill(NaN, total + 1)
    ub = fill(NaN, total + 1)
    if conservative
        qerr = _bootstrap_infer_quantile(abs.(held_out), 1 - alpha)
        for j in 1:(tpost + 1)
            col = vec(@view(jack_ests[4, j, :]))
            lb[t0 + j] = _bootstrap_infer_quantile(col, 0.0) - qerr
            ub[t0 + j] = _bootstrap_infer_quantile(col, 1.0) + qerr
        end
    else
        for j in 1:(tpost + 1)
            lower = vec(@view(jack_ests[2, j, :]))
            upper = vec(@view(jack_ests[1, j, :]))
            lb[t0 + j] = _bootstrap_infer_quantile(lower, alpha / 2)
            ub[t0 + j] = _bootstrap_infer_quantile(upper, 1 - alpha / 2)
        end
    end

    y1 = vcat(base_counter, mean(base_counter[(t0 + 1):end]))
    lb_shift = y1 - ub
    ub_shift = y1 - lb
    att = vcat(base_att, mean(base_att[(t0 + 1):end]))
    held = vcat(held_out, base_att[(t0 + 1):end], mean(base_att[(t0 + 1):end]))
    (att = att, lb = lb_shift, ub = ub_shift, heldout_att = held)
end

function _jackknife_unit_std_plain_scm!(X::AbstractMatrix{Float64},
                                       y::AbstractMatrix{Float64},
                                       trt::AbstractVector{Float64})
    problem = _prepare_plain_scm_problem(X, y, trt)
    t0 = problem.t0
    tpost = problem.tpost
    n = size(X, 1)

    base_fit = _solve_plain_scm(problem)
    base_weights = vec(base_fit.weights)
    base_counter = _plain_scm_counterfactual(problem, base_weights)
    base_att = _plain_scm_att(problem, base_weights, base_counter)

    sel_controls = findall(round.(base_weights, digits = 3) .!= 0.0)
    sel_treated = problem.n1 > 1 ? collect(1:problem.n1) : Int[]
    if isempty(sel_controls) && isempty(sel_treated)
        error("No units selected for jackknife")
    end

    ests = Matrix{Float64}(undef, tpost + 1, length(sel_controls) + length(sel_treated))
    col = 1

    if !isempty(sel_controls)
        mean_excl = Vector{Float64}(undef, t0)
        x1_excl = Vector{Float64}(undef, t0)
        Dbuf = Matrix{Float64}(undef, problem.n0 - 1, t0)
        Ybuf = Matrix{Float64}(undef, problem.n0 - 1, tpost)
        Gbuf = Matrix{Float64}(undef, problem.n0 - 1, problem.n0 - 1)
        qbuf = Vector{Float64}(undef, problem.n0 - 1)
        initbuf = Vector{Float64}(undef, max(problem.n0 - 1, 1))
        postbuf = Vector{Float64}(undef, tpost)

        for omit in sel_controls
            @inbounds for j in 1:t0
                mean_excl[j] = (problem.control_pre_sum[j] - problem.X0_raw[omit, j]) / (problem.n0 - 1)
                x1_excl[j] = problem.x1_raw[j] - mean_excl[j]
            end

            row_out = 1
            @inbounds for row in 1:problem.n0
                if row == omit
                    continue
                end
                for j in 1:t0
                    Dbuf[row_out, j] = problem.X0_raw[row, j] - mean_excl[j]
                end
                for j in 1:tpost
                    Ybuf[row_out, j] = problem.Y0_raw[row, j]
                end
                row_out += 1
            end

            mul!(Gbuf, Dbuf, transpose(Dbuf))
            mul!(qbuf, Dbuf, x1_excl)
            @. qbuf = -qbuf

            init = nothing
            if problem.n0 > 1
                if omit > 1
                    copyto!(initbuf, 1, base_weights, 1, omit - 1)
                end
                if omit < problem.n0
                    copyto!(initbuf, omit, base_weights, omit + 1, problem.n0 - omit)
                end
                s = sum(@view(initbuf[1:(problem.n0 - 1)]))
                if s > 0
                    @views @. initbuf[1:(problem.n0 - 1)] = initbuf[1:(problem.n0 - 1)] / s
                    init = @view(initbuf[1:(problem.n0 - 1)])
                end
            end

            weights_i = _solve_simplex_qp_warm(Gbuf, qbuf; init = init)
            mul!(postbuf, transpose(Ybuf), weights_i)
            est_mean = 0.0
            @inbounds for j in 1:tpost
                v = problem.y1_raw[j] - postbuf[j]
                ests[j, col] = v
                est_mean += v
            end
            ests[tpost + 1, col] = est_mean / tpost
            col += 1
        end
    end

    if !isempty(sel_treated)
        control_pre_mean = problem.control_pre_sum ./ problem.n0
        x1_excl = Vector{Float64}(undef, t0)
        y1_excl = Vector{Float64}(undef, tpost)
        qbuf = Vector{Float64}(undef, problem.n0)
        postbuf = Vector{Float64}(undef, tpost)

        for omit in sel_treated
            unit_idx = problem.treated_idx[omit]
            @inbounds for j in 1:t0
                x1_excl[j] = (problem.treated_pre_sum[j] - X[unit_idx, j]) / (problem.n1 - 1) - control_pre_mean[j]
            end
            @inbounds for j in 1:tpost
                y1_excl[j] = (problem.treated_post_sum[j] - y[unit_idx, j]) / (problem.n1 - 1)
            end

            mul!(qbuf, problem.X0, x1_excl)
            @. qbuf = -qbuf
            weights_i = _solve_simplex_qp_warm(problem.gram, qbuf; init = base_weights)
            mul!(postbuf, transpose(problem.Y0_raw), weights_i)

            est_mean = 0.0
            @inbounds for j in 1:tpost
                v = y1_excl[j] - postbuf[j]
                ests[j, col] = v
                est_mean += v
            end
            ests[tpost + 1, col] = est_mean / tpost
            col += 1
        end
    end

    avg = vec(mean(ests; dims = 2))
    se = sqrt.((n - 1) / n .* vec(sum((ests .- avg) .^ 2; dims = 2)))
    (att = vcat(base_att, mean(base_att[(t0 + 1):end])), se = vcat(fill(NaN, t0), se))
end

function _conformal_build_cache(X::AbstractMatrix{Float64},
                               y::AbstractMatrix{Float64},
                               trt::AbstractVector{Float64};
                               trt_mask::Union{AbstractVector{Bool}, Nothing} = nothing,
                               V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                               fixedeff::Bool = false,
                               shift_cols::Union{AbstractVector{Int}, UnitRange{Int}, Nothing} = nothing)
    if size(y, 1) != size(X, 1)
        error("X and y must have matching row counts")
    end
    if size(y, 2) <= 0
        error("y must contain post-treatment columns")
    end

    n = size(X, 1)
    if length(trt) != n
        error("trt must have one entry per row of X/y")
    end

    treated = trt_mask === nothing ? trt .> 0.5 : trt_mask
    control = .!treated
    if count(treated) == 0
        error("No treated units found")
    end
    if count(control) == 0
        error("No control units found")
    end

    Xc = copy(X)
    if fixedeff
        means = vec(sum(Xc; dims = 2))
        means .+= vec(sum(y; dims = 2))
        means ./= size(Xc, 2) + size(y, 2)
        Xc .-= means
    end

    control_means = vec(mean(Xc[control, :], dims = 1))
    Xc = Xc .- transpose(control_means)
    y_ctrl = copy(y[control, :])

    control_idx = findall(control)
    treated_idx = findall(treated)
    X0_raw = Xc[control, :]
    X1 = vec(mean(Xc[treated, :], dims = 1))

    if V !== nothing
        Vmat = Matrix(V)
        if size(Vmat, 1) != size(X0_raw, 2)
            error("V has incompatible dimensions")
        end
        sqrtV = _matrix_sqrt_psd(Vmat)
        X0 = X0_raw * sqrtV
        X1 = reshape(X1, 1, :) * sqrtV |> vec
    else
        X0 = X0_raw
    end

    gram = Matrix(X0 * transpose(X0))
    t0 = size(X, 2)
    shift_idx = if shift_cols === nothing
        collect(t0:t0)
    else
        collect(shift_cols)
    end
    if isempty(shift_idx)
        error("shift_cols must not be empty")
    end
    (
        X0 = X0,
        X0_raw = X0_raw,
        y0 = y_ctrl,
        x1 = X1,
        gram = gram,
        q_base = -(X0 * X1),
        shift_feature = vec(sum(@view(X0[:, shift_idx]); dims = 2)),
        shift_cols = shift_idx,
        control_idx = control_idx,
        treated_idx = treated_idx,
        t0 = t0,
        tpost = size(y, 2),
        n0 = size(X0, 1)
    )
end

@inline function _can_use_pointwise_conformal_stats(ridge::Bool,
                                                    scm::Bool,
                                                    V::Union{AbstractMatrix{Float64}, Nothing},
                                                    fixedeff::Bool)
    !ridge && scm && V === nothing && !fixedeff
end

function _pointwise_conformal_stats(X::AbstractMatrix{Float64},
                                    y::AbstractMatrix{Float64},
                                    trt::AbstractVector{Float64})
    if size(y, 1) != size(X, 1)
        error("X and y must have matching row counts")
    end
    if size(y, 2) <= 0
        error("y must contain post-treatment columns")
    end

    n = size(X, 1)
    if length(trt) != n
        error("trt must have one entry per row of X/y")
    end

    treated = trt .> 0.5
    control = .!treated
    if count(treated) == 0
        error("No treated units found")
    end
    if count(control) == 0
        error("No control units found")
    end

    control_idx = findall(control)
    treated_idx = findall(treated)

    X_control = X[control_idx, :]
    X_means = vec(mean(X_control; dims = 1))
    X0 = X_control .- transpose(X_means)
    x1 = vec(mean(X[treated_idx, :]; dims = 1)) .- X_means

    y_control = y[control_idx, :]
    y_means = vec(mean(y_control; dims = 1))
    y0 = y_control .- transpose(y_means)
    y1 = vec(mean(y[treated_idx, :]; dims = 1)) .- y_means

    (
        X0 = X0,
        y0 = y0,
        x1 = x1,
        y1 = y1,
        gram = Matrix(X0 * transpose(X0)),
        q_base = -(X0 * x1),
        control_idx = control_idx,
        treated_idx = treated_idx,
        t0 = size(X, 2),
        tpost = size(y, 2),
        n0 = length(control_idx)
    )
end

function _pointwise_conformal_cache(stats::NamedTuple, post_index::Int)
    if post_index < 1 || post_index > stats.tpost
        error("post_index out of range")
    end

    shift_feature = copy(@view(stats.y0[:, post_index]))
    gram = copy(stats.gram)
    BLAS.ger!(1.0, shift_feature, shift_feature, gram)

    q_base = copy(stats.q_base)
    @. q_base -= shift_feature * stats.y1[post_index]

    if stats.tpost == 1
        y0 = zeros(Float64, stats.n0, 1)
        y1 = zeros(Float64, 1)
    else
        y0 = Matrix{Float64}(undef, stats.n0, stats.tpost - 1)
        col_out = 1
        @inbounds for col in 1:stats.tpost
            if col == post_index
                continue
            end
            y0[:, col_out] .= @view stats.y0[:, col]
            col_out += 1
        end
        y1 = Vector{Float64}(undef, stats.tpost - 1)
        col_out = 1
        @inbounds for col in 1:stats.tpost
            if col == post_index
                continue
            end
            y1[col_out] = stats.y1[col]
            col_out += 1
        end
    end

    X0 = hcat(stats.X0, shift_feature)
    (
        X0 = X0,
        X0_raw = X0,
        y0 = y0,
        x1 = vcat(stats.x1, stats.y1[post_index]),
        y1 = y1,
        gram = gram,
        q_base = q_base,
        shift_feature = shift_feature,
        shift_cols = [stats.t0 + 1],
        control_idx = stats.control_idx,
        treated_idx = stats.treated_idx,
        t0 = stats.t0 + 1,
        tpost = size(y0, 2),
        n0 = stats.n0,
        direct_resids = true
    )
end

@inline function _placeholder_X(cache::NamedTuple)
    Matrix{Float64}(undef, 0, cache.t0)
end

@inline function _placeholder_y(cache::NamedTuple)
    Matrix{Float64}(undef, 0, cache.tpost)
end

function _fit_from_conformal_cache(cache,
                                  h0::Float64;
                                  init_weights::Union{AbstractVector{Float64}, Nothing} = nothing)
    if cache.n0 == 0
        error("No control units after fixing treatment indicators")
    end

    q = similar(cache.q_base)
    @. q = cache.q_base + h0 * cache.shift_feature

    init = if init_weights === nothing || length(init_weights) != cache.n0
        nothing
    else
        init_weights
    end

    syn = _solve_simplex_qp_warm(cache.gram, q; init = init)
    weights = copy(syn)
    if :direct_resids in propertynames(cache)
        return (
            weights = reshape(weights, :, 1),
            syn = reshape(syn, :, 1),
            mhat = Float64[],
            lambda = NaN,
            l2_imbalance = NaN,
            scaled_l2_imbalance = NaN,
            t0 = cache.t0,
            tpost = cache.tpost,
            control_idx = cache.control_idx,
            treated_idx = cache.treated_idx
        )
    end

    m1_pre = vec(weights' * cache.X0_raw)
    m1_post = vec(weights' * cache.y0)

    (
        weights = reshape(weights, :, 1),
        syn = reshape(syn, :, 1),
        mhat = vcat(m1_pre, m1_post),
        lambda = NaN,
        l2_imbalance = NaN,
        scaled_l2_imbalance = NaN,
        t0 = cache.t0,
        tpost = cache.tpost,
        control_idx = cache.control_idx,
        treated_idx = cache.treated_idx
    )
end

function _conformal_resids_from_cache(cache,
                                     X,
                                     y,
                                     fit::NamedTuple,
                                     h0::Float64)
    w = vec(fit.weights)
    if :direct_resids in propertynames(cache)
        resids = vcat(
            cache.x1 .- vec(transpose(cache.X0_raw) * w),
            cache.y1 .- vec(transpose(cache.y0) * w)
        )
        if h0 != 0.0
            @inbounds for idx in cache.shift_cols
                resids[idx] -= h0
            end
        end
        return resids
    end

    mhat = fit.mhat
    t0 = fit.t0
    tpost = fit.tpost
    control = fit.control_idx
    treated_idx = cache.treated_idx

    xbar = vec(X[control, 1:t0]' * w)
    ybar = vec(y[control, :]' * w)
    y0 = mhat .+ vcat(xbar .- mhat[1:t0], ybar .- mhat[(t0 + 1):(t0 + tpost)])

    treated_obs = vcat(
        vec(mean(X[treated_idx, :], dims = 1)),
        vec(mean(y[treated_idx, :], dims = 1))
    )
    if h0 != 0.0
        @inbounds for idx in cache.shift_cols
            treated_obs[idx] -= h0
        end
    end

    vec(treated_obs) .- y0
end

function _fit_augsynth_single!(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                               trt::AbstractVector{Float64};
                               ridge::Bool,
                               scm::Bool,
                               lambda::Float64,
                               trt_mask::Union{AbstractVector{Bool}, Nothing} = nothing,
                               holdout_length::Int = 1,
                               min1se::Bool = true,
                               V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                               fixedeff::Bool = false,
                               init_weights::Union{AbstractVector{Float64}, Nothing} = nothing)
    if size(y, 1) != size(X, 1)
        error("X and y must have matching row counts")
    end
    if size(y, 2) <= 0
        error("y must contain post-treatment columns")
    end

    n = size(X, 1)
    if length(trt) != n
        error("trt must have one entry per row of X/y")
    end

    treated = trt_mask === nothing ? trt .> 0.5 : trt_mask
    control = .!treated
    if count(treated) == 0
        error("No treated units found")
    end
    if count(control) == 0
        error("No control units found")
    end

    Xc = copy(X)
    if fixedeff
        means = vec(sum(Xc; dims = 2))
        means .+= vec(sum(y; dims = 2))
        means ./= size(Xc, 2) + size(y, 2)
        Xc .-= means
    end

    control_means = vec(mean(Xc[control, :], dims = 1))
    Xc = Xc .- transpose(control_means)

    n0 = size(Xc[control, :], 1)
    if n0 == 0
        error("No control units after fixing treatment indicators")
    end

    X0 = Xc[control, :]
    X1 = vec(mean(Xc[treated, :], dims = 1))

    if V !== nothing
        Vmat = Matrix(V)
        if size(Vmat, 1) != size(X0, 2)
            error("V has incompatible dimensions")
        end
        sqrtV = _matrix_sqrt_psd(Vmat)
        X0 = X0 * sqrtV
        X1 = reshape(X1, 1, :) * sqrtV |> vec
    end

    control_idx = findall(control)
    treated_idx = findall(treated)

    syn = if scm
        init = if init_weights === nothing || length(init_weights) != n0
            nothing
        else
            init_weights
        end
        gram = Matrix(X0 * transpose(X0))
        q = -(X0 * X1)
        _solve_simplex_qp_warm(gram, q; init = init)
    else
        fill(1.0 / n0, n0)
    end
    weights = copy(syn)
    λ = NaN

    if ridge
        if !isfinite(lambda) || lambda < 0.0
            error("lambda must be finite and non-negative")
        end
        λ = lambda
        weights = weights .+ _ridge_adjustment(X0, X1, syn, λ)
    end

    unif_w = fill(1.0 / n0, n0)
    l2_imbalance = sqrt(sum((transpose(X0) * weights .- X1) .^ 2))
    unif_l2 = sqrt(sum((transpose(X0) * unif_w .- X1) .^ 2))
    scaled_l2 = iszero(unif_l2) ? NaN : l2_imbalance / unif_l2

    base_series = vcat(
        vec(weights' * Xc[control, :]),
        vec(weights' * y[control, :])
    )

    return (
        weights = reshape(weights, :, 1),
        syn = reshape(syn, :, 1),
        lambda = λ,
        mhat = base_series,
        l2_imbalance = l2_imbalance,
        scaled_l2_imbalance = scaled_l2,
        t0 = size(X, 2),
        tpost = size(y, 2),
        control_idx = control_idx,
        treated_idx = treated_idx
    )
end

function _predict_counterfactual(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                                trt::AbstractVector{Float64},
                                fit::NamedTuple)
    if fit.t0 < 1 || fit.tpost < 1
        error("fit must include pre- and post-treatment periods")
    end

    t0 = fit.t0
    compact = fit.mhat isa AbstractVector{Float64}
    control_idx = fit.control_idx
    treated_idx = fit.treated_idx
    ncontrol = length(control_idx)
    ntreated = length(treated_idx)
    tpre = t0
    tpost = fit.tpost
    if ncontrol == 0 || ntreated == 0
        error("No control units available for prediction")
    end

    if compact
        m1 = vec(fit.mhat)
        m1_pre = m1[1:tpre]
        m1_post = m1[(tpre + 1):(tpre + tpost)]
    else
        m1 = vec(mean(fit.mhat[treated_idx, :], dims = 1))
        m1_pre = m1[1:tpre]
        m1_post = m1[(tpre + 1):(tpre + tpost)]
    end
    w = vec(fit.weights)
    xbar = vec(X[control_idx, 1:tpre]' * w)
    ybar = vec(y[control_idx, :]' * w)

    if !compact
        control_mhat = vec(mean(fit.mhat[control_idx, :], dims = 1))
        m1_pre = control_mhat[1:tpre]
        m1_post = control_mhat[(tpre + 1):(tpre + tpost)]
    end

    resid_pre = Vector{Float64}(undef, tpre)
    resid_post = Vector{Float64}(undef, tpost)

    @inbounds for j in 1:tpre
        resid_pre[j] = xbar[j] - m1_pre[j]
    end
    @inbounds for j in 1:tpost
        resid_post[j] = ybar[j] - m1_post[j]
    end
    return m1 .+ vcat(resid_pre, resid_post)
end

function _predict_att(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                     trt::AbstractVector{Float64}, fit::NamedTuple)
    treated_idx = fit.treated_idx
    y0 = vec(_predict_counterfactual(X, y, trt, fit))
    treated_obs = vcat(
        vec(mean(X[treated_idx, :], dims = 1)),
        vec(mean(y[treated_idx, :], dims = 1))
    )
    att = treated_obs .- y0
    att
end

function _bootstrap_infer_quantile(v::AbstractVector{Float64}, p::Float64)
    _quantile(collect(v), p)
end

function _jackknife_fit_stat_matrix(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                                   trt::AbstractVector{Float64};
                                   ridge::Bool,
                                   scm::Bool,
                                   lambda::Float64,
                                   trt_mask::Union{AbstractVector{Bool}, Nothing} = nothing,
                                   holdout_length::Int = 1,
                                   min1se::Bool = true,
                                   V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                                   fixedeff::Bool = false)
    fit = _fit_augsynth_single!(X, y, trt;
        ridge = ridge, scm = scm, lambda = lambda,
        trt_mask = trt_mask,
        holdout_length = holdout_length, min1se = min1se,
        V = V, fixedeff = fixedeff
    )
    att = _predict_att(X, y, trt, fit)
    out = vcat(att, mean(att[(size(X, 2) + 1):end]))
    (fit = fit, att = att, out = out, control_count = sum(.!(trt .> 0.5)),
     treated = trt .> 0.5)
end

function _jackknife_plus_row!(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                             trt::AbstractVector{Float64};
                             ridge::Bool,
                             scm::Bool,
                             lambda::Float64,
                             conservative::Bool,
                             alpha::Float64,
                             holdout_length::Int = 1,
                             min1se::Bool = true,
                             V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                             fixedeff::Bool = false)
    if _can_use_plain_scm_fastpath(ridge, scm, V, fixedeff)
        return _jackknife_plus_row_plain_scm!(
            X, y, trt;
            conservative = conservative,
            alpha = alpha
        )
    end

    trt_mask = trt .> 0.5
    base = _jackknife_fit_stat_matrix(
        X, y, trt;
        ridge = ridge, scm = scm, lambda = lambda,
        holdout_length = holdout_length, min1se = min1se,
        trt_mask = trt_mask,
        V = V, fixedeff = fixedeff
    )
    base_att = base.att
    n, t0 = size(X)
    tpost = size(y, 2)
    t_final = t0 + tpost
    trt_mask_view = trt .> 0.5
    treated_col_means = vec(mean(X[trt_mask_view, :], dims = 1))

    jack_ests = Array{Float64}(undef, 4, tpost + 1, t0)
    held_out = zeros(t0)
    subXbuf = Matrix{Float64}(undef, n, t0 - 1)
    subYbuf = Matrix{Float64}(undef, n, tpost + 1)

    if Threads.nthreads() > 1 && t0 > 8
        nthreads = Threads.nthreads()
        x_pool = [Matrix{Float64}(undef, n, t0 - 1) for _ in 1:nthreads]
        y_pool = [Matrix{Float64}(undef, n, tpost + 1) for _ in 1:nthreads]
        base_w = vec(base.fit.weights)
        Threads.@threads for i in 1:t0
            tid = threadid()
            subX = x_pool[tid]
            suby = y_pool[tid]
            _fill_omit_pre_col!(subX, suby, X, y, i)
            sub = _fit_augsynth_single!(
                subX, suby, trt;
                ridge = ridge, scm = scm, lambda = lambda,
                trt_mask = trt_mask,
                init_weights = base_w,
                holdout_length = holdout_length, min1se = min1se,
                V = V, fixedeff = fixedeff
            )
            counter = _predict_counterfactual(subX, suby, trt, sub)

            est = @view(counter[(t0 + 1):length(counter)])
            est_mean = 0.0
            @inbounds for j in eachindex(est)
                est_mean += est[j]
            end
            est_mean /= tpost

            held = treated_col_means[i] - counter[t0]
            held_out[i] = held
            @inbounds begin
                for j in 1:tpost
                    v = est[j]
                    jack_ests[1, j, i] = v + abs(held)
                    jack_ests[2, j, i] = v - abs(held)
                    jack_ests[3, j, i] = v + held
                    jack_ests[4, j, i] = v
                end
                jack_ests[1, tpost + 1, i] = est_mean + abs(held)
                jack_ests[2, tpost + 1, i] = est_mean - abs(held)
                jack_ests[3, tpost + 1, i] = est_mean + held
                jack_ests[4, tpost + 1, i] = est_mean
            end
        end
    else
        for i in 1:t0
            _fill_omit_pre_col!(subXbuf, subYbuf, X, y, i)

            sub = _fit_augsynth_single!(
                subXbuf, subYbuf, trt;
                ridge = ridge, scm = scm, lambda = lambda,
                trt_mask = trt_mask,
                init_weights = vec(base.fit.weights),
                holdout_length = holdout_length, min1se = min1se,
                V = V, fixedeff = fixedeff
            )
            counter = _predict_counterfactual(subXbuf, subYbuf, trt, sub)

            est = @view(counter[(t0 + 1):length(counter)])
            est_mean = 0.0
            @inbounds for j in eachindex(est)
                est_mean += est[j]
            end
            est_mean /= tpost

            held = treated_col_means[i] - counter[t0]
            held_out[i] = held
            @inbounds begin
                for j in 1:tpost
                    v = est[j]
                    jack_ests[1, j, i] = v + abs(held)
                    jack_ests[2, j, i] = v - abs(held)
                    jack_ests[3, j, i] = v + held
                    jack_ests[4, j, i] = v
                end
                jack_ests[1, tpost + 1, i] = est_mean + abs(held)
                jack_ests[2, tpost + 1, i] = est_mean - abs(held)
                jack_ests[3, tpost + 1, i] = est_mean + held
                jack_ests[4, tpost + 1, i] = est_mean
            end
        end
    end

    if conservative
        qerr = _bootstrap_infer_quantile(abs.(held_out), 1 - alpha)
        lb = fill(NaN, t_final + 1)
        ub = similar(lb)
        for j in 1:(tpost + 1)
            col = vec(@view(jack_ests[4, j, :]))
            lb[t0 + j] = _bootstrap_infer_quantile(col, 0.0) - qerr
            ub[t0 + j] = _bootstrap_infer_quantile(col, 1.0) + qerr
        end
    else
        lb = fill(NaN, t_final + 1)
        ub = fill(NaN, t_final + 1)
        for j in 1:(tpost + 1)
            upper = vec(@view(jack_ests[1, j, :]))
            lower = vec(@view(jack_ests[2, j, :]))
            lb[t0 + j] = _bootstrap_infer_quantile(lower, alpha / 2)
            ub[t0 + j] = _bootstrap_infer_quantile(upper, 1 - alpha / 2)
        end
    end

    base_counter = _predict_counterfactual(X, y, trt, base.fit)
    y1 = vcat(base_counter, mean(base_counter[(t0 + 1):end]))
    lb_shift = y1 - ub
    ub_shift = y1 - lb

    out_att = vcat(base_att, mean(base_att[(t0 + 1):end]))
    held = vcat(held_out, base_att[(t0 + 1):end], mean(base_att[(t0 + 1):end]))
    (att = out_att, lb = lb_shift, ub = ub_shift, heldout_att = held)
end

function _jackknife_unit_std!(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                             trt::AbstractVector{Float64};
                             ridge::Bool,
                             scm::Bool,
                             lambda::Float64,
                             holdout_length::Int = 1,
                             min1se::Bool = true,
                             V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                             fixedeff::Bool = false)
    if _can_use_plain_scm_fastpath(ridge, scm, V, fixedeff)
        return _jackknife_unit_std_plain_scm!(X, y, trt)
    end

    n, t0 = size(X)
    tpost = size(y, 2)
    t_final = t0 + tpost

    trt_mask = trt .> 0.5
    control_mask = .!trt_mask

    base = _jackknife_fit_stat_matrix(
        X, y, trt;
        ridge = ridge, scm = scm, lambda = lambda,
        holdout_length = holdout_length, min1se = min1se, trt_mask = trt_mask,
        V = V, fixedeff = fixedeff
    )
    base_weights = vec(base.fit.weights)
    att = base.att

    nnz = falses(n)
    nnz[control_mask] .= (round.(base_weights, digits = 3) .!= 0.0)
    if count(trt_mask) > 1
        nnz[trt_mask] .= true
    end
    sel = findall(nnz)
    if isempty(sel)
        error("No units selected for jackknife")
    end

    base_control_count = sum(control_mask)
    control_positions = zeros(Int, n)
    cpos = 1
    for i in 1:n
        if control_mask[i]
            control_positions[i] = cpos
            cpos += 1
        end
    end

    sel_len = length(sel)
    mean_ests = zeros(tpost + 1)
    m2_ests = zeros(tpost + 1)
    sample_count = 0

    if Threads.nthreads() > 1 && length(sel) > 8
        nthreads = Threads.nthreads()
        x_pool = [Matrix{Float64}(undef, n, t0) for _ in 1:nthreads]
        y_pool = [Matrix{Float64}(undef, n, tpost) for _ in 1:nthreads]
        trt_pool = [Vector{Float64}(undef, n) for _ in 1:nthreads]
        init_pool = [Vector{Float64}(undef, max(base_control_count - 1, 1)) for _ in 1:nthreads]
        thread_counts = zeros(Int, nthreads)
        thread_means = [zeros(tpost + 1) for _ in 1:nthreads]
        thread_m2 = [zeros(tpost + 1) for _ in 1:nthreads]

        Threads.@threads for idx in eachindex(sel)
            i = sel[idx]
            subX = x_pool[threadid()]
            subY = y_pool[threadid()]
            subTrt = trt_pool[threadid()]
            tid = threadid()
            init = nothing

            if control_mask[i]
                k = control_positions[i]
                init = init_pool[tid]
                fill!(init, 0.0)
                if k > 1
                    copyto!(init, 1, base_weights, 1, k - 1)
                end
                if k < base_control_count
                    copyto!(init, k, base_weights, k + 1, base_control_count - k)
                end
                if base_control_count > 1
                    s = sum(view(init, 1:(base_control_count - 1)))
                    if s > 0
                        invs = 1.0 / s
                        @inbounds for j in 1:(base_control_count - 1)
                            init[j] *= invs
                        end
                    else
                        init = nothing
                    end
                end
            end

            _fill_omit_row!(subX, subY, subTrt, X, y, trt, i)
            subTrtMask = subTrt[1:(n - 1)] .> 0.5
            sub = _fit_augsynth_single!(
                subX[1:(n - 1), :], subY[1:(n - 1), :], subTrt[1:(n - 1)];
                ridge = ridge, scm = scm, lambda = lambda,
                trt_mask = subTrtMask,
                init_weights = init,
                holdout_length = holdout_length, min1se = min1se,
                V = V, fixedeff = fixedeff
            )
            att_sub = _predict_att(
                subX[1:(n - 1), :],
                subY[1:(n - 1), :],
                subTrt[1:(n - 1)],
                sub
            )

            est = @view(att_sub[(t0 + 1):end])
            tpost1 = tpost + 1
            est_sum = 0.0
            @inbounds for j in 1:tpost
                x = est[j]
                est_sum += x
            end
            est_mean = est_sum / tpost

            local_count = thread_counts[tid] + 1
            local_mean = thread_means[tid]
            local_m2 = thread_m2[tid]
            @inbounds for j in 1:tpost
                x = est[j]
                delta = x - local_mean[j]
                local_mean[j] += delta / local_count
                delta2 = x - local_mean[j]
                local_m2[j] += delta * delta2
            end
            x = est_mean
            delta = x - local_mean[tpost1]
            local_mean[tpost1] += delta / local_count
            delta2 = x - local_mean[tpost1]
            local_m2[tpost1] += delta * delta2
            thread_counts[tid] = local_count
        end

        running_count = 0
        running_means = mean_ests
        running_m2 = m2_ests
        for tid in 1:nthreads
            c = thread_counts[tid]
            if c == 0
                continue
            end
            local_mean = thread_means[tid]
            local_m2 = thread_m2[tid]
            if running_count == 0
                copyto!(running_means, local_mean)
                copyto!(running_m2, local_m2)
                running_count = c
                continue
            end
            n1 = Float64(running_count)
            n2 = Float64(c)
            denom = n1 + n2
            @inbounds for j in 1:(tpost + 1)
                delta = local_mean[j] - running_means[j]
                running_means[j] += (n2 / denom) * delta
                running_m2[j] += local_m2[j] + (n1 * n2 / denom) * delta^2
            end
            running_count += c
        end
        sample_count = running_count
    else
        subX = Matrix{Float64}(undef, n - 1, t0)
        subY = Matrix{Float64}(undef, n - 1, tpost)
        subTrt = Vector{Float64}(undef, n - 1)
        init = nothing
        init_pool = Vector{Float64}(undef, max(base_control_count - 1, 1))
        for (k, i) in pairs(sel)
            init = nothing
            if control_mask[i]
                cpos = control_positions[i]
                if base_control_count > 1
                    fill!(init_pool, 0.0)
                    if cpos > 1
                        copyto!(init_pool, 1, base_weights, 1, cpos - 1)
                    end
                    if cpos < base_control_count
                        copyto!(init_pool, cpos, base_weights, cpos + 1, base_control_count - cpos)
                    end
                    s = sum(view(init_pool, 1:(base_control_count - 1)))
                    if s > 0
                        invs = 1.0 / s
                        @inbounds for j in 1:(base_control_count - 1)
                            init_pool[j] *= invs
                        end
                        init = view(init_pool, 1:(base_control_count - 1))
                    else
                        init = nothing
                    end
                end
            end
            _fill_omit_row!(subX, subY, subTrt, X, y, trt, i)
            subTrtMask = subTrt[1:(n - 1)] .> 0.5
            sub = _fit_augsynth_single!(
                subX, subY, subTrt;
                ridge = ridge, scm = scm, lambda = lambda,
                trt_mask = subTrtMask,
                init_weights = init,
                holdout_length = holdout_length, min1se = min1se,
                V = V, fixedeff = fixedeff
            )
            att_sub = _predict_att(subX, subY, subTrt, sub)
            est = @view(att_sub[(t0 + 1):end])
            est_sum = 0.0
            @inbounds for j in eachindex(est)
                est_sum += est[j]
            end
            est_mean = est_sum / tpost
            sample_count += 1
            @inbounds for j in 1:tpost
                x = est[j]
                delta = x - mean_ests[j]
                mean_ests[j] += delta / sample_count
                delta2 = x - mean_ests[j]
                m2_ests[j] += delta * delta2
            end
            x = est_mean
            delta = x - mean_ests[tpost + 1]
            mean_ests[tpost + 1] += delta / sample_count
            delta2 = x - mean_ests[tpost + 1]
            m2_ests[tpost + 1] += delta * delta2
        end
    end

    if sample_count != sel_len
        error("jackknife unit standard error sample count does not match selected units")
    end
    se = sqrt.((n - 1) / n .* m2_ests)
    (att = vcat(att, mean(att[(t0 + 1):end])), se = vcat(fill(NaN, t0), se))
end

@inline function _jackknife_permute_stat(x::AbstractVector{Float64}, q::Float64)
    n = length(x)
    if n == 0
        return 0.0
    end
    if q == 1.0
        s = 0.0
        @inbounds for j in eachindex(x)
            s += abs(x[j])
        end
        return s / sqrt(n)
    elseif q == 2.0
        return sqrt(sum(abs2, x)) / sqrt(n)
    end
    s = 0.0
    invsqn = 1 / sqrt(n)
    @inbounds for j in eachindex(x)
        s += abs(x[j])^q
    end
    (s * invsqn) ^ (1 / q)
end

@inline function _jackknife_permute_stat_cyclic(x::AbstractVector{Float64}, t0::Int, q::Float64, start::Int)
    t = length(x)
    tpost = t - t0
    if tpost <= 0
        return 0.0
    end
    acc = 0.0
    idx = mod1(t0 + start + 1, t)
    for _ in 1:tpost
        acc += abs(x[idx])^q
        idx = idx == t ? 1 : idx + 1
    end
    if q == 1.0
        return acc / sqrt(tpost)
    elseif q == 2.0
        return sqrt(acc) / sqrt(tpost)
    end
    (acc / sqrt(tpost)) ^ (1 / q)
end

@inline function _single_post_permute_stats(obs::AbstractVector{Float64})
    out = Vector{Float64}(undef, length(obs))
    @inbounds @simd for i in eachindex(obs)
        out[i] = abs(obs[i])
    end
    out
end

function _compute_permute_test_stats(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                                    trt::AbstractVector{Float64};
                                    h0::Float64,
                                    post_length::Int,
                                    type::Int,
                                    q::Float64,
                                    ns::Int,
                                    ridge::Bool,
                                    scm::Bool,
                                    lambda::Float64,
                                    holdout_length::Int = 1,
                                    min1se::Bool = true,
                                    V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                                    fixedeff::Bool = false,
                                    init_weights::Union{AbstractVector{Float64}, Nothing} = nothing,
                                    threaded_permutations::Bool = true,
                                    fit_cache::Union{NamedTuple, Nothing} = nothing)
    ncol_x = size(X, 2)
    t0 = ncol_x - post_length
    if t0 <= 0
        error("post_length must be smaller than number of columns in X")
    end
    tpost = t0 + post_length

    use_cache = fit_cache !== nothing && !ridge && scm
    if use_cache
        fit = _fit_from_conformal_cache(fit_cache, h0; init_weights = init_weights)
        resids = _conformal_resids_from_cache(fit_cache, X, y, fit, h0)
    else
        X2 = copy(X)
        treated = trt .> 0.5
        X2[treated, (t0 + 1):tpost] .-= h0
        fit = _fit_augsynth_single!(
            X2, y, trt;
            ridge = ridge, scm = scm, lambda = lambda,
            init_weights = init_weights,
            holdout_length = holdout_length, min1se = min1se,
            V = V, fixedeff = fixedeff
        )
        resids = _predict_att(X2, y, trt, fit)
    end
    obs = resids[1:tpost]
    stat = _jackknife_permute_stat(obs[(t0 + 1):tpost], q)

    if type == 0 && post_length == 1
        return (resids = obs, test_stats = _single_post_permute_stats(obs), stat = stat, weights = vec(fit.weights))
    end

    if type == 0
        out = zeros(ns)
        nthreads = Threads.nthreads()
        if threaded_permutations && nthreads > 1 && ns > 16
            reorder_pool = [similar(obs) for _ in 1:nthreads]
            rng_pool = [Random.Xoshiro(0x3d4f1d2f + tid) for tid in 1:nthreads]
            Threads.@threads for i in 1:ns
                tid = threadid()
                reorder = reorder_pool[tid]
                rng = rng_pool[tid]
                copyto!(reorder, obs)
                shuffle!(rng, reorder)
                out[i] = _jackknife_permute_stat(@view(reorder[(t0 + 1):tpost]), q)
            end
        else
            reorder = similar(obs)
            for i in 1:ns
                copyto!(reorder, obs)
                shuffle!(reorder)
                out[i] = _jackknife_permute_stat(@view(reorder[(t0 + 1):tpost]), q)
            end
        end
    else
        out = zeros(tpost)
        for i in 1:tpost
            out[i] = _jackknife_permute_stat_cyclic(obs, t0, q, i)
        end
    end

    (resids = obs, test_stats = out, stat = stat, weights = vec(fit.weights))
end

function _compute_permute_pval(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                              trt::AbstractVector{Float64};
                              h0::Float64,
                              post_length::Int,
                              type::Int,
                              q::Float64,
                              ns::Int,
                              ridge::Bool,
                              scm::Bool,
                              lambda::Float64,
                              holdout_length::Int = 1,
                              min1se::Bool = true,
                              V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                              fixedeff::Bool = false,
                              init_weights::Union{AbstractVector{Float64}, Nothing} = nothing,
                              threaded_permutations::Bool = true,
                              fit_cache::Union{NamedTuple, Nothing} = nothing)
    t0 = size(X, 2) - post_length
    tpost = t0 + post_length
    out = _compute_permute_test_stats(
        X, y, trt;
        h0 = h0, post_length = post_length,
        type = type, q = q, ns = ns,
        ridge = ridge, scm = scm,
        lambda = lambda, holdout_length = holdout_length,
        min1se = min1se, V = V, fixedeff = fixedeff,
        init_weights = init_weights,
        threaded_permutations = threaded_permutations,
        fit_cache = fit_cache
    )
    (mean(out.stat .<= out.test_stats), out.weights)
end

function _compute_permute_ci_grid(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                                 trt::AbstractVector{Float64},
                                 grid::AbstractVector{Float64},
                                 post_length::Int,
                                 alpha::Float64,
                                 type::Int,
                                 q::Float64,
                                 ns::Int,
                                 ridge::Bool,
                                 scm::Bool,
                                 lambda::Float64,
                                 ;
                                 holdout_length::Int = 1,
                                 min1se::Bool = true,
                                 V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                                 fixedeff::Bool = false,
                                 init_weights::Union{AbstractVector{Float64}, Nothing} = nothing,
                                 threaded_permutations::Bool = true,
                                 fit_cache::Union{NamedTuple, Nothing} = nothing)
    extended = sort(unique(vcat(grid, 0.0)))
    ps = zeros(Float64, length(extended))
    last_weights = init_weights
    for (k, h0) in pairs(extended)
        pval, last_weights = _compute_permute_pval(
            X, y, trt;
            h0 = h0, post_length = post_length,
            type = type, q = q, ns = ns,
            ridge = ridge, scm = scm, lambda = lambda,
            holdout_length = holdout_length, min1se = min1se,
            V = V, fixedeff = fixedeff, init_weights = last_weights,
            threaded_permutations = threaded_permutations,
            fit_cache = fit_cache
        )
        ps[k] = pval
    end

    valid = findall(x -> x >= alpha, ps)
    if isempty(valid)
        p_zero = findfirst(==(0.0), extended)
        return (
            NaN,
            NaN,
            p_zero === nothing ? NaN : ps[p_zero],
            last_weights
        )
    end
    p_zero = findfirst(==(0.0), extended)
    return (
        minimum(extended[valid]),
        maximum(extended[valid]),
        p_zero === nothing ? NaN : ps[p_zero],
        last_weights
    )
end

function _compute_permute_ci(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                            trt::AbstractVector{Float64},
                            grid::AbstractVector{Float64},
                            post_length::Int,
                            alpha::Float64,
                            type::Int,
                            q::Float64,
                            ns::Int,
                            ridge::Bool,
                            scm::Bool,
                            lambda::Float64,
                            ;
                            holdout_length::Int = 1,
                            min1se::Bool = true,
                            conformal_mode::Int = CONFORMAL_MODE_FAST,
                            V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                            fixedeff::Bool = false,
                            init_weights::Union{AbstractVector{Float64}, Nothing} = nothing,
                            threaded_permutations::Bool = true,
                            fit_cache::Union{NamedTuple, Nothing} = nothing)
    grid_sorted = sort(unique(collect(grid)))
    if isempty(grid_sorted)
        error("grid must not be empty")
    end
    if conformal_mode == CONFORMAL_MODE_REFERENCE
        return _compute_permute_ci_grid(
            X, y, trt, grid_sorted,
            post_length, alpha, type, q, ns,
            ridge, scm, lambda;
            holdout_length = holdout_length,
            min1se = min1se,
            V = V,
            fixedeff = fixedeff,
            init_weights = init_weights,
            threaded_permutations = threaded_permutations,
            fit_cache = fit_cache
        )
    elseif conformal_mode != CONFORMAL_MODE_FAST
        error("invalid conformal_mode")
    end

    center = 0.5 * (first(grid_sorted) + last(grid_sorted))
    radius = 0.5 * (last(grid_sorted) - first(grid_sorted))
    if !isfinite(radius) || radius <= 0.0
        radius = max(abs(center), 1.0)
    end

    tol = max(
        2.0 * radius / max(length(grid_sorted) - 1, 1),
        sqrt(eps(Float64)) * max(1.0, abs(center), radius)
    )
    search_radius = max(radius, tol)
    max_expand = 8
    max_refine = max(6, ceil(Int, log2(max(length(grid_sorted), 2))) + 4)

    cache = Dict{Float64, Tuple{Float64, Vector{Float64}}}()
    function eval_p(h0::Float64, seed_weights::Union{AbstractVector{Float64}, Nothing})
        if haskey(cache, h0)
            return cache[h0]
        end
        pval, weights = _compute_permute_pval(
            X, y, trt;
            h0 = h0, post_length = post_length,
            type = type, q = q, ns = ns,
            ridge = ridge, scm = scm, lambda = lambda,
            holdout_length = holdout_length, min1se = min1se,
            V = V, fixedeff = fixedeff, init_weights = seed_weights,
            threaded_permutations = threaded_permutations,
            fit_cache = fit_cache
        )
        result = (pval, copy(weights))
        cache[h0] = result
        result
    end

    p_center, w_center = eval_p(center, init_weights)
    p_zero, w_zero = if center == 0.0
        p_center, w_center
    else
        eval_p(0.0, w_center)
    end

    fallback_radius = search_radius
    fallback_points = max(length(grid_sorted), 25)
    fallback = function()
        dense_grid = collect(range(center - fallback_radius, stop = center + fallback_radius, length = fallback_points))
        _compute_permute_ci_grid(
            X, y, trt, dense_grid,
            post_length, alpha, type, q, ns,
            ridge, scm, lambda;
            holdout_length = holdout_length,
            min1se = min1se,
            V = V,
            fixedeff = fixedeff,
            init_weights = w_zero,
            threaded_permutations = threaded_permutations,
            fit_cache = fit_cache
        )
    end

    if p_center < alpha
        return fallback()
    end

    function lower_boundary()
        accepted_h = center
        accepted_w = w_center
        rejected_h = NaN
        probe_radius = search_radius
        for _ in 1:max_expand
            candidate = center - probe_radius
            pval, weights = eval_p(candidate, accepted_w)
            if pval >= alpha
                accepted_h = candidate
                accepted_w = weights
                probe_radius *= 2.0
                fallback_radius = max(fallback_radius, probe_radius)
            else
                rejected_h = candidate
                break
            end
        end
        if !isfinite(rejected_h)
            return (NaN, false)
        end

        left = rejected_h
        right = accepted_h
        right_w = accepted_w
        for _ in 1:max_refine
            if right - left <= tol
                break
            end
            mid = 0.5 * (left + right)
            pval, weights = eval_p(mid, right_w)
            if pval >= alpha
                right = mid
                right_w = weights
            else
                left = mid
            end
        end
        (right, true)
    end

    function upper_boundary()
        accepted_h = center
        accepted_w = w_center
        rejected_h = NaN
        probe_radius = search_radius
        for _ in 1:max_expand
            candidate = center + probe_radius
            pval, weights = eval_p(candidate, accepted_w)
            if pval >= alpha
                accepted_h = candidate
                accepted_w = weights
                probe_radius *= 2.0
                fallback_radius = max(fallback_radius, probe_radius)
            else
                rejected_h = candidate
                break
            end
        end
        if !isfinite(rejected_h)
            return (NaN, false)
        end

        left = accepted_h
        right = rejected_h
        left_w = accepted_w
        for _ in 1:max_refine
            if right - left <= tol
                break
            end
            mid = 0.5 * (left + right)
            pval, weights = eval_p(mid, left_w)
            if pval >= alpha
                left = mid
                left_w = weights
            else
                right = mid
            end
        end
        (left, true)
    end

    lo, ok_lo = lower_boundary()
    hi, ok_hi = upper_boundary()
    if !ok_lo || !ok_hi || !isfinite(lo) || !isfinite(hi) || lo > hi
        return fallback()
    end

    (lo, hi, p_zero, w_zero)
end

function _conformal!(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                    trt::AbstractVector{Float64};
                    ridge::Bool,
                    scm::Bool,
                    lambda::Float64,
                    alpha::Float64,
                    type::Int,
                    q::Float64,
                    ns::Int,
                    grid_size::Int,
                    conformal_mode::Int = CONFORMAL_MODE_FAST,
                    holdout_length::Int = 1,
                    min1se::Bool = true,
                    V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                    fixedeff::Bool = false)
    n, t0 = size(X)
    tpost = size(y, 2)
    t_total = t0 + tpost

    base = _fit_augsynth_single!(
        X, y, trt;
        ridge = ridge, scm = scm, lambda = lambda,
        holdout_length = holdout_length, min1se = min1se,
        V = V, fixedeff = fixedeff
    )
    base_att = _predict_att(X, y, trt, base)
    post_sd = sqrt(mean(base_att[(t0 + 1):end] .^ 2))

    ci = zeros(Float64, 3, tpost)
    Xj = Matrix{Float64}(undef, n, t0 + 1)
    yj = Matrix{Float64}(undef, n, max(1, tpost - 1))
    base_pred = _predict_counterfactual(X, y, trt, base)
    base_weights = vec(base.weights)
    nthreads = Threads.nthreads()
    # Pointwise iid conformal CIs now use the exact single-post path, so
    # threading those tiny jobs is usually slower than leaving them serial.
    # Reserve threads for the true multi-post iid null where the permutation
    # work still scales with ns.
    use_outer_threads = tpost > 1 && nthreads > 1 && type != 0
    use_inner_threads = type == 0 && nthreads > 1 && ns >= 500
    use_pointwise_stats = _can_use_pointwise_conformal_stats(ridge, scm, V, fixedeff)
    pointwise_stats = use_pointwise_stats ? _pointwise_conformal_stats(X, y, trt) : nothing
    if use_outer_threads
        lo_out = zeros(Float64, tpost)
        hi_out = zeros(Float64, tpost)
        pv_out = zeros(Float64, tpost)
        final_weights = base_weights
        xj_pool = [Matrix{Float64}(undef, n, t0 + 1) for _ in 1:nthreads]
        yj_pool = [Matrix{Float64}(undef, n, tpost - 1) for _ in 1:nthreads]

        Threads.@threads for j in 1:tpost
            tid = threadid()
            Xj_thread = xj_pool[tid]
            yj_thread = yj_pool[tid]

            Xj_eval, yj_eval, conf_cache = if use_pointwise_stats
                cache = _pointwise_conformal_cache(pointwise_stats, j)
                (_placeholder_X(cache), _placeholder_y(cache), cache)
            else
                Xj_thread[:, 1:t0] .= X
                @views Xj_thread[:, t0 + 1] .= y[:, j]

                if j == 1
                    @views yj_thread[:, 1:(tpost - 1)] .= y[:, 2:tpost]
                elseif j == tpost
                    @views yj_thread[:, 1:(tpost - 1)] .= y[:, 1:(tpost - 1)]
                else
                    @views yj_thread[:, 1:(j - 1)] .= y[:, 1:(j - 1)]
                    @views yj_thread[:, j:(tpost - 1)] .= y[:, (j + 1):tpost]
                end

                cache = if !ridge && scm
                    _conformal_build_cache(
                        Xj_thread, yj_thread, trt;
                        V = V, fixedeff = fixedeff
                    )
                else
                    nothing
                end
                (Xj_thread, yj_thread, cache)
            end

            grid = collect(
                range(base_att[t0 + j] - 2 * post_sd, stop = base_att[t0 + j] + 2 * post_sd, length = grid_size)
            )

            lo, hi, pv, _ = _compute_permute_ci(
                Xj_eval, yj_eval, trt, grid,
                1, alpha, type, q, ns,
                ridge, scm, lambda;
                init_weights = base_weights,
                holdout_length = holdout_length, min1se = min1se,
                conformal_mode = conformal_mode,
                V = V, fixedeff = fixedeff,
                threaded_permutations = false,
                fit_cache = conf_cache
            )
            lo_out[j] = lo
            hi_out[j] = hi
            pv_out[j] = pv
        end

        ci[1, :] = lo_out
        ci[2, :] = hi_out
        ci[3, :] = pv_out
    else
        ci_weights = base_weights
        final_weights = ci_weights
        for j in 1:tpost
            Xj_eval, yj_eval, conf_cache = if use_pointwise_stats
                cache = _pointwise_conformal_cache(pointwise_stats, j)
                (_placeholder_X(cache), _placeholder_y(cache), cache)
            else
                Xj[:, 1:t0] .= X
                @views Xj[:, t0 + 1] .= y[:, j]
                if tpost > 1
                    if j == 1
                        @views yj[:, 1:(tpost - 1)] .= y[:, 2:tpost]
                    elseif j == tpost
                        @views yj[:, 1:(tpost - 1)] .= y[:, 1:(tpost - 1)]
                    else
                        @views yj[:, 1:(j - 1)] .= y[:, 1:(j - 1)]
                        @views yj[:, j:(tpost - 1)] .= y[:, (j + 1):tpost]
                    end
                else
                    fill!(yj, 1.0)
                end

                cache = if !ridge && scm
                    _conformal_build_cache(
                        Xj, yj, trt;
                        V = V, fixedeff = fixedeff
                    )
                else
                    nothing
                end
                (Xj, yj, cache)
            end

            grid = collect(
                range(base_att[t0 + j] - 2 * post_sd, stop = base_att[t0 + j] + 2 * post_sd, length = grid_size)
            )
            lo, hi, pv, ci_weights = _compute_permute_ci(
                Xj_eval, yj_eval, trt, grid,
                1, alpha, type, q, ns,
                ridge, scm, lambda;
                init_weights = ci_weights,
                holdout_length = holdout_length, min1se = min1se,
                conformal_mode = conformal_mode,
                V = V, fixedeff = fixedeff,
                threaded_permutations = false,
                fit_cache = conf_cache
            )
            ci[1, j] = lo
            ci[2, j] = hi
            ci[3, j] = pv
        end
    end

    null_X = hcat(X, y)
    null_y = ones(n, 1)
    null_cache = if !ridge && scm
        _conformal_build_cache(
            null_X, null_y, trt;
            V = V, fixedeff = fixedeff,
            shift_cols = (t0 + 1):(t0 + tpost)
        )
    else
        nothing
    end

    null_p = _compute_permute_pval(
        null_X, null_y, trt;
        h0 = 0.0,
        post_length = tpost,
        type = type,
        q = q,
        ns = ns,
        ridge = ridge,
        scm = scm,
        lambda = lambda,
        holdout_length = holdout_length,
        min1se = min1se,
        V = V, init_weights = final_weights,
        fixedeff = fixedeff,
        threaded_permutations = use_inner_threads,
        fit_cache = null_cache
    )[1]

    att = vcat(base_att, mean(base_att[(t0 + 1):t_total]))
    y1 = vcat(base_pred, mean(base_pred[(t0 + 1):end]))

    lb_tail = isempty(ci[1, :]) ? fill(NaN, 1) : [minimum(filter(isfinite, ci[1, :]))]
    ub_tail = isempty(ci[2, :]) ? fill(NaN, 1) : [maximum(filter(isfinite, ci[2, :]))]
    lb = vcat(fill(NaN, t0), ci[1, :], lb_tail)
    ub = vcat(fill(NaN, t0), ci[2, :], ub_tail)
    pval = vcat(fill(NaN, t0), ci[3, :], null_p)

    (att = att, lb = lb, ub = ub, p_val = pval, alpha = alpha)
end

function jackknife_plus!(n::Int32, t0::Int32, tpost::Int32,
                        Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble}, trtptr::Ptr{Cdouble},
                        attptr::Ptr{Cdouble}, lbptr::Ptr{Cdouble}, ubptr::Ptr{Cdouble},
                        heldptr::Ptr{Cdouble},
                        alpha::Cdouble, conservativeflag::Cint,
                        ridgeflag::Cint, scmflag::Cint, lambdaptr::Ptr{Cdouble},
                        holdout_length::Cint, min1seflag::Cint,
                        errptr::Ptr{UInt8}, errlen::Cint)::Cint
    try
        if n <= 0 || t0 <= 0 || tpost <= 0
            _set_err!(errptr, errlen, "invalid dimensions: require n > 0, t0 > 0, tpost > 0")
            return ERR_BAD_INPUT
        end

        ni = Int(n)
        t0i = Int(t0)
        tposti = Int(tpost)

        X = unsafe_wrap(Array, Xptr, (ni, t0i); own = false)
        y = unsafe_wrap(Array, yptr, (ni, tposti); own = false)
        trt = unsafe_wrap(Array, trtptr, ni; own = false)

        ridge = ridgeflag != 0
        scm = scmflag != 0
        min1se = min1seflag != 0
        conservative = conservativeflag != 0
        lambda = unsafe_wrap(Array, lambdaptr, 1; own = false)[1]

        if holdout_length <= 0
            _set_err!(errptr, errlen, "holdout_length must be positive")
            return ERR_BAD_INPUT
        end

        out = _jackknife_plus_row!(
            X, y, trt;
            ridge = ridge,
            scm = scm,
            lambda = lambda,
            conservative = conservative,
            alpha = Float64(alpha),
            holdout_length = Int(holdout_length),
            min1se = min1se
        )

        total = t0i + tposti + 1
        unsafe_copyto!(attptr, pointer(out.att), total)
        unsafe_copyto!(lbptr, pointer(out.lb), total)
        unsafe_copyto!(ubptr, pointer(out.ub), total)
        unsafe_copyto!(heldptr, pointer(out.heldout_att), total)

        return ERR_OK
    catch e
        _set_err!(errptr, errlen, sprint(showerror, e))
        return ERR_EXCEPTION
    end
end

function jackknife_unit_std!(n::Int32, t0::Int32, tpost::Int32,
                            Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble}, trtptr::Ptr{Cdouble},
                            attptr::Ptr{Cdouble}, septr::Ptr{Cdouble},
                            ridgeflag::Cint, scmflag::Cint, lambdaptr::Ptr{Cdouble},
                            holdout_length::Cint, min1seflag::Cint,
                            errptr::Ptr{UInt8}, errlen::Cint)::Cint
    try
        if n <= 0 || t0 <= 0 || tpost <= 0
            _set_err!(errptr, errlen, "invalid dimensions: require n > 0, t0 > 0, tpost > 0")
            return ERR_BAD_INPUT
        end

        ni = Int(n)
        t0i = Int(t0)
        tposti = Int(tpost)

        X = unsafe_wrap(Array, Xptr, (ni, t0i); own = false)
        y = unsafe_wrap(Array, yptr, (ni, tposti); own = false)
        trt = unsafe_wrap(Array, trtptr, ni; own = false)

        ridge = ridgeflag != 0
        scm = scmflag != 0
        min1se = min1seflag != 0
        lambda = unsafe_wrap(Array, lambdaptr, 1; own = false)[1]

        if holdout_length <= 0
            _set_err!(errptr, errlen, "holdout_length must be positive")
            return ERR_BAD_INPUT
        end

        out = _jackknife_unit_std!(
            X, y, trt;
            ridge = ridge,
            scm = scm,
            lambda = lambda,
            holdout_length = Int(holdout_length),
            min1se = min1se
        )

        total = t0i + tposti + 1
        unsafe_copyto!(attptr, pointer(out.att), total)
        unsafe_copyto!(septr, pointer(out.se), total)
        return ERR_OK
    catch e
        _set_err!(errptr, errlen, sprint(showerror, e))
        return ERR_EXCEPTION
    end
end

function conformal_inference!(n::Int32, t0::Int32, tpost::Int32,
                             Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble}, trtptr::Ptr{Cdouble},
                             attptr::Ptr{Cdouble}, lbptr::Ptr{Cdouble}, ubptr::Ptr{Cdouble},
                             pvalptr::Ptr{Cdouble},
                             alpha::Cdouble, typeflag::Cint,
                             q::Cdouble, ns::Cint, grid_size::Cint,
                             ridgeflag::Cint, scmflag::Cint, lambdaptr::Ptr{Cdouble},
                             holdout_length::Cint, min1seflag::Cint,
                             errptr::Ptr{UInt8}, errlen::Cint)::Cint
    try
        if n <= 0 || t0 <= 0 || tpost <= 0
            _set_err!(errptr, errlen, "invalid dimensions: require n > 0, t0 > 0, tpost > 0")
            return ERR_BAD_INPUT
        end

        ni = Int(n)
        t0i = Int(t0)
        tposti = Int(tpost)

        X = unsafe_wrap(Array, Xptr, (ni, t0i); own = false)
        y = unsafe_wrap(Array, yptr, (ni, tposti); own = false)
        trt = unsafe_wrap(Array, trtptr, ni; own = false)

        ridge = ridgeflag != 0
        scm = scmflag != 0
        min1se = min1seflag != 0
        lambda = unsafe_wrap(Array, lambdaptr, 1; own = false)[1]
        tpe = Int(typeflag)

        if ns <= 0
            _set_err!(errptr, errlen, "ns must be positive")
            return ERR_BAD_INPUT
        end
        if grid_size <= 0
            _set_err!(errptr, errlen, "grid_size must be positive")
            return ERR_BAD_INPUT
        end
        if holdout_length <= 0
            _set_err!(errptr, errlen, "holdout_length must be positive")
            return ERR_BAD_INPUT
        end

        out = _conformal!(
            X, y, trt;
            ridge = ridge,
            scm = scm,
            lambda = lambda,
            alpha = Float64(alpha),
            type = tpe,
            q = Float64(q),
            ns = Int(ns),
            grid_size = Int(grid_size),
            holdout_length = Int(holdout_length),
            min1se = min1se
        )

        total = t0i + tposti + 1
        unsafe_copyto!(attptr, pointer(out.att), total)
        unsafe_copyto!(lbptr, pointer(out.lb), total)
        unsafe_copyto!(ubptr, pointer(out.ub), total)
        unsafe_copyto!(pvalptr, pointer(out.p_val), total)
        return ERR_OK
    catch e
        _set_err!(errptr, errlen, sprint(showerror, e))
        return ERR_EXCEPTION
    end
end

function fit_synth_weights!(n0::Int32, t0::Int32,
                            X0ptr::Ptr{Cdouble}, x1ptr::Ptr{Cdouble},
                            weightsptr::Ptr{Cdouble},
                            errptr::Ptr{UInt8}, errlen::Int32)::Cint
    try
        if n0 <= 0 || t0 <= 0
            _set_err!(errptr, errlen, "invalid dimensions: require n0 > 0 and t0 > 0")
            return ERR_BAD_INPUT
        end

        donors = unsafe_wrap(Array, X0ptr, (Int(n0), Int(t0)); own = false)
        target = unsafe_wrap(Array, x1ptr, Int(t0); own = false)

        weights = _solve_simplex_qp(donors, target)
        unsafe_copyto!(weightsptr, pointer(weights), Int(n0))

        return ERR_OK
    catch e
        _set_err!(errptr, errlen, sprint(showerror, e))
        return ERR_EXCEPTION
    end
end

function fit_ridge_augsynth_inner!(n0::Int32, t0::Int32,
                                   Xcptr::Ptr{Cdouble}, x1ptr::Ptr{Cdouble},
                                   ridgeflag::Int32, scmflag::Int32,
                                   selectlambdaflag::Int32,
                                   nlambda::Int32, lambdasptr::Ptr{Cdouble},
                                   holdout_length::Int32, min1seflag::Int32,
                                   weightsptr::Ptr{Cdouble}, synptr::Ptr{Cdouble},
                                   lambdaptr::Ptr{Cdouble},
                                   errorsptr::Ptr{Cdouble}, errorsseptr::Ptr{Cdouble},
                                   errptr::Ptr{UInt8}, errlen::Int32)::Cint
    try
        if n0 <= 0 || t0 <= 0
            _set_err!(errptr, errlen, "invalid dimensions: require n0 > 0 and t0 > 0")
            return ERR_BAD_INPUT
        end
        if holdout_length <= 0
            _set_err!(errptr, errlen, "holdout_length must be positive")
            return ERR_BAD_INPUT
        end

        ni = Int(n0)
        ti = Int(t0)
        li = Int(nlambda)

        Xc = unsafe_wrap(Array, Xcptr, (ni, ti); own = false)
        x1 = unsafe_wrap(Array, x1ptr, ti; own = false)
        lambdas = li > 0 ? collect(unsafe_wrap(Array, lambdasptr, li; own = false)) : Float64[]

        ridge = ridgeflag != 0
        scm = scmflag != 0
        select_lambda = selectlambdaflag != 0
        min1se = min1seflag != 0

        if ridge && li <= 0
            _set_err!(errptr, errlen, "ridge requires at least one lambda")
            return ERR_BAD_INPUT
        end
        if select_lambda && li <= 0
            _set_err!(errptr, errlen, "lambda tuning requires a non-empty lambda grid")
            return ERR_BAD_INPUT
        end
        if ridge && any(!isfinite(λ) || λ < 0.0 for λ in lambdas)
            _set_err!(errptr, errlen, "lambdas must be finite and non-negative")
            return ERR_BAD_INPUT
        end

        syn = scm ? _solve_simplex_qp(Xc, x1) : fill(1.0 / ni, ni)
        weights = copy(syn)
        lambda_errors = li > 0 ? fill(NaN, li) : Float64[]
        lambda_errors_se = li > 0 ? fill(NaN, li) : Float64[]
        λ = NaN

        if ridge
            if select_lambda
                lambda_errors, lambda_errors_se = _lambda_errors(
                    Xc, x1, lambdas, Int(holdout_length), scm
                )
                λ = _choose_lambda(lambdas, lambda_errors, lambda_errors_se, min1se)
            else
                λ = lambdas[1]
            end
            weights .+= _ridge_adjustment(Xc, x1, syn, λ)
        end

        unsafe_copyto!(weightsptr, pointer(weights), ni)
        unsafe_copyto!(synptr, pointer(syn), ni)
        unsafe_store!(lambdaptr, λ)
        if li > 0
            unsafe_copyto!(errorsptr, pointer(lambda_errors), li)
            unsafe_copyto!(errorsseptr, pointer(lambda_errors_se), li)
        end

        return ERR_OK
    catch e
        _set_err!(errptr, errlen, sprint(showerror, e))
        return ERR_EXCEPTION
    end
end

function _set_err!(errptr::Ptr{UInt8}, errlen::Int32, msg::AbstractString)
    if errlen <= 0
        return
    end
    bytes = codeunits(msg)
    ncopy = min(length(bytes), Int(errlen) - 1)
    unsafe_copyto!(errptr, pointer(bytes), ncopy)
    unsafe_store!(errptr + ncopy, 0x00)
end

Base.@ccallable function fit_ols_dense(n::Cint, p::Cint,
                                       Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble},
                                       coefptr::Ptr{Cdouble}, sigma2ptr::Ptr{Cdouble},
                                       dfresidptr::Ptr{Cint}, rssptr::Ptr{Cdouble},
                                       errptr::Ptr{UInt8}, errlen::Cint)::Cint
    return fit_ols_dense!(n, p, Xptr, yptr, coefptr, sigma2ptr, dfresidptr, rssptr, errptr, errlen)
end

Base.@ccallable function fit_ridge_loocv_dense(n::Cint, p::Cint,
                                               Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble},
                                               nlambda::Cint, lambdasptr::Ptr{Cdouble},
                                               coefptr::Ptr{Cdouble},
                                               bestlambdaptr::Ptr{Cdouble}, bestmseptr::Ptr{Cdouble},
                                               errptr::Ptr{UInt8}, errlen::Cint)::Cint
    return fit_ridge_loocv_dense!(n, p, Xptr, yptr, nlambda, lambdasptr,
                                  coefptr, bestlambdaptr, bestmseptr, errptr, errlen)
end

Base.@ccallable function fit_synth_weights(n0::Cint, t0::Cint,
                                           X0ptr::Ptr{Cdouble}, x1ptr::Ptr{Cdouble},
                                           weightsptr::Ptr{Cdouble},
                                           errptr::Ptr{UInt8}, errlen::Cint)::Cint
    return fit_synth_weights!(n0, t0, X0ptr, x1ptr, weightsptr, errptr, errlen)
end

Base.@ccallable function fit_ridge_augsynth_inner(n0::Cint, t0::Cint,
                                                  Xcptr::Ptr{Cdouble}, x1ptr::Ptr{Cdouble},
                                                  ridgeflag::Cint, scmflag::Cint,
                                                  selectlambdaflag::Cint,
                                                  nlambda::Cint, lambdasptr::Ptr{Cdouble},
                                                  holdout_length::Cint, min1seflag::Cint,
                                                  weightsptr::Ptr{Cdouble}, synptr::Ptr{Cdouble},
                                                  lambdaptr::Ptr{Cdouble},
                                                  errorsptr::Ptr{Cdouble}, errorsseptr::Ptr{Cdouble},
                                                  errptr::Ptr{UInt8}, errlen::Cint)::Cint
    return fit_ridge_augsynth_inner!(
        n0, t0, Xcptr, x1ptr, ridgeflag, scmflag, selectlambdaflag,
        nlambda, lambdasptr, holdout_length, min1seflag, weightsptr, synptr,
        lambdaptr, errorsptr, errorsseptr, errptr, errlen
    )
end

Base.@ccallable function jackknife_plus(n::Cint, t0::Cint, tpost::Cint,
                                       Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble}, trtptr::Ptr{Cdouble},
                                       attptr::Ptr{Cdouble}, lbptr::Ptr{Cdouble}, ubptr::Ptr{Cdouble},
                                       heldptr::Ptr{Cdouble},
                                       alpha::Cdouble, conservativeflag::Cint,
                                       ridgeflag::Cint, scmflag::Cint, lambdaptr::Ptr{Cdouble},
                                       holdout_length::Cint, min1seflag::Cint,
                                       errptr::Ptr{UInt8}, errlen::Cint)::Cint
    return jackknife_plus!(
        n, t0, tpost,
        Xptr, yptr, trtptr,
        attptr, lbptr, ubptr,
        heldptr,
        alpha, conservativeflag,
        ridgeflag, scmflag, lambdaptr,
        holdout_length, min1seflag,
        errptr, errlen
    )
end

Base.@ccallable function jackknife_unit_std(n::Cint, t0::Cint, tpost::Cint,
                                           Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble}, trtptr::Ptr{Cdouble},
                                           attptr::Ptr{Cdouble}, septr::Ptr{Cdouble},
                                           ridgeflag::Cint, scmflag::Cint, lambdaptr::Ptr{Cdouble},
                                           holdout_length::Cint, min1seflag::Cint,
                                           errptr::Ptr{UInt8}, errlen::Cint)::Cint
    return jackknife_unit_std!(
        n, t0, tpost,
        Xptr, yptr, trtptr,
        attptr, septr,
        ridgeflag, scmflag, lambdaptr,
        holdout_length, min1seflag,
        errptr, errlen
    )
end

Base.@ccallable function backend_thread_count()::Cint
    return Cint(Threads.nthreads())
end

Base.@ccallable function conformal_inference(n::Cint, t0::Cint, tpost::Cint,
                                            Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble}, trtptr::Ptr{Cdouble},
                                            attptr::Ptr{Cdouble}, lbptr::Ptr{Cdouble}, ubptr::Ptr{Cdouble},
                                            pvalptr::Ptr{Cdouble},
                                            alpha::Cdouble, typeflag::Cint,
                                            q::Cdouble, ns::Cint, grid_size::Cint,
                                            ridgeflag::Cint, scmflag::Cint, lambdaptr::Ptr{Cdouble},
                                            holdout_length::Cint, min1seflag::Cint,
                                            errptr::Ptr{UInt8}, errlen::Cint)::Cint
    return conformal_inference!(
        n, t0, tpost,
        Xptr, yptr, trtptr,
        attptr, lbptr, ubptr,
        pvalptr,
        alpha, typeflag,
        q, ns, grid_size,
        ridgeflag, scmflag, lambdaptr,
        holdout_length, min1seflag,
        errptr, errlen
    )
end

const INFERENCE_NONE = 0
const INFERENCE_JACKKNIFE = 1
const INFERENCE_JACKKNIFE_PLUS = 2
const INFERENCE_CONFORMAL = 3

Base.@ccallable function augsynth_inference(n::Cint, t0::Cint, tpost::Cint,
                                           Xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble}, trtptr::Ptr{Cdouble},
                                           attptr::Ptr{Cdouble}, lbptr::Ptr{Cdouble}, ubptr::Ptr{Cdouble},
                                           septr::Ptr{Cdouble}, heldptr::Ptr{Cdouble}, pvalptr::Ptr{Cdouble},
                                           inf_type::Cint,
                                           alpha::Cdouble, conservativeflag::Cint,
                                           typeflag::Cint, q::Cdouble, ns::Cint, grid_size::Cint, conformal_modeflag::Cint,
                                           ridgeflag::Cint, scmflag::Cint, lambdaptr::Ptr{Cdouble},
                                           holdout_length::Cint, min1seflag::Cint,
                                           errptr::Ptr{UInt8}, errlen::Cint)::Cint
    try
        if n <= 0 || t0 <= 0 || tpost <= 0
            _set_err!(errptr, errlen, "invalid dimensions: require n > 0, t0 > 0, tpost > 0")
            return ERR_BAD_INPUT
        end
        if inf_type < INFERENCE_NONE || inf_type > INFERENCE_CONFORMAL
            _set_err!(errptr, errlen, "invalid inf_type; expected one of 1 (jackknife), 2 (jackknife+), 3 (conformal)")
            return ERR_BAD_INPUT
        end
        if conformal_modeflag < CONFORMAL_MODE_FAST || conformal_modeflag > CONFORMAL_MODE_REFERENCE
            _set_err!(errptr, errlen, "invalid conformal_mode; expected 0 (fast) or 1 (reference)")
            return ERR_BAD_INPUT
        end

        ni = Int(n)
        t0i = Int(t0)
        tposti = Int(tpost)

        X = unsafe_wrap(Array, Xptr, (ni, t0i); own = false)
        y = unsafe_wrap(Array, yptr, (ni, tposti); own = false)
        trt = unsafe_wrap(Array, trtptr, ni; own = false)

        ridge = ridgeflag != 0
        scm = scmflag != 0
        min1se = min1seflag != 0
        lambda = unsafe_wrap(Array, lambdaptr, 1; own = false)[1]

        if holdout_length <= 0
            _set_err!(errptr, errlen, "holdout_length must be positive")
            return ERR_BAD_INPUT
        end

        total = t0i + tposti + 1
        fill!(unsafe_wrap(Array, lbptr, total; own = false), NaN)
        fill!(unsafe_wrap(Array, ubptr, total; own = false), NaN)
        fill!(unsafe_wrap(Array, septr, total; own = false), NaN)
        fill!(unsafe_wrap(Array, heldptr, total; own = false), NaN)
        fill!(unsafe_wrap(Array, pvalptr, total; own = false), NaN)

        if inf_type == INFERENCE_JACKKNIFE
            out = _jackknife_unit_std!(
                X, y, trt;
                ridge = ridge,
                scm = scm,
                lambda = lambda,
                holdout_length = Int(holdout_length),
                min1se = min1se
            )
            unsafe_copyto!(attptr, pointer(out.att), total)
            unsafe_copyto!(septr, pointer(out.se), total)
            return ERR_OK
        elseif inf_type == INFERENCE_JACKKNIFE_PLUS
            out = _jackknife_plus_row!(
                X, y, trt;
                ridge = ridge,
                scm = scm,
                lambda = lambda,
                conservative = conservativeflag != 0,
                alpha = Float64(alpha),
                holdout_length = Int(holdout_length),
                min1se = min1se
            )
            unsafe_copyto!(attptr, pointer(out.att), total)
            unsafe_copyto!(lbptr, pointer(out.lb), total)
            unsafe_copyto!(ubptr, pointer(out.ub), total)
            unsafe_copyto!(heldptr, pointer(out.heldout_att), total)
            return ERR_OK
        elseif inf_type == INFERENCE_CONFORMAL
            tpe = Int(typeflag)
            out = _conformal!(
                X, y, trt;
                ridge = ridge,
                scm = scm,
                lambda = lambda,
                alpha = Float64(alpha),
                type = tpe,
                q = Float64(q),
                ns = Int(ns),
                grid_size = Int(grid_size),
                conformal_mode = Int(conformal_modeflag),
                holdout_length = Int(holdout_length),
                min1se = min1se
            )
            unsafe_copyto!(attptr, pointer(out.att), total)
            unsafe_copyto!(lbptr, pointer(out.lb), total)
            unsafe_copyto!(ubptr, pointer(out.ub), total)
            unsafe_copyto!(pvalptr, pointer(out.p_val), total)
            return ERR_OK
        else
            _set_err!(errptr, errlen, "unsupported inf_type")
            return ERR_BAD_INPUT
        end
    catch e
        _set_err!(errptr, errlen, sprint(showerror, e))
        return ERR_EXCEPTION
    end
end

end # module
