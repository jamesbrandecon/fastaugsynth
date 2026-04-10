module StatlibBackend

using LinearAlgebra
using Statistics
using Random

const ERR_OK = 0
const ERR_BAD_INPUT = 1
const ERR_LINALG = 2
const ERR_EXCEPTION = 3

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

function _project_simplex(v::AbstractVector{Float64})
    n = length(v)
    u = sort(collect(v); rev = true)
    cssv = cumsum(u) .- 1.0
    ρ = 0
    @inbounds for j in 1:n
        if u[j] > cssv[j] / j
            ρ = j
        end
    end
    if ρ == 0
        return fill(1.0 / n, n)
    end
    θ = cssv[ρ] / ρ
    w = similar(collect(v))
    @inbounds @simd for j in eachindex(w)
        w[j] = max(v[j] - θ, 0.0)
    end
    s = sum(w)
    if !isfinite(s) || s <= 0.0
        return fill(1.0 / n, n)
    end
    w ./= s
    return w
end

function _solve_simplex_qp(donors::AbstractMatrix{Float64},
                           target::AbstractVector{Float64};
                           tol::Float64 = 1e-10,
                           maxiter::Int = 50_000)
    n0 = size(donors, 1)
    if n0 == 1
        return ones(Float64, 1)
    end

    P = Matrix(donors * transpose(donors))
    q = -(donors * target)

    diagvals = diag(P)
    start = argmin(0.5 .* diagvals .+ q)
    w = zeros(Float64, n0)
    w[start] = 1.0
    z = copy(w)
    t = 1.0

    L = max(opnorm(Symmetric(P), 2), 1.0)
    step = 1.0 / L
    last_obj = Inf

    for _ in 1:maxiter
        g = P * z .+ q
        w_new = _project_simplex(z .- step .* g)
        obj = 0.5 * dot(w_new, P * w_new) + dot(q, w_new)

        if !isfinite(obj)
            error("non-finite objective in simplex solve")
        end

        if obj > last_obj + 1e-12
            z = copy(w)
            t = 1.0
            continue
        end

        if maximum(abs.(w_new .- w)) <= tol
            return w_new
        end

        t_new = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t))
        z = w_new .+ ((t - 1.0) / t_new) .* (w_new .- w)
        w = w_new
        t = t_new
        last_obj = obj
    end

    error("simplex QP did not converge")
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

function _fit_augsynth_single!(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                               trt::AbstractVector{Float64};
                               ridge::Bool,
                               scm::Bool,
                               lambda::Float64,
                               holdout_length::Int = 1,
                               min1se::Bool = true,
                               V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                               fixedeff::Bool = false)
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

    Xc = copy(X)
    yc = copy(y)
    if fixedeff
        means = vec(mean(hcat(Xc, yc); dims = 2))
        Xc .-= means
        yc .-= means
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

    syn = scm ? _solve_simplex_qp(X0, X1) : fill(1.0 / n0, n0)
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

    base_series = weights' * (hcat(Xc, y)[control, :])
    mhat = repeat(base_series, size(X, 1), 1)

    return (
        weights = reshape(weights, :, 1),
        syn = reshape(syn, :, 1),
        lambda = λ,
        mhat = mhat,
        l2_imbalance = l2_imbalance,
        scaled_l2_imbalance = scaled_l2,
        t0 = size(X, 2),
        tpost = size(y, 2)
    )
end

function _predict_counterfactual(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                                trt::AbstractVector{Float64},
                                fit::NamedTuple)
    if fit.t0 < 1 || fit.tpost < 1
        error("fit must include pre- and post-treatment periods")
    end

    comb = hcat(X, y)
    treated = trt .> 0.5
    control = .!treated
    m1 = vec(mean(fit.mhat[treated, :], dims = 1))
    resid = comb[control, :] .- fit.mhat[control, :]
    return vec(m1 .+ vec(transpose(resid) * fit.weights))
end

function _predict_att(X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64},
                     trt::AbstractVector{Float64}, fit::NamedTuple)
    comb = hcat(X, y)
    treated = trt .> 0.5
    y0 = vec(_predict_counterfactual(X, y, trt, fit))
    att = vec(mean(comb[treated, :], dims = 1) .- y0')
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
                                   holdout_length::Int = 1,
                                   min1se::Bool = true,
                                   V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                                   fixedeff::Bool = false)
    fit = _fit_augsynth_single!(X, y, trt;
        ridge = ridge, scm = scm, lambda = lambda,
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
    base = _jackknife_fit_stat_matrix(
        X, y, trt;
        ridge = ridge, scm = scm, lambda = lambda,
        holdout_length = holdout_length, min1se = min1se,
        V = V, fixedeff = fixedeff
    )
    base_att = base.att
    n, t0 = size(X)
    tpost = size(y, 2)
    t_final = t0 + tpost

    jack_ests = Array{Float64}(undef, 4, tpost + 1, t0)
    held_out = zeros(t0)

    for i in 1:t0
        keep_cols = trues(t0)
        keep_cols[i] = false
        subX = X[:, keep_cols]
        suby = hcat(X[:, i], y)

        sub = _fit_augsynth_single!(
            subX, suby, trt;
            ridge = ridge, scm = scm, lambda = lambda,
            holdout_length = holdout_length, min1se = min1se,
            V = V, fixedeff = fixedeff
        )
        counter = _predict_counterfactual(subX, suby, trt, sub)

        est = counter[(t0 + 1):end]
        est_mean = mean(est)
        est = vcat(est, est_mean)

        held = mean(X[trt .> 0.5, i]) - counter[t0]
        held_out[i] = held
        jack_ests[1, :, i] .= est .+ abs(held)
        jack_ests[2, :, i] .= est .- abs(held)
        jack_ests[3, :, i] .= est .+ held
        jack_ests[4, :, i] .= est
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
    n, t0 = size(X)
    tpost = size(y, 2)
    t_final = t0 + tpost

    base = _jackknife_fit_stat_matrix(
        X, y, trt;
        ridge = ridge, scm = scm, lambda = lambda,
        holdout_length = holdout_length, min1se = min1se,
        V = V, fixedeff = fixedeff
    )
    att = base.att

    control = trt .<= 0.5
    nnz = copy(control)
    nnz[.!control] .= true
    control_weights = vec(base.fit.weights)
    nnz[control] .= (round.(control_weights, digits = 3) .!= 0.0)
    nnz[trt .> 0.5] .= true
    sel = findall(nnz)
    if isempty(sel)
        error("No units selected for jackknife")
    end

    all_ests = Matrix{Float64}(undef, tpost + 1, length(sel))
    for (k, i) in pairs(sel)
        keep = trues(n)
        keep[i] = false
        sub = _fit_augsynth_single!(
            X[keep, :], y[keep, :], trt[keep];
            ridge = ridge, scm = scm, lambda = lambda,
            holdout_length = holdout_length, min1se = min1se,
            V = V, fixedeff = fixedeff
        )
        att_sub = _predict_att(X[keep, :], y[keep, :], trt[keep], sub)
        est = att_sub[(t0 + 1):end]
        all_ests[:, k] .= vcat(est, mean(est))
    end

    avg = vec(mean(all_ests; dims = 2))
    se = sqrt.((n - 1) / n .* vec(sum((all_ests .- avg).^2; dims = 2)))
    (att = vcat(att, mean(att[(t0 + 1):end])), se = vcat(fill(NaN, t0), se))
end

function _jackknife_permute_stat(x::AbstractVector{Float64}, q::Float64)
    (sum(abs.(x) .^ q) / sqrt(length(x))) ^ (1 / q)
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
                                    fixedeff::Bool = false)
    ncol_x = size(X, 2)
    t0 = ncol_x - post_length
    if t0 <= 0
        error("post_length must be smaller than number of columns in X")
    end
    tpost = t0 + post_length

    X2 = copy(X)
    treated = trt .> 0.5
    X2[treated, (t0 + 1):tpost] .-= h0

    fit = _fit_augsynth_single!(
        X2, y, trt;
        ridge = ridge, scm = scm, lambda = lambda,
        holdout_length = holdout_length, min1se = min1se,
        V = V, fixedeff = fixedeff
    )
    resids = _predict_att(X2, y, trt, fit)
    obs = resids[1:tpost]
    stat = _jackknife_permute_stat(obs[(t0 + 1):tpost], q)

    if type == 0
        out = zeros(ns)
        for i in 1:ns
            reorder = shuffle!(copy(obs))
            out[i] = _jackknife_permute_stat(reorder[(t0 + 1):tpost], q)
        end
    else
        out = zeros(tpost)
        idx = collect(1:tpost)
        for i in 1:tpost
            reorder = obs[mod1.(idx .+ (i - 1), tpost)]
            out[i] = _jackknife_permute_stat(reorder[(t0 + 1):tpost], q)
        end
    end

    (resids = obs, test_stats = out, stat = stat)
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
                              fixedeff::Bool = false)
    t0 = size(X, 2) - post_length
    tpost = t0 + post_length
    out = _compute_permute_test_stats(
        X, y, trt;
        h0 = h0, post_length = post_length,
        type = type, q = q, ns = ns,
        ridge = ridge, scm = scm,
        lambda = lambda, holdout_length = holdout_length,
        min1se = min1se, V = V, fixedeff = fixedeff
    )
    mean(out.stat .<= out.test_stats)
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
                            holdout_length::Int = 1,
                            min1se::Bool = true,
                            V::Union{AbstractMatrix{Float64}, Nothing} = nothing,
                            fixedeff::Bool = false)
    extended = vcat(grid, 0.0)
    ps = zeros(Float64, length(extended))
    for (k, h0) in pairs(extended)
        ps[k] = _compute_permute_pval(
            X, y, trt;
            h0 = h0, post_length = post_length,
            type = type, q = q, ns = ns,
            ridge = ridge, scm = scm, lambda = lambda,
            holdout_length = holdout_length, min1se = min1se,
            V = V, fixedeff = fixedeff
        )
    end

    valid = findall(x -> x >= alpha, ps)
    if isempty(valid)
        return (NaN, NaN, ps[findfirst(==(0.0), extended)] |> x -> (x === nothing) ? NaN : ps[x])
    end
    p_zero = findfirst(==(0.0), extended)
    return (minimum(extended[valid]), maximum(extended[valid]), p_zero === nothing ? NaN : ps[p_zero])
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
    for j in 1:tpost
        Xj = hcat(X, y[:, j:j])
        if tpost > 1
            keep = trues(tpost)
            keep[j] = false
            yj = y[:, keep]
        else
            yj = ones(n, 1)
        end
        grid = collect(
            range(base_att[t0 + j] - 2 * post_sd, stop = base_att[t0 + j] + 2 * post_sd, length = grid_size)
        )
        lo, hi, pv = _compute_permute_ci(
            Xj, yj, trt, grid,
            1, alpha, type, q, ns,
            ridge, scm, lambda,
            holdout_length, min1se, V, fixedeff
        )
        ci[1, j] = lo
        ci[2, j] = hi
        ci[3, j] = pv
    end

    null_p = _compute_permute_pval(
        hcat(X, y), ones(n, 1), trt;
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
        V = V,
        fixedeff = fixedeff
    )

    att = vcat(base_att, mean(base_att[(t0 + 1):t_total]))
    y1 = vcat(_predict_counterfactual(X, y, trt, base), mean(_predict_counterfactual(X, y, trt, base)[(t0 + 1):end]))

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

end # module
