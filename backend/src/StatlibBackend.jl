module StatlibBackend

using LinearAlgebra
using Statistics

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
        for j in eachindex(lambdas)
            λ = lambdas[j]
            aug = syn .+ _ridge_adjustment(Xfit, xfit, syn, λ)
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

end # module
