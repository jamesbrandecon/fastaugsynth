module StatlibBackend

using LinearAlgebra

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

end # module
