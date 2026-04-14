using Random
using StatlibBackend

Random.seed!(20260412)

let
    n = 24
    p = 5
    X = randn(n, p)
    β = collect(1.0:p)
    y = X * β .+ 0.1 .* randn(n)

    coef = zeros(p)
    sigma2 = Ref{Float64}(0.0)
    df = Ref{Cint}(0)
    rss = Ref{Float64}(0.0)
    err = zeros(UInt8, 512)

    StatlibBackend.fit_ols_dense!(
        Cint(n), Cint(p),
        pointer(X), pointer(y),
        pointer(coef),
        Base.unsafe_convert(Ptr{Float64}, sigma2),
        Base.unsafe_convert(Ptr{Cint}, df),
        Base.unsafe_convert(Ptr{Float64}, rss),
        pointer(err), Cint(length(err))
    )

    λgrid = 10.0 .^ range(-6, 1; length = 16)
    coef_ridge = zeros(p)
    bestλ = Ref{Float64}(0.0)
    bestmse = Ref{Float64}(0.0)
    StatlibBackend.fit_ridge_loocv_dense!(
        Cint(n), Cint(p),
        pointer(X), pointer(y),
        Cint(length(λgrid)), pointer(λgrid),
        pointer(coef_ridge),
        Base.unsafe_convert(Ptr{Float64}, bestλ),
        Base.unsafe_convert(Ptr{Float64}, bestmse),
        pointer(err), Cint(length(err))
    )
end

let
    n = 30
    t0 = 8
    tpost = 4
    X = randn(n, t0)
    y = randn(n, tpost)
    trt = zeros(Float64, n)
    trt[(n - 1):n] .= 1.0
    y .+= reshape(collect(1:tpost), 1, tpost) .* trt

    total = t0 + tpost + 1
    att = zeros(Float64, total)
    lb = zeros(Float64, total)
    ub = zeros(Float64, total)
    se = zeros(Float64, total)
    held = zeros(Float64, total)
    pval = zeros(Float64, total)
    err = zeros(UInt8, 512)
    λ = [0.0]

    StatlibBackend.augsynth_inference(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att), pointer(lb), pointer(ub),
        pointer(se), pointer(held), pointer(pval),
        Cint(1), 0.1, Cint(0),
        Cint(0), 1.0, Cint(32), Cint(21), Cint(0),
        Cint(0), Cint(1), pointer(λ),
        Cint(1), Cint(1),
        pointer(err), Cint(length(err))
    )

    StatlibBackend.augsynth_inference(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att), pointer(lb), pointer(ub),
        pointer(se), pointer(held), pointer(pval),
        Cint(2), 0.1, Cint(0),
        Cint(0), 1.0, Cint(32), Cint(21), Cint(0),
        Cint(0), Cint(1), pointer(λ),
        Cint(1), Cint(1),
        pointer(err), Cint(length(err))
    )

    StatlibBackend.augsynth_inference(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att), pointer(lb), pointer(ub),
        pointer(se), pointer(held), pointer(pval),
        Cint(3), 0.1, Cint(0),
        Cint(0), 1.0, Cint(32), Cint(21), Cint(0),
        Cint(0), Cint(1), pointer(λ),
        Cint(1), Cint(1),
        pointer(err), Cint(length(err))
    )
end

nothing
