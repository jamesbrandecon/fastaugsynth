using Test
using LinearAlgebra
using StatlibBackend

@testset "OLS ABI" begin
    n, p = 100, 3
    X = randn(n, p)
    βtrue = [1.0, -2.0, 0.5]
    y = X * βtrue .+ 0.01 .* randn(n)

    coef = zeros(p)
    sigma2 = Ref{Float64}(0.0)
    df = Ref{Cint}(0)
    rss = Ref{Float64}(0.0)
    err = zeros(UInt8, 512)

    status = StatlibBackend.fit_ols_dense!(n, p, pointer(X), pointer(y), pointer(coef),
        Base.unsafe_convert(Ptr{Float64}, sigma2), Base.unsafe_convert(Ptr{Cint}, df),
        Base.unsafe_convert(Ptr{Float64}, rss), pointer(err), 512)

    @test status == 0
    @test maximum(abs.(coef .- (X \ y))) < 1e-8
    @test df[] == n - p
    @test rss[] > 0
end

@testset "Ridge LOOCV ABI" begin
    n, p = 120, 12
    X = randn(n, p)
    βtrue = vcat(fill(2.0, 4), fill(0.0, p - 4))
    y = X * βtrue .+ 0.15 .* randn(n)
    λgrid = 10.0 .^ range(-6, 2; length = 48)

    coef = zeros(p)
    bestλ = Ref{Float64}(0.0)
    bestmse = Ref{Float64}(0.0)
    err = zeros(UInt8, 512)

    status = StatlibBackend.fit_ridge_loocv_dense!(n, p,
        pointer(X), pointer(y), length(λgrid), pointer(λgrid),
        pointer(coef), Base.unsafe_convert(Ptr{Float64}, bestλ),
        Base.unsafe_convert(Ptr{Float64}, bestmse), pointer(err), 512)

    @test status == 0
    @test bestλ[] in λgrid
    @test isfinite(bestmse[])

    # Validate selected model predicts well on a holdout split.
    ntr = 100
    Xtr, ytr = X[1:ntr, :], y[1:ntr]
    Xte, yte = X[(ntr + 1):end, :], y[(ntr + 1):end]
    βridge = (Xtr' * Xtr + bestλ[] * I) \ (Xtr' * ytr)
    rmse = sqrt(mean((yte - Xte * βridge) .^ 2))
    @test rmse < 1.0
end
