using Test
using LinearAlgebra
using Statistics
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

    status = StatlibBackend.fit_ols_dense!(Cint(n), Cint(p), pointer(X), pointer(y), pointer(coef),
        Base.unsafe_convert(Ptr{Float64}, sigma2), Base.unsafe_convert(Ptr{Cint}, df),
        Base.unsafe_convert(Ptr{Float64}, rss), pointer(err), Cint(length(err)))

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

    status = StatlibBackend.fit_ridge_loocv_dense!(Cint(n), Cint(p),
        pointer(X), pointer(y), Cint(length(λgrid)), pointer(λgrid),
        pointer(coef), Base.unsafe_convert(Ptr{Float64}, bestλ),
        Base.unsafe_convert(Ptr{Float64}, bestmse), pointer(err), Cint(length(err)))

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

@testset "SCM simplex ABI" begin
    donors = [
        1.0 0.0
        0.0 1.0
    ]
    target = [0.25, 0.75]
    weights = zeros(size(donors, 1))
    err = zeros(UInt8, 512)

    status = StatlibBackend.fit_synth_weights!(
        Cint(size(donors, 1)),
        Cint(size(donors, 2)),
        pointer(donors),
        pointer(target),
        pointer(weights),
        pointer(err),
        Cint(length(err))
    )

    @test status == 0
    @test maximum(abs.(weights .- [0.25, 0.75])) < 1e-8
end

@testset "Ridge ASCM ABI" begin
    Xc = [
        1.0 2.0 3.0 4.0
        1.5 2.5 3.5 4.5
        2.0 3.0 4.0 5.0
    ]
    x1 = [1.25, 2.25, 3.25, 4.25]
    λgrid = [0.01, 0.1, 1.0, 10.0]

    weights = zeros(size(Xc, 1))
    syn = zeros(size(Xc, 1))
    bestλ = Ref{Float64}(NaN)
    errs = zeros(length(λgrid))
    errs_se = zeros(length(λgrid))
    err = zeros(UInt8, 512)

    status = StatlibBackend.fit_ridge_augsynth_inner!(
        Cint(size(Xc, 1)),
        Cint(size(Xc, 2)),
        pointer(Xc),
        pointer(x1),
        Cint(1),
        Cint(1),
        Cint(1),
        Cint(length(λgrid)),
        pointer(λgrid),
        Cint(1),
        Cint(1),
        pointer(weights),
        pointer(syn),
        Base.unsafe_convert(Ptr{Float64}, bestλ),
        pointer(errs),
        pointer(errs_se),
        pointer(err),
        Cint(length(err))
    )

    @test status == 0
    @test bestλ[] in λgrid

    ref_syn = StatlibBackend._solve_simplex_qp(Xc, x1)
    ref_errs, ref_errs_se = StatlibBackend._lambda_errors(Xc, x1, λgrid, 1, true)
    ref_λ = StatlibBackend._choose_lambda(λgrid, ref_errs, ref_errs_se, true)
    ref_weights = ref_syn .+ StatlibBackend._ridge_adjustment(Xc, x1, ref_syn, ref_λ)

    @test maximum(abs.(syn .- ref_syn)) < 1e-8
    @test maximum(abs.(errs .- ref_errs)) < 1e-8
    @test maximum(abs.(errs_se .- ref_errs_se)) < 1e-8
    @test bestλ[] == ref_λ
    @test maximum(abs.(weights .- ref_weights)) < 1e-8
end
