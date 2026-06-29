using Test
using LinearAlgebra
using Statistics
using Random
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

@testset "Active-set simplex warm path" begin
    rng = MersenneTwister(20260412)
    donors = randn(rng, 6, 6)
    target = randn(rng, 6)
    P = donors * transpose(donors)
    q = -(donors * target)

    w_ref = StatlibBackend._solve_simplex_qp_cached(P, q)
    w_fast = StatlibBackend._solve_simplex_qp_warm(P, q)

    @test abs(sum(w_fast) - 1.0) < 1e-8
    @test minimum(w_fast) >= -1e-8
    @test maximum(abs.(w_fast .- w_ref)) < 1e-6

    q_shift = q .+ 0.2 .* randn(rng, length(q))
    w_ref_shift = StatlibBackend._solve_simplex_qp_cached(P, q_shift; init = w_fast)
    w_fast_shift = StatlibBackend._solve_simplex_qp_warm(P, q_shift; init = w_fast)

    @test abs(sum(w_fast_shift) - 1.0) < 1e-8
    @test minimum(w_fast_shift) >= -1e-8
    @test maximum(abs.(w_fast_shift .- w_ref_shift)) < 1e-6
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

@testset "Augsynth inference ABI" begin
    n = 40
    t0 = 12
    tpost = 6
    X = randn(n, t0)
    β = randn(t0)
    y = randn(n, tpost)
    trt = zeros(Float64, n)
    trt[1:20] .= 1.0

    # Give nontrivial post dynamics to avoid degenerate fits.
    y .+= X * β .+ reshape(collect(1:tpost), 1, tpost) .* trt
    total = t0 + tpost + 1

    err = zeros(UInt8, 512)
    att = zeros(Float64, total)
    lb = zeros(Float64, total)
    ub = zeros(Float64, total)
    heldout = zeros(Float64, total)

    status = StatlibBackend.jackknife_plus!(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att), pointer(lb), pointer(ub), pointer(heldout),
        0.1, Cint(0), Cint(0), Cint(0),
        pointer([0.0]), Cint(1), Cint(1), pointer(err), Cint(length(err))
    )

    @test status == 0
    @test length(att) == total
    @test length(lb) == total
    @test length(ub) == total
    @test length(heldout) == total
    @test all(isfinite, att[t0 + 1:end])
    @test all(isfinite, lb[t0 + 1:end])
    @test all(isfinite, ub[t0 + 1:end])
    @test all(isfinite, heldout)

    att_jack = zeros(Float64, total)
    se_jack = zeros(Float64, total)
    fill!(err, 0)

    status = StatlibBackend.jackknife_unit_std!(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att_jack), pointer(se_jack),
        Cint(0), Cint(0), pointer([0.0]),
        Cint(1), Cint(1), pointer(err), Cint(length(err))
    )

    @test status == 0
    @test all(isfinite, se_jack[t0 + 1:end])
    @test all(se_jack[1:t0] .!= 0.0)

    att_conf = zeros(Float64, total)
    lb_conf = zeros(Float64, total)
    ub_conf = zeros(Float64, total)
    pval_conf = zeros(Float64, total)
    fill!(err, 0)

    status = StatlibBackend.conformal_inference!(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att_conf), pointer(lb_conf), pointer(ub_conf), pointer(pval_conf),
        0.1, Cint(0), 1.0, Cint(40), Cint(25),
        Cint(0), Cint(0), pointer([0.0]),
        Cint(1), Cint(1), pointer(err), Cint(length(err))
    )

    @test status == 0
    @test all(isfinite, att_conf[t0 + 1:end])
    @test all(isfinite, lb_conf[t0 + 1:end])
    @test all(isfinite, ub_conf[t0 + 1:end])
    @test all(isfinite, pval_conf[t0 + 1:end])

    att_unified = zeros(Float64, total)
    lb_unified = zeros(Float64, total)
    ub_unified = zeros(Float64, total)
    se_unified = zeros(Float64, total)
    heldout_unified = zeros(Float64, total)
    pval_unified = zeros(Float64, total)
    fill!(err, 0)
    status = StatlibBackend.augsynth_inference(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att_unified), pointer(lb_unified), pointer(ub_unified),
        pointer(se_unified), pointer(heldout_unified), pointer(pval_unified),
        Cint(1), 0.1, Cint(0),
        Cint(0), 1.0, Cint(40), Cint(25),
        Cint(0),
        Cint(0), Cint(0), pointer([0.0]),
        Cint(1), Cint(1),
        pointer(err), Cint(length(err))
    )

    @test status == 0
    @test all(isfinite, att_unified[t0 + 1:end])
    @test all(isfinite, se_unified[t0 + 1:end])

    fill!(err, 0)
    status = StatlibBackend.augsynth_inference(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att_unified), pointer(lb_unified), pointer(ub_unified),
        pointer(se_unified), pointer(heldout_unified), pointer(pval_unified),
        Cint(2), 0.1, Cint(0),
        Cint(0), 1.0, Cint(40), Cint(25),
        Cint(0),
        Cint(0), Cint(0), pointer([0.0]),
        Cint(1), Cint(1),
        pointer(err), Cint(length(err))
    )

    @test status == 0
    @test all(isfinite, att_unified[t0 + 1:end])

    fill!(err, 0)
    status = StatlibBackend.augsynth_inference(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att_unified), pointer(lb_unified), pointer(ub_unified),
        pointer(se_unified), pointer(heldout_unified), pointer(pval_unified),
        Cint(3), 0.1, Cint(0),
        Cint(0), 1.0, Cint(40), Cint(25),
        Cint(0),
        Cint(0), Cint(0), pointer([0.0]),
        Cint(1), Cint(1),
        pointer(err), Cint(length(err))
    )

    @test status == 0
    @test all(isfinite, att_unified[t0 + 1:end])
    @test all(isfinite, lb_unified[t0 + 1:end])
    @test all(isfinite, ub_unified[t0 + 1:end])
    @test all(isfinite, pval_unified[t0 + 1:end])

    fill!(err, 0)
    status = StatlibBackend.augsynth_inference(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att_unified), pointer(lb_unified), pointer(ub_unified),
        pointer(se_unified), pointer(heldout_unified), pointer(pval_unified),
        Cint(3), 0.1, Cint(0),
        Cint(0), 1.0, Cint(40), Cint(25),
        Cint(1),
        Cint(0), Cint(0), pointer([0.0]),
        Cint(1), Cint(1),
        pointer(err), Cint(length(err))
    )

    @test status == 0
    @test all(isfinite, lb_unified[t0 + 1:end])
    @test all(isfinite, ub_unified[t0 + 1:end])
    @test all(isfinite, pval_unified[t0 + 1:end])
end

@testset "Single treated jackknife stays valid" begin
    n = 18
    t0 = 8
    tpost = 4
    X = randn(n, t0)
    y = randn(n, tpost)
    trt = zeros(Float64, n)
    trt[end] = 1.0
    y[end, :] .+= 1.5

    total = t0 + tpost + 1
    err = zeros(UInt8, 512)
    att = zeros(Float64, total)
    se = zeros(Float64, total)
    lb = zeros(Float64, total)
    ub = zeros(Float64, total)
    held = zeros(Float64, total)
    pval = zeros(Float64, total)

    status = StatlibBackend.jackknife_unit_std!(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att), pointer(se),
        Cint(0), Cint(1), pointer([0.0]),
        Cint(1), Cint(1), pointer(err), Cint(length(err))
    )

    @test status == 0
    @test all(isfinite, att[t0 + 1:end])
    @test all(isfinite, se[t0 + 1:end])

    fill!(err, 0)
    status = StatlibBackend.augsynth_inference(
        Cint(n), Cint(t0), Cint(tpost),
        pointer(X), pointer(y), pointer(trt),
        pointer(att), pointer(lb), pointer(ub),
        pointer(se), pointer(held), pointer(pval),
        Cint(1), 0.1, Cint(0),
        Cint(0), 1.0, Cint(32), Cint(21),
        Cint(0),
        Cint(0), Cint(1), pointer([0.0]),
        Cint(1), Cint(1),
        pointer(err), Cint(length(err))
    )

    @test status == 0
    @test all(isfinite, se[t0 + 1:end])
end

@testset "Cached conformal path stays valid" begin
    n = 41
    t0 = 80
    tpost = 40
    X = randn(n, t0)
    y = randn(n, tpost)
    trt = zeros(Float64, n)
    trt[end] = 1.0
    y[end, :] .+= 5.0 .+ 0.1 .* collect(1:tpost)

    j = 1
    Xj = Matrix{Float64}(undef, n, t0 + 1)
    Xj[:, 1:t0] .= X
    @views Xj[:, t0 + 1] .= y[:, j]
    yj = @view(y[:, 2:end])

    cache = StatlibBackend._conformal_build_cache(Xj, yj, trt)
    stats = StatlibBackend._pointwise_conformal_stats(X, y, trt)
    fast_cache = StatlibBackend._pointwise_conformal_cache(stats, j)

    @test maximum(abs.(fast_cache.gram .- cache.gram)) < 1e-10
    @test maximum(abs.(fast_cache.q_base .- cache.q_base)) < 1e-10
    @test maximum(abs.(fast_cache.shift_feature .- cache.shift_feature)) < 1e-10
    @test maximum(abs.(fast_cache.X0_raw .- cache.X0_raw)) < 1e-10
    @test maximum(abs.(fast_cache.x1 .- cache.x1)) < 1e-10

    fit = StatlibBackend._fit_from_conformal_cache(cache, 0.0)
    resids = StatlibBackend._conformal_resids_from_cache(cache, Xj, yj, fit, 0.0)
    fast_fit = StatlibBackend._fit_from_conformal_cache(fast_cache, 0.0)
    fast_resids = StatlibBackend._conformal_resids_from_cache(fast_cache, nothing, nothing, fast_fit, 0.0)

    @test length(resids) == size(Xj, 2) + size(yj, 2)
    @test all(isfinite, resids)
    @test maximum(abs.(fast_resids .- resids)) < 1e-8

    pval, weights = StatlibBackend._compute_permute_pval(
        Xj, yj, trt;
        h0 = 0.0,
        post_length = 1,
        type = 0,
        q = 1.0,
        ns = 128,
        ridge = false,
        scm = true,
        lambda = 0.0,
        fit_cache = cache,
        threaded_permutations = false
    )

    @test isfinite(pval)
    @test 0.0 <= pval <= 1.0
    @test length(weights) == sum(trt .== 0.0)

    fast_pval, fast_weights = StatlibBackend._compute_permute_pval(
        StatlibBackend._placeholder_X(fast_cache),
        StatlibBackend._placeholder_y(fast_cache),
        trt;
        h0 = 0.0,
        post_length = 1,
        type = 0,
        q = 1.0,
        ns = 128,
        ridge = false,
        scm = true,
        lambda = 0.0,
        fit_cache = fast_cache,
        threaded_permutations = false
    )

    @test fast_pval == pval
    @test maximum(abs.(fast_weights .- weights)) < 1e-8

    pval_iid_small, _ = StatlibBackend._compute_permute_pval(
        Xj, yj, trt;
        h0 = 0.0,
        post_length = 1,
        type = 0,
        q = 1.0,
        ns = 16,
        ridge = false,
        scm = true,
        lambda = 0.0,
        fit_cache = cache,
        threaded_permutations = false
    )
    pval_iid_large, _ = StatlibBackend._compute_permute_pval(
        Xj, yj, trt;
        h0 = 0.0,
        post_length = 1,
        type = 0,
        q = 1.0,
        ns = 1024,
        ridge = false,
        scm = true,
        lambda = 0.0,
        fit_cache = cache,
        threaded_permutations = false
    )
    pval_block, _ = StatlibBackend._compute_permute_pval(
        Xj, yj, trt;
        h0 = 0.0,
        post_length = 1,
        type = 1,
        q = 1.0,
        ns = 16,
        ridge = false,
        scm = true,
        lambda = 0.0,
        fit_cache = cache,
        threaded_permutations = false
    )

    @test pval_iid_small == pval_iid_large
    @test pval_iid_small == pval_block

    grid = collect(range(-2.0, stop = 2.0, length = 25))
    lo_ref, hi_ref, p_ref, w_ref = StatlibBackend._compute_permute_ci(
        Xj, yj, trt, grid,
        1, 0.1, 1, 1.0, 32,
        false, true, 0.0;
        conformal_mode = StatlibBackend.CONFORMAL_MODE_REFERENCE,
        fit_cache = cache,
        threaded_permutations = false
    )
    lo_grid, hi_grid, p_grid, w_grid = StatlibBackend._compute_permute_ci_grid(
        Xj, yj, trt, grid,
        1, 0.1, 1, 1.0, 32,
        false, true, 0.0;
        fit_cache = cache,
        threaded_permutations = false
    )

    @test isequal(lo_ref, lo_grid)
    @test isequal(hi_ref, hi_grid)
    @test isequal(p_ref, p_grid)
    @test isequal(w_ref, w_grid)

    one_post_stats = StatlibBackend._pointwise_conformal_stats(X, y[:, 1:1], trt)
    one_post_cache = StatlibBackend._pointwise_conformal_cache(one_post_stats, 1)
    @test one_post_cache.tpost == 1
    @test size(one_post_cache.y0, 2) == 1
    @test size(one_post_cache.y1, 1) == 1
    one_post_fit = StatlibBackend._fit_from_conformal_cache(one_post_cache, 0.0)
    one_post_resids = StatlibBackend._conformal_resids_from_cache(
        one_post_cache, nothing, nothing, one_post_fit, 0.0
    )
    @test length(one_post_resids) == t0 + 2
    @test all(isfinite, one_post_resids)
end
