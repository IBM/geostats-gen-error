using GeoStats
using SpectralGaussianSimulation
using DensityRatioEstimation
using CategoricalArrays
using LossFunctions
using ProgressMeter
using DataFrames
using MLJ, CSV
using LinearAlgebra
using Random

# reproducible results
Random.seed!(123)

# generate random images with given spatial mean, range and sill
function generator(nimgs=1; mean=0., range=1., sill=1., size=(100,100))
  γ = GaussianVariogram(range=range, sill=sill)
  p = SimulationProblem(RegularGrid{Float64}(size), (:Z1=>Float64,:Z2=>Float64), nimgs)
  s = SpecGaussSim(:Z1=>(mean=mean, variogram=γ), :Z2=>(mean=mean, variogram=γ))
  solve(p, s)
end

# generate a covariate shift configuration with given δ and τ
function covariateshift(δ, τ, r; ns=1, nt=1, size=(100,100))
  μ₁, σ₁ = 0.0, 1.0
  μ₂, σ₂ = 3*√2*σ₁*δ, τ*σ₁
  simgs = generator(ns, mean=μ₁, sill=σ₁^2, range=r, size=size)
  timgs = generator(nt, mean=μ₂, sill=σ₂^2, range=r, size=size)
  simgs, timgs
end

# geostatistical learning problem with given covariate shift configuration
function problem(; δ=0.0, τ=1.0, r=10.0, size=(100,100))
  # covariate shift
  simgs, timgs = covariateshift(δ, τ, r, nt=101, size=size)

  # sine-norm labeling function
  label(z, p=1) = sin(4*norm(z, p)) < 0 ? 1 : 0
  f(Γ) = georef(OrderedDict(:LABEL => categorical([label(z) for z in view(Γ, [:Z1,:Z2])])), domain(Γ))

  # add labels to all samples
  Ωss = [join(Ω, f(Ω)) for Ω in simgs]
  Ωts = [join(Ω, f(Ω)) for Ω in timgs]

  # geostatistical learning problem
  p = LearningProblem(Ωss[1], Ωts[1], ClassificationTask((:Z1,:Z2), :LABEL))

  # return problem and other Ωt samples
  p, Ωts[2:end]
end

# estimators of generalization error
error_cv(m, p, k) = error(PointwiseLearn(m), p, CrossValidation(k))
error_bv(m, p, r) = error(PointwiseLearn(m), p, BlockCrossValidation(r))
error_dr(m, p, k) = error(PointwiseLearn(m), p, DensityRatioValidation(k,estimator=LSIF(σ=2.0,b=10)))

# true error (empirical approximation)
function error_empirical(m, p, Ωts)
  # train on source data
  l = GeoStats.learn(task(p), sourcedata(p), m)

  # test on various samples of target data
  es = map(Ωts) do Ωt
    y = vec(Ωt[:LABEL])
    ŷ = vec(perform(task(p), Ωt, l)[:LABEL])
    𝔏 = MisclassLoss()
    LossFunctions.value(𝔏, y, ŷ, AggMode.Mean())
  end

  # averate misclassification rate
  mean(es)
end

function error_comparison(m, δ, τ, r)
  # sample a problem
  p, Ωts = problem(δ=δ, τ=τ, r=r)

  # parameters for validation methods
  rᵦ = 20.
  s  = size(sourcedata(p))
  k  = round(Int, prod(s ./ rᵦ))

  # try different error estimates
  cv = error_cv(m, p, k)[:LABEL]
  bv = error_bv(m, p, rᵦ)[:LABEL]
  dr = error_dr(m, p, k)[:LABEL]

  # actual error (empirical estimate)
  actual = error_empirical(m, p, Ωts)

  (δ=δ, τ=τ, r=r, CV=cv, BV=bv, DR=dr, ACTUAL=actual, MODEL=info(m).name)
end

# -------------
# MAIN SCRIPT
# -------------

# learning models
@load KNNClassifier
@load DecisionTreeClassifier

# parameter ranges
δrange = 0.0:0.1:0.7
τrange = 0.5:0.1:1.0
rrange = [1e-4,1e+1,2e+1]
mrange = [DecisionTreeClassifier(),KNNClassifier()]

results = DataFrame()
@showprogress for m in mrange, δ in δrange, τ in τrange, r in rrange
  try
    push!(results, error_comparison(m, δ, τ, r))
  catch e
    println("Skipped m=$m δ=$δ τ=$τ r=$r")
  end
end

CSV.write("results/gaussian.csv", results)
