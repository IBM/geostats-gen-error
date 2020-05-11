# instantiate environment
using Pkg; Pkg.instantiate()

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
Random.seed!(2020)

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
  μ₂, σ₂ = (3*√2*σ₁)δ + μ₁, τ*σ₁
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
  function f(Γ)
    labels = [label(z) for z in view(Γ, [:Z1,:Z2])]
    ldict  = OrderedDict(:LABEL => categorical(labels))
    georef(ldict, domain(Γ))
  end

  # add labels to all samples
  Ωss = [join(Ω, f(Ω)) for Ω in simgs]
  Ωts = [join(Ω, f(Ω)) for Ω in timgs]

  # geostatistical learning problem
  t = ClassificationTask((:Z1,:Z2), :LABEL)
  p = LearningProblem(Ωss[1], Ωts[1], t)

  # return problem and other Ωt samples
  p, Ωts[2:end]
end

# estimators of generalization error
error_cv( m, p, k, ℒ) = error(PointwiseLearn(m), p, CrossValidation(k, loss=ℒ))
error_bcv(m, p, r, ℒ) = error(PointwiseLearn(m), p, BlockCrossValidation(r, loss=ℒ))
error_drv(m, p, k, ℒ) = error(PointwiseLearn(m), p, DensityRatioValidation(k, loss=ℒ, estimator=LSIF(σ=2.0,b=10)))

# true error (empirical approximation)
function error_empirical(m, p, Ωts, ℒ)
  # train on source data
  lm = learn(task(p), sourcedata(p), m)

  # test on various samples of target data
  ϵs = map(Ωts) do Ωt
    Ω̂t = perform(task(p), Ωt, lm)
    y  = vec(Ωt[:LABEL])
    ŷ  = vec(Ω̂t[:LABEL])
    LossFunctions.value(ℒ[:LABEL], y, ŷ, AggMode.Mean())
  end

  # return mean loss
  mean(ϵs)
end

function experiment(m, δ, τ, r, ℒ)
  # sample a problem
  p, Ωts = problem(δ=δ, τ=τ, r=r)

  # parameters for validation methods
  rᵦ = 20.
  s  = size(sourcedata(p))
  k  = round(Int, prod(s ./ rᵦ))

  # try different error estimates
  cv  = error_cv( m, p, k,  ℒ)[:LABEL]
  bcv = error_bcv(m, p, rᵦ, ℒ)[:LABEL]
  drv = error_drv(m, p, k,  ℒ)[:LABEL]

  # actual error (empirical estimate)
  actual = error_empirical(m, p, Ωts, ℒ)

  # model name (without "Classifier" suffix)
  model = chop(info(m).name, tail=10)

  (δ=δ, τ=τ, r=r, CV=cv, BCV=bcv, DRV=drv, ACTUAL=actual, MODEL=model)
end

# -------------
# MAIN SCRIPT
# -------------

# learning models
@load KNNClassifier
@load DecisionTreeClassifier

# misclassification loss
ℒ = Dict(:LABEL => MisclassLoss())

# parameter ranges
mrange = [DecisionTreeClassifier(),KNNClassifier()]
δrange = 0.0:0.1:0.7
τrange = 0.5:0.1:1.0
rrange = [1e-4,1e+1,2e+1]

# experiment iterator and progress
iterator = Iterators.product(mrange, δrange, τrange, rrange)
progress = Progress(length(iterator), "Gaussian experiment:")

# perform experiments
results = DataFrame()
for iter in iterator
  m, δ, τ, r = iter
  try
    push!(results, experiment(m, δ, τ, r, ℒ))
  catch e
    e isa InterruptException && break
    println("Skipped m=$m δ=$δ τ=$τ r=$r")
  end
  model = chop(info(m).name, tail=10)
  next!(progress, showvalues=[(:model,model), (:δ,δ), (:τ,τ), (:r,r)])
end

# save results
fname = joinpath(@__DIR__,"results","gaussian.csv")
CSV.write(fname, results)
