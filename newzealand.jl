using Pkg; Pkg.instantiate()

using GeoStats
using DensityRatioEstimation
using LossFunctions
using ProgressMeter
using DataFrames
using MLJ, CSV
using Statistics
using Random

# reproducible results
Random.seed!(123)

# estimators of generalization error
error_cv( m, p, k, ℒ) = error(PointwiseLearn(m), p, CrossValidation(k, loss=ℒ))
error_bcv(m, p, r, ℒ) = error(PointwiseLearn(m), p, BlockCrossValidation(r, loss=ℒ))
error_drv(m, p, k, σ, ℒ) = error(PointwiseLearn(m), p, DensityRatioValidation(k, loss=ℒ, estimator=LSIF(σ=σ,b=10)))

# true error (known labels)
function error_empirical(m, p, ℒ)
  results = map(outputvars(task(p))) do var
    y = targetdata(p)[var]
    ŷ = solve(p, PointwiseLearn(m))[var]
    var => LossFunctions.value(ℒ[var], y, ŷ, AggMode.Mean())
  end
  Dict(results)
end

function experiment(m, p, σ, r, k, ℒ)
    # try different error estimates
    cv  = error_cv(m, p, k, ℒ)
    bcv = error_bcv(m, p, r, ℒ)
    drv = error_drv(m, p, k, σ, ℒ)

    # actual error (unhide labels)
    actual = error_empirical(m, p, ℒ)

    map(outputvars(task(p))) do var
      (σ=σ, CV=cv[var], BCV=bcv[var], DRV=drv[var],
       ACTUAL=actual[var], MODEL=info(m).name, TARGET=var)
    end
end

# -------------
# MAIN SCRIPT
# -------------

# logs used in the experiment
logs = [:GR,:SP,:DENS,:NEUT,:DTC]

# read/clean raw data
df = CSV.read("data/newzealand.csv")
df = df[:,[:X,:Y,:Z,logs...,:FORMATION,:ONSHORE]]
dropmissing!(df)
categorical!(df, :FORMATION)
categorical!(df, :ONSHORE)
for log in logs
  x = df[!,log]
  μ = mean(x)
  σ = std(x, mean=μ)
  df[!,log] .= (x .- μ) ./ σ
end

# define spatial data
wells = GeoDataFrame(df, [:X,:Y,:Z])

# select the two most frequent formations
formations = groupby(wells, :FORMATION)
frequency = sortperm(npoints.(formations), rev=true)
Ω = DataCollection(formations[frequency[1:2]])

# split onshore (True) vs. offshore (False)
onoff = groupby(Ω, :ONSHORE)
ordered = sortperm(onoff[:values], rev=true)
Ωs, Ωt = onoff[ordered]

# sample the data
# Ωs = sample(Ωs, 10000)
# Ωt = sample(Ωt,  2000)

# materialize the views (to avoid indirections to memory)
Ds = OrderedDict{Symbol,AbstractArray}(v => Ωs[v] for (v,V) in variables(Ωs))
Dt = OrderedDict{Symbol,AbstractArray}(v => Ωt[v] for (v,V) in variables(Ωt))
Ωs = PointSetData(Ds, coordinates(Ωs))
Ωt = PointSetData(Dt, coordinates(Ωt))

# set block side and equivalent number of folds
r = 500.
k = length(GeoStats.partition(Ωs, BlockPartitioner(rᵦ)))

# ---------------
# CLASSIFICATION
# ---------------
t = ClassificationTask(logs, :FORMATION)
p = LearningProblem(Ωs, Ωt, t)

@load DecisionTreeClassifier
@load KNNClassifier

ℒ = Dict(:FORMATION => MisclassLoss())

# parameter ranges
mrange = [DecisionTreeClassifier(),KNNClassifier()]
σrange = [1.,5.,10.,15.,20.,25.]

# experiment iterator and progress
iterator = Iterators.product(mrange, σrange)
progress = Progress(length(iterator), "New Zealand classification:")

# return missing in case of failure
skip = e -> (println("Skipped: $e"); missing)

# perform experiments
cresults = progress_pmap(iterator, progress=progress,
                        on_error=skip) do (m, σ)
  experiment(m, p, σ, rᵦ, k, ℒ)
end

# -----------
# REGRESSION
# -----------
@load LinearRegressor pkg="MLJLinearModels"
@load DecisionTreeRegressor pkg="DecisionTree"
@load RandomForestRegressor pkg="DecisionTree"
@load KNNRegressor

# parameter ranges
mrange = [LinearRegressor(),DecisionTreeRegressor(),
          RandomForestRegressor(),KNNRegressor()]
vrange = [:GR]

# experiment iterator and progress
iterator = Iterators.product(mrange, σrange, vrange)
progress = Progress(length(iterator), "New Zealand regression:")

# perform experiments
rresults = progress_pmap(iterator, progress=progress,
                        on_error=skip) do (m, σ, v)
  t = RegressionTask(logs[logs .!= v], v)
  p = LearningProblem(Ωs, Ωt, t)
  ℒ = Dict(v => L2DistLoss())
  experiment(m, p, σ, rᵦ, k, ℒ)
end

# merge all results into dataframe
all = vcat(cresults, rresults)
res = DataFrame(skipmissing(Iterators.flatten(all)))

# save all results to disk
fname = joinpath(@__DIR__,"results","newzealand.csv")
CSV.write(fname, res)
