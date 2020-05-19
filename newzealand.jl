using Pkg; Pkg.instantiate()

using GeoStats
using DensityRatioEstimation
using LossFunctions
using ProgressMeter
using DataFrames
using DataDeps
using MLJ, CSV
using Statistics
using Random

# download dataset if needed
register(DataDep("NewZealand",
         "Taranaki Basin Curated Well Logs",
         "https://dax-cdn.cdn.appdomain.cloud/dax-taranaki-basin-curated-well-logs/1.0.0/taranaki-basin-curated-well-logs.tar.gz",
         "608f7aad5a4e9fded6441fd44f242382544d3f61790446175f5ede83f15f4d11",
         post_fetch_method=unpack))

# name of the CSV file in the dataset
csv = joinpath(datadep"NewZealand","taranaki-basin-curated-well-logs","logs.csv")

# reproducible results
Random.seed!(2020)

# estimators of generalization error
error_cv( m, p, k, ℒ) = error(PointwiseLearn(m), p, CrossValidation(k, loss=ℒ))
error_bcv(m, p, r, ℒ) = error(PointwiseLearn(m), p, BlockCrossValidation(r, loss=ℒ))
error_drv(m, p, k, ℒ) = error(PointwiseLearn(m), p, DensityRatioValidation(k, loss=ℒ, estimator=LSIF(σ=2.0,b=10)))

# true error (known labels)
function error_empirical(m, p, ℒ)
  results = map(outputvars(task(p))) do var
    y = targetdata(p)[var]
    ŷ = solve(p, PointwiseLearn(m))[var]
    var => LossFunctions.value(ℒ[var], y, ŷ, AggMode.Mean())
  end
  Dict(results)
end

function experiment(m, p, r, k, ℒ)
    # try different error estimates
    cv  = error_cv( m, p, k, ℒ)
    bcv = error_bcv(m, p, r, ℒ)
    drv = error_drv(m, p, k, ℒ)

    # actual error (unhide labels)
    actual = error_empirical(m, p, ℒ)

    map(outputvars(task(p))) do var
      (CV=cv[var], BCV=bcv[var], DRV=drv[var],
       ACTUAL=actual[var], MODEL=info(m).name, TARGET=var)
    end
end

# -------------
# MAIN SCRIPT
# -------------

# logs used in the experiment
logs = [:GR,:SP,:DENS,:NEUT,:DTC]

# read/clean raw data
df = CSV.read(csv)
df = df[:,[logs...,:X,:Y,:Z,:FORMATION,:ONSHORE]]
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
𝒞 = DataCollection(formations[frequency[1:2]])

# eliminate duplicate coordinates
Ω = uniquecoords(𝒞)

# split onshore (True) vs. offshore (False)
onoff = groupby(Ω, :ONSHORE)
order = sortperm(onoff[:values], rev=true)
Ωs, Ωt = onoff[order]

# set block sides and equivalent number of folds
r = (10000.,10000.,500.)
k = length(GeoStats.partition(Ωs, BlockPartitioner(r)))

# ---------------
# CLASSIFICATION
# ---------------
@load LDA pkg="MultivariateStats"
@load KNNClassifier pkg="NearestNeighbors"
@load EvoTreeClassifier pkg="EvoTrees"
@load GaussianNBClassifier pkg="NaiveBayes"

# parameter ranges
mrange = [LDA(), ConstantClassifier(), KNNClassifier(),
          EvoTreeClassifier(), GaussianNBClassifier()]

# experiment iterator and progress
iterator = Iterators.product(mrange)
progress = Progress(length(iterator), "New Zealand classification:")

# return missing in case of failure
skip = e -> (println("Skipped: $e"); missing)

# perform experiments
cresults = progress_pmap(iterator, progress=progress,
                         on_error=skip) do (m,)
  t = ClassificationTask(logs, :FORMATION)
  p = LearningProblem(Ωs, Ωt, t)
  ℒ = Dict(:FORMATION => MisclassLoss())
  experiment(m, p, r, k, ℒ)
end

# -----------
# REGRESSION
# -----------
@load KNNRegressor pkg="NearestNeighbors"
@load DecisionTreeRegressor pkg="DecisionTree"

# parameter ranges
mrange = [ConstantRegressor(), KNNRegressor(), DecisionTreeRegressor()]
vrange = logs

# experiment iterator and progress
iterator = Iterators.product(mrange, vrange)
progress = Progress(length(iterator), "New Zealand regression:")

# perform experiments
rresults = progress_pmap(iterator, progress=progress,
                         on_error=skip) do (m, v)
  t = RegressionTask(logs[logs .!= v], v)
  p = LearningProblem(Ωs, Ωt, t)
  ℒ = Dict(v => L2DistLoss())
  experiment(m, p, r, k, ℒ)
end

# merge all results into dataframe
allres = vcat(cresults[:], rresults[:])
resdf = DataFrame(Iterators.flatten(skipmissing(allres)))

# save all results to disk
fname = joinpath(@__DIR__,"results","newzealand.csv")
CSV.write(fname, resdf)
