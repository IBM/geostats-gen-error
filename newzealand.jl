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

# reproducible results
Random.seed!(2020)

# estimators of generalization error
error_cv( m, p, k, ℒ) = error(PointwiseLearn(m), p, CrossValidation(k, loss=ℒ))
error_bcv(m, p, r, ℒ) = error(PointwiseLearn(m), p, BlockCrossValidation(r, loss=ℒ))
error_drv(m, p, k, ℒ) = error(PointwiseLearn(m), p, DensityRatioValidation(k, loss=ℒ, estimator=LSIF(σ=2.0,b=10)))

# actual error (known labels)
function error_empirical(m, p, ℒ)
  t  = task(p)
  Ωs = sourcedata(p)
  Ωt = targetdata(p)

  # learn task on source domain and perform
  # it on both source and target domains
  lm = learn(t, Ωs, m)
  ŷs = perform(t, Ωs, lm)
  ŷt = perform(t, Ωt, lm)

  # error on source
  ϵs = map(outputvars(t)) do var
    var => LossFunctions.value(ℒ[var], Ωs[var], ŷs[var], AggMode.Mean())
  end

  # error on target
  ϵt = map(outputvars(t)) do var
    var => LossFunctions.value(ℒ[var], Ωt[var], ŷt[var], AggMode.Mean())
  end

  Dict(ϵs), Dict(ϵt)
end

function experiment(m, p, r, k, ℒ)
  # try different error estimates
  cv  = error_cv( m, p, k, ℒ)
  bcv = error_bcv(m, p, r, ℒ)
  drv = error_drv(m, p, k, ℒ)

  # actual error (unhide labels)
  ϵs, ϵt = error_empirical(m, p, ℒ)

  # model name without suffix
  model = replace(info(m).name, r"(.*)(Regressor|Classifier)" => s"\g<1>")

  map(outputvars(task(p))) do var
    (VARIABLE=var, MODEL=model,
     SOURCE=ϵs[var], TARGET=ϵt[var],
     CV=cv[var], BCV=bcv[var], DRV=drv[var])
  end
end

# -------------
# MAIN SCRIPT
# -------------

# download dataset if needed
register(DataDep("NewZealand",
         "Taranaki Basin Curated Well Logs",
         "https://dax-cdn.cdn.appdomain.cloud/dax-taranaki-basin-curated-well-logs/1.0.0/taranaki-basin-curated-well-logs.tar.gz",
         "608f7aad5a4e9fded6441fd44f242382544d3f61790446175f5ede83f15f4d11",
         post_fetch_method=DataDeps.unpack))

# name of the CSV file in the dataset
csv = joinpath(datadep"NewZealand","taranaki-basin-curated-well-logs","logs.csv")

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

# we are left with two formations onshore and offshore
# make sure that these two are balanced for classification
fs, ft = Ωs[:FORMATION], Ωt[:FORMATION]

# formation counts
ms = count(isequal("Urenui"), fs)
ns = count(isequal("Manganui"), fs)
mt = count(isequal("Urenui"), ft)
nt = count(isequal("Manganui"), ft)

# formation proportions
ps = ms / (ms + ns)
pt = mt / (mt + nt)

# weighted sampling
ws = [f == "Urenui" ? 0.5/ps : 0.5/(1-ps) for f in fs]
wt = [f == "Urenui" ? 0.5/pt : 0.5/(1-pt) for f in ft]
Ωs = sample(Ωs, 300000, ws, replace=false)
Ωt = sample(Ωt,  50000, wt, replace=false)

# drop levels to avoid known downstream issues in MLJ
fs, ft = Ωs[:FORMATION], Ωt[:FORMATION]
levels!(fs, ["Urenui","Manganui"])
levels!(ft, ["Urenui","Manganui"])
𝒫s = georef(OrderedDict(:FORMATION => fs), domain(Ωs))
𝒫t = georef(OrderedDict(:FORMATION => ft), domain(Ωt))
Ωs = join(view(Ωs, logs), 𝒫s)
Ωt = join(view(Ωt, logs), 𝒫t)

# block sides and number of folds for error estimators
r = (10000., 10000., 500.)
k = length(GeoStats.partition(Ωs, BlockPartitioner(r)))

# ---------------
# CLASSIFICATION
# ---------------
@load DummyClassifier pkg="ScikitLearn"
@load RidgeClassifier pkg="ScikitLearn"
@load LogisticClassifier pkg="ScikitLearn"
@load KNeighborsClassifier pkg="ScikitLearn"
@load GaussianNBClassifier pkg="ScikitLearn"
@load BayesianLDA pkg="ScikitLearn"
@load PerceptronClassifier pkg="ScikitLearn"
@load DecisionTreeClassifier pkg="DecisionTree"

# list of models
mrange = [RidgeClassifier(), LogisticClassifier(), KNeighborsClassifier(),
          GaussianNBClassifier(), BayesianLDA(), PerceptronClassifier(),
          DecisionTreeClassifier(), DummyClassifier()]

# experiment iterator and progress
iterator = Iterators.product(mrange)
progress = Progress(length(iterator), "Formation prediction:")

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
@load DummyRegressor pkg="ScikitLearn"
@load RidgeRegressor pkg="ScikitLearn"
@load LassoRegressor pkg="ScikitLearn"
@load KNeighborsRegressor pkg="ScikitLearn"
@load DecisionTreeRegressor pkg="DecisionTree"

# list of models
mrange = [RidgeRegressor(), LassoRegressor(), KNeighborsRegressor(),
          DecisionTreeRegressor(), DummyRegressor()]

# variables to predict
vrange = logs

# experiment iterator and progress
iterator = Iterators.product(mrange, vrange)
progress = Progress(length(iterator), "Well log prediction:")

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
