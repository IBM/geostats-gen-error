using GeoStats
using DensityRatioEstimation
using LossFunctions
using ProgressMeter
using DataFrames
using MLJ, CSV
using Random

# reproducible results
Random.seed!(123)

# estimators of generalization error
error_cv( m, p, k, ℒ) = error(PointwiseLearn(m), p, CrossValidation(k, loss=ℒ))
error_bcv(m, p, r, ℒ) = error(PointwiseLearn(m), p, BlockCrossValidation(r, loss=ℒ))
error_drv(m, p, k, ℒ) = error(PointwiseLearn(m), p, DensityRatioValidation(k, loss=ℒ, estimator=LSIF(σ=2.0,b=10)))

# true error (known labels)
function error_empirical(m, p, Ωts, ℒ)
  results = map(keys(loss)) do var
    y = targetdata(p)[var]
    ŷ = solve(p, PointwiseLearn(m))[var]
    var => LossFunctions.value(ℒ[var], y, ŷ, AggMode.Mean())
  end
  Dict(results)
end

# TODO: simplify the comparison code
function error_comparison(m, p, Ωts, rᵦ, k, loss, col)
    # parameters for validation methods
    #@assert rᵦ ≥ r "block size smaller than correlation length"

    # try different error estimates
    loss_dict = Dict(col => loss)
    println("Performing CV")
    @time cv = error_cv(m, p, k, loss_dict)[col]
    @show cv
    println("Performing BCV")
    @time bv = error_bv(m, p, rᵦ, loss_dict)[col]
    drv_results = Dict()
    for σ in [1., 5., 10., 15.,  20., 25.]
        println("Performing DRV with σ=$(σ)")
        try
            @time drv = error_wv(m, p, k, loss_dict, σ)[col]
            drv_results[Symbol("DRV_$(Int(σ))")] = drv
        catch e
            println("skipping DRV - invalid σ")
            println(e)
        end
    end

    # true error
    actual = error_empirical(m, p, Ωts,loss_dict)[col]

    merge((rᵦ=rᵦ, k=k, CV=cv, BCV=bv),
          drv_results,
          (ACTUAL=actual, MODEL=info(m).name, target=col))
end

# -------------
# MAIN SCRIPT
# -------------

# read/clean raw data
df  = CSV.read("data/new_zealand/logs_no_duplicates.csv")
dfc = dropmissing(df[[:X,:Y,:Z,:GR,:SP,:DENS,:DTC,:TEMP,:FORMATION,:ONSHORE]])
categorical!(dfc, :FORMATION)
categorical!(dfc, :ONSHORE)

# define spatial data
wells = GeoDataFrame(dfc, [:X,:Y,:Z])

# group formations in terms of number of points
formations = groupby(wells, :FORMATION)
ind = sortperm(npoints.(formations), rev=true)
G1 = ind[1:2]
G2 = ind[3:4]
G3 = ind[5:end]

# only consider formations in G1
Ω = DataCollection(formations[G1])

# split onshore (True) vs. offshore (False)
groups = groupby(Ω, :ONSHORE)
ordered = sortperm(groups[:values], rev=true)
Ωs, Ωt = groups[ordered]

# distinguish types of variables
allvars = keys(variables(Ωs))
discard = [:WELL_NAME,:DIRECTIONAL_SURVEY,:ONSHORE,:DEPT,:BS, :FORMATION]
numeric = collect(setdiff(allvars, discard))

#TODO find the best rᵦ using variograms
#EmpiricalVariogram(Ωs, :TEMP)
rᵦ=500
k = length(GeoStats.partition(Ωs, BlockPartitioner(rᵦ)))
show_all(x) = show(stdout, "text/plain", x)

# --------
# all_class_models = models(m->m.is_pure_julia && m.is_supervised &&
#                           m.target_scitype == AbstractVector{<:Finite})
# show_all(all_class_models)
#
# @load DecisionTreeClassifier
# @load KNNClassifier
# # mrange = [DecisionTreeClassifier(), KNNClassifier()]
# class_models = [DecisionTreeClassifier(), KNNClassifier()]
# loss = ZeroOneLoss()
# t = ClassificationTask((:GR,:SP,:DENS,:DTC,:TEMP), :FORMATION)
# problem = LearningProblem(Ωs, Ωt, t)
#
# Random.seed!(42)
#
# class_results = DataFrame()
#
# for model in class_models#, δ in δrange, τ in τrange, r in rrange, ρ in ρrange
#     @show model, rᵦ, k#, δ, τ, r, ρ
#     try
#         result = DataFrame([error_comparison(model, problem, Ωt, rᵦ, k, loss, :FORMATION)
#                             for i in 1:1])
#         append!(class_results, result)
#     catch e
#         println("skipped")
#         println(e)
#     end
# end
#
# CSV.write("results/new_zealand_classification.csv", class_results)
# println("Classification comparison is done!")
# --------------------
all_reg_models = models(m->m.is_pure_julia && m.is_supervised &&
                        m.target_scitype == AbstractArray{Continuous,1})

show_all(all_reg_models)
Random.seed!(42)

reg_results = DataFrame()

@load LinearRegressor pkg="MLJLinearModels"
@load DecisionTreeRegressor pkg="DecisionTree"
@load RandomForestRegressor pkg="DecisionTree"
@load KNNRegressor

reg_models = [LinearRegressor(), DecisionTreeRegressor(),
              RandomForestRegressor(), KNNRegressor()]
loss = L2DistLoss()

for model in reg_models, target in numeric

    t = RegressionTask(numeric[numeric .!= target], target)
    problem = LearningProblem(Ωs, Ωt, t)

    @show model, target, task#, δ, τ, r, ρ
    try
        result = DataFrame([error_comparison(model, problem, Ωt, rᵦ, k, loss, target)
                            for i in 1:1])
        append!(reg_results, result)
        @show result
    catch e
        println("skipped")
        println(e)
    end
    CSV.write("results/new_zealand_regression_drv.csv", reg_results)
end

# CSV.write("results/new_zealand_regression.csv", reg_results)
