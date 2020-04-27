import Pkg; Pkg.activate(".")

using GeoStats
using DataFrames
using CSV
using MLJ
using Distributed
using DensityRatioEstimation
# using ProgressMeter
using Random
using LossFunctions

# read raw data
df = CSV.read("data/new_zealand/logs_no_duplicates.csv")
df.FORMATION = categorical(df.FORMATION)
df.ONSHORE   = categorical(df.ONSHORE)

# data for formation classification
dfc = dropmissing(df[[:X,:Y,:Z,:GR,:SP,:DENS,:DTC,:TEMP,:FORMATION,:ONSHORE]])

# create spatial data
wells = GeoDataFrame(dfc, [:X,:Y,:Z])

variables(wells)

npoints(wells)

formations = groupby(wells, :FORMATION)

fvalues = get.(formations[:values])
fsizes  = length.(subsets(formations))

ordforms = sortperm(fsizes, rev=true)

G1 = ordforms[1:2]
G2 = ordforms[3:4]
G3 = ordforms[5:end];


Ω = DataCollection(formations[G1])

groups = groupby(Ω, :ONSHORE)

# onshore (True) first and offshore (False) last
ordered = sortperm(groups[:values], rev=true)

Ωs, Ωt = groups[ordered]

allvars = keys(variables(wells))
discard = [:WELL_NAME,:DIRECTIONAL_SURVEY,:ONSHORE,:DEPT,:BS, :FORMATION]
numeric = collect(setdiff(allvars, discard))


# t = RegressionTask((:GR,:DENS,:DTC,:TEMP,:RESD), :SP)
subtypes(AbstractErrorEstimator)


function error_cv(m, p, k, loss)
    s = PointwiseLearn(m)
    v = CrossValidation(k, loss=loss)
    error(s, p, v)
end

function error_bv(m, p, rᵦ, loss)
    s = PointwiseLearn(m)
    v = BlockCrossValidation(rᵦ, loss=loss)
    error(s, p, v)
end

function error_wv(m, p, k, loss, σ=15.,b=10)
    s = PointwiseLearn(m)
    v = DensityRatioValidation(k, estimator=LSIF(σ=σ,b=b), loss=loss)
    error(s, p, v)
end

function error_empirical(m, p, Ωts, loss)
    ŷ = solve(p, PointwiseLearn(m))
    y = targetdata(p)
    result = Dict()
    for (col, 𝔏) in loss
        result[col] = LossFunctions.value(𝔏, y[col], ŷ[col], AggMode.Mean())
    end
    result
end

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
