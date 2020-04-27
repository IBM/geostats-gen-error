using GeoStats
using DataFrames
using CSV
#using DensityRatioEstimation
# using ProgressMeter
using Random
# using Variography
using Plots


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

Ω = DataCollection(wells)

groups = groupby(Ω, :ONSHORE)

# onshore (True) first and offshore (False) last
ordered = sortperm(groups[:values], rev=true)

Ωonshore, Ωoffshore = groups[ordered]

allvars = keys(variables(wells))
discard = [:WELL_NAME,:DIRECTIONAL_SURVEY,:ONSHORE,:DEPT,:BS, :FORMATION]
numeric = collect(setdiff(allvars, discard))


# Do proper variography
#@time

plt = plot(p_x,p_y,p_z, layout=(3,1))
savefig(plt, "$(var)_variogram.svg")
maxlag=100.
tol = 10.
nlags = 50
for var in numeric
    Ωᵧ = PointSetData(OrderedDict(var=>Ωonshore[var]), coordinates(Ωs))
    Ωᵧ = sample(Ωᵧ, 10000, replace=false)
    # EmpiricalVariogram(Ωᵧ, var, maxlag=10)

    p_x = plot(DirectionalVariogram(Ωᵧ, (1., 0., 0.), var,
               maxlag=maxlag, tol=tol, nlags=nlags),
               title = "$var, maxlag=$maxlag, axis=x")
    p_y = plot(DirectionalVariogram(Ωᵧ, (0., 1., 0.), var,
               maxlag=maxlag, tol=tol, nlags=nlags),
               title = "$var, maxlag=$maxlag, axis=y")
    p_z = plot(DirectionalVariogram(Ωᵧ, (0., 0., 1.), var,
               maxlag=maxlag, tol=tol, nlags=nlags),
               title = "$var, maxlag=$maxlag, axis=z")
    plt = plot(p_x,p_y,p_z, layout=(3,1))
    savefig(plt, "$(var)_variogram.svg")
