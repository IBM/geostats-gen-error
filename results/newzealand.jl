# instantiate environment
using Pkg; Pkg.instantiate()

using DataFrames
using CSV

using StringDistances
using Statistics
# -------------
# MAIN SCRIPT
# -------------

# load results table
df = CSV.read("newzealand.csv", missingstring="NaN")
#dropmissing!(df)

# pretty_table(df, backend=:latex, tf=latex_simple,
# nosubheader=true, formatters=ft_round(2,3:6))

# this is a script to compare 3 cross-validation methods (columns)
# on how they rank a fixed set of models (lines)
#  I will convert the data to a Matrix.
#  To do: check how to use direclty the dataframe...
M = convert(Matrix,df)

# get the values I need
MyAccuracies = Float64.(M[:,3:5])
TrueAccuracies = Float64.(M[:,6])

##
@info("STRATEGY ONE: rank the CV methods for each single classifier (line)")
# This helps to see if there is a CV method that on average ranks
# better for many classifiers

# compute "deltas", i.e, the absolute deviation from ideal estimate
deltas = zeros(size(MyAccuracies))
for k=1:size(MyAccuracies,2)
    deltas[:,k] =  abs.(MyAccuracies[:,k]-TrueAccuracies)
end

println(" Average abs deviations from true estimate $(mean(deltas,dims=1)) ")

# sort accuracies for each row, eg 0.4, 0.2, 0.5 -> 2 1 3
ranksRows = zeros(size(MyAccuracies))
for row=1:size(deltas,1)
    ranksRows[row,:] = sortperm(deltas[row,:])
end

println(" Average rank of estimators are $(mean(ranksRows,dims=1)) ")

# Now, let's use an alternative measure to compare overall ranks with the "ideal" rank
# Let's use eg. Levenshtein distance
scores = [evaluate(Levenshtein(), ranksRows[:,k], ones(length(TrueAccuracies))) for k in 1:size(ranksRows,2)]
println(" Average scores using Levenshtein distance $scores ")

##
@info("STRATEGY TWO: rank the classifiers according to each CV method")
# Compare how a certain CV ranking aggrees with the ideal ranking
ranksCols = zeros(size(M[:,3:6]))
for col=1:size(ranksCols,2)
    ranksCols[:,col] = sortperm(M[:,col])
end

scoresCols = [evaluate(Levenshtein(), ranksCols[:,k], ranksCols[:,end]) for k in 1:size(ranksCols,2)]
println(" Average scores using Levenshtein distance $scoresCols ")
