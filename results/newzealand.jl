# instantiate environment
using Pkg; Pkg.instantiate()

using DataFrames
using CSV

# -------------
# MAIN SCRIPT
# -------------

# load results table
df = CSV.read("newzealand.csv", missingstring="NaN")
dropmissing!(df)

# pretty_table(df, backend=:latex, tf=latex_simple,
             # nosubheader=true, formatters=ft_round(2,3:6))
