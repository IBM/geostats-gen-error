# instantiate environment
using Pkg; Pkg.instantiate()

using DataFrames
using PrettyTables
using CSV

# -------------
# MAIN SCRIPT
# -------------

# load results table
df = CSV.read(joinpath(@__DIR__,"newzealand.csv"))

# highlight table entries
best = Highlighter(
  (d, i, j) -> j ≤ 3 && argmin([abs(d[i,k] - d[i,4]) for k in 1:3]) == j,
  bold=true, foreground=:blue
)
worst = Highlighter(
  (d, i, j) -> j ≤ 3 && argmax([abs(d[i,k] - d[i,4]) for k in 1:3]) == j,
  bold=true, foreground=:red
)
actual = Highlighter(
  (d, i, j) -> (j == 4),
  foreground=:dark_gray
)

for g in groupby(df, :TARGET)
  pretty_table(g, nosubheader=true, crop=:none,
               formatters=ft_round(3,1:4),
               highlighters=(best, worst, actual))

  # model ranking based on each method
  ranks = map([:CV,:BCV,:DRV,:ACTUAL]) do err
    r = sortperm(g[!,err])
    Symbol(err," RANK") => g[!,:MODEL][r]
  end
  r = DataFrame(ranks)
  pretty_table(r, nosubheader=true, crop=:none)
end

# pretty_table(df, backend=:latex, tf=latex_simple,
#              nosubheader=true, formatters=ft_round(2,3:6))
