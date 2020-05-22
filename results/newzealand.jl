# instantiate environment
using Pkg; Pkg.instantiate()

using DataFrames
using PrettyTables
using ColorSchemes
using CSV

# valid colors for the integer range 1, 2, ..., kmax
function color(k, kmax)
  cs = reverse(colorschemes[:nuuk])
  nc = length(cs)
  i = ceil(Int, (k / kmax) * nc)
  R = round(Int,cs[i].r*255)
  G = round(Int,cs[i].g*255)
  B = round(Int,cs[i].b*255)
  R, G, B
end

# -------------
# MAIN SCRIPT
# -------------

# load results table
df = CSV.read(joinpath(@__DIR__,"newzealand.csv"))

# name columns
serror = 3
terror = 4
errors = 5:7

# highlight table entries
best = Highlighter((d, i, j) -> begin
  k = argmin([abs(d[i,k] - d[i,terror]) for k in errors])
  j ∈ errors && errors[k] == j
  end, bold=true, foreground=:blue
)
worst = Highlighter((d, i, j) -> begin
  k = argmax([abs(d[i,k] - d[i,terror]) for k in errors])
  j ∈ errors && errors[k] == j
  end, bold=true, foreground=:red
)
actual = Highlighter((d, i, j) ->
  j ∈ serror ∪ terror, foreground=:dark_gray
)

for g in groupby(df, :VARIABLE)
  pretty_table(g, nosubheader=true, crop=:none,
               alignment=[:c,:c,:r,:r,:r,:r,:r],
               formatters=ft_round(3, serror ∪ terror ∪ errors),
               highlighters=(best, worst, actual))

  # model ranking based on each method
  ranks = map([:TARGET,:CV,:BCV,:DRV]) do err
    r = sortperm(g[!,err])
    Symbol(err," RANK") => g[!,:MODEL][r]
  end
  r = DataFrame(ranks)

  hs = Tuple([Highlighter((d, i, j) -> d[i,j] == r[k,Symbol("TARGET RANK")],
                          background=color(k,size(r,1)), foreground=:black) for k in 1:size(r,1)])
  pretty_table(r, nosubheader=true, crop=:none,
               highlighters=hs, alignment=:c)
end

# pretty_table(df, backend=:latex, tf=latex_simple,
#              nosubheader=true, formatters=ft_round(2,3:6))
