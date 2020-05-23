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

# index of source, target, and error estimate columns
serror, terror, errors = 2, 3, 4:6

# enable/disable LaTeX output
print_latex = true

# highlight table entries
function best(d, i, j)
  k = argmin([abs(d[i,k] - d[i,terror]) for k in errors])
  j ∈ errors && errors[k] == j
end
function worst(d, i, j)
  k = argmax([abs(d[i,k] - d[i,terror]) for k in errors])
  j ∈ errors && errors[k] == j
end
function target(d, i, j)
  j == terror
end
hb  = Highlighter(best, bold=true, foreground=:blue)
hw  = Highlighter(worst, bold=true, foreground=:red)
ht  = Highlighter(target, bold=true)
hbl = LatexHighlighter(best, ["color{NavyBlue}","textbf"])
hwl = LatexHighlighter(worst, ["color{Maroon}","textbf"])
htl = LatexHighlighter(target, ["textbf"])

for g in groupby(df, :SHIFT)
  # shift and number of models
  s = g[1,:SHIFT] == "YES" ? "onoff" : "noshift"
  n = size(g, 1)

  # drop shift column
  g = g[!,Not(:SHIFT)]

  pretty_table(g, nosubheader=true, crop=:none, vlines=[0,1,3,6],
               alignment=[:c,:r,:r,:r,:r,:r],
               formatters=ft_round(3, serror ∪ terror ∪ errors),
               highlighters=(hb, hw, ht))

  if print_latex
    open("newzealand-$s.tex", "w") do io
      pretty_table(io, g, backend=:latex, tf=latex_simple,
                   nosubheader=true, alignment=:c, vlines=[1,3],
                   formatters=ft_round(3, serror ∪ terror ∪ errors),
                   highlighters=(hbl, hwl, htl))
    end
  end

  # model ranking based on each method
  ranks = map([:TARGET,:CV,:BCV,:DRV]) do err
    r = sortperm(g[!,err])
    Symbol(err," RANK") => g[!,:MODEL][r]
  end
  r = DataFrame(ranks)

  colorval(k) = color(k, n)
  hs = Tuple([Highlighter((d, i, j) -> d[i,j] == r[k,Symbol("TARGET RANK")],
             background=colorval(k), foreground=:black) for k in 1:n])
  pretty_table(r, nosubheader=true, crop=:none, alignment=:c, highlighters=hs)

  if print_latex
    colorstr(k) = "cellcolor[RGB]{" * join(color(k, n), ",") * "}"
    hs = Tuple([LatexHighlighter((d, i, j) -> d[i,j] == r[k,Symbol("TARGET RANK")],
               [colorstr(k), "color{white}", "textbf"]) for k in 1:n])
    open("newzealand-rank-$s.tex", "w") do io
      pretty_table(io, r, backend=:latex, tf=latex_simple,
                   nosubheader=true, alignment=:c, highlighters=hs)
    end
  end
end
