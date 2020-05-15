# instantiate environment
using Pkg; Pkg.instantiate()

using Gadfly
using DataFrames
using Printf
using CSV

# intersection between two circles
function intersection(c₁, r₁, c₂, r₂)
  a = r₁ * r₁
  b = r₂ * r₂
  d = sqrt(2 * (c₁ - c₂) * (c₁ - c₂))

  # trivial cases
  d ≥ r₁ + r₂      && return 0.0
  d ≤ abs(r₂ - r₁) && return π * min(a, b)

  # general case
  C1 = a * acos((d * d + a - b) / (2d*r₁))
  C2 = b * acos((d * d + b - a) / (2d*r₂))
  C3 = (1 / 2) * sqrt((-d + r₂ + r₁) * (d + r₂ - r₁) * (d - r₂ + r₁) * (d + r₂ + r₁))

  C1 + C2 - C3
end

# -----------------
# SHIFT FUNCTIONS
# -----------------
kldiv(δ, τ) = δ^2 + τ^2 - log(τ^4) - 1

function jaccard(c₁, r₁, c₂, r₂)
  I = intersection(c₁, r₁, c₂, r₂)
  U = π * r₁^2 + π * r₂^2 - I

  1 - I / U
end

jaccard(δ, τ) = jaccard(0, 3, 3*√2δ + 0, 3τ)

function novelty(c₁, r₁, c₂, r₂)
  I = intersection(c₁, r₁, c₂, r₂)
  O = π * r₂^2 - I
  B = π * r₂^2
  R = (O - I) / B

  (R + 1) / 2
end

novelty(δ, τ) = novelty(0, 3, 3*√2δ + 0, 3τ)

# covariate shift configuration
shiftconfig(δ, τ) = 2δ ≤ 1 - τ ?
                    "inside" :
                    2δ ≥ 1 + τ ?
                    "outside" :
                    "partial"

# -------------
# MAIN SCRIPT
# -------------

# load results table
df = CSV.read(joinpath(@__DIR__,"gaussian.csv"), missingstring="NaN")

# drop outliers due to numerical instability
df = filter(row -> row[:DRV] ≤ 0.5 && row[:ACTUAL] ≤ 0.5, df)
df = dropmissing(df)


# shift functions
df[!,:KLDivergence]    = kldiv.(df[!,:δ], df[!,:τ])
df[!,:JaccardDistance] = jaccard.(df[!,:δ], df[!,:τ])
df[!,:NoveltyFactor]   = novelty.(df[!,:δ], df[!,:τ])

# shift configuration
df[!,:config] = shiftconfig.(df[!,:δ], df[!,:τ])

# correlation length as factor
df[!,:rfactor] = map(df[!,:r]) do r
  @sprintf "r=%.1f" r
end

# filter "inside" configuration
ff = filter(row -> row[:config] == "inside", df)

# set plotting theme
theme = Gadfly.get_theme(Val(:default))
Gadfly.push_theme(theme)
theme = style(point_size=2.5px,
              key_position=:top,
              default_color=colorant"black")
colors = ("#1b9e77","#7570b3","#d95f02")

# generalization error vs. shift function
Gadfly.with_theme(theme) do
  xcols = (:KLDivergence,:JaccardDistance,:NoveltyFactor)
  set_default_plot_size(24cm, 14cm)
  p = plot(df, x=Col.value(xcols...), y=:ACTUAL,
       xgroup=Col.index(xcols...), ygroup=:MODEL,
       color=:config,
       Guide.xlabel("Covariate shift"),
       Guide.ylabel("Error by models"),
       Guide.title("Error vs. covariate shift"),
       Guide.colorkey(title="Configuration"),
       Scale.color_discrete_manual(colors...),
       Geom.subplot_grid(layer(Geom.point), free_x_axis=true,
                         layer(yintercept=[0.0], Geom.hline(color="gray", style=:dash)),
                         layer(yintercept=[0.5], Geom.hline(color="gray", style=:dash)),
                         Coord.cartesian(xmin=0.0,ymax=0.5),
                         Guide.ylabel(orientation=:vertical)))
  p |> SVG(joinpath(@__DIR__,"gaussian-plot1.svg"))
  p
end

# generalization error by different methods
Gadfly.with_theme(theme) do
  ycols = (:CV,:BCV,:DRV,:ACTUAL)
  set_default_plot_size(24cm, 18cm)
  p1 = plot(df, x=:NoveltyFactor, y=Col.value(ycols...),
       xgroup=Col.index(ycols...), color=:rfactor,
       Guide.xlabel("Covariate shift by methods"),
       Guide.ylabel("Error"),
       Guide.title("Error vs. covariate shift"),
       Guide.colorkey(title="Correlation length"),
       Scale.color_discrete_manual(colors...),
       Geom.subplot_grid(layer(Geom.point),
                         layer(yintercept=[0.0], Geom.hline(color="gray", style=:dash)),
                         layer(yintercept=[0.5], Geom.hline(color="gray", style=:dash)),
                         Coord.cartesian(xmin=0.0,xmax=1.0,ymax=0.5)))
  p2 = plot(ff, x=:rfactor, y=Col.value(ycols...),
            xgroup=Col.index(ycols...), color=:rfactor,
            Guide.xlabel("Correlation length by methods"),
            Guide.ylabel("Error"),
            Guide.title("Error vs. correlation length for inside configuration"),
            Guide.colorkey(title="Correlation length"),
            Scale.color_discrete_manual(colors...),
            Geom.subplot_grid(Geom.boxplot))
  p = vstack(p1, p2)
  p |> SVG(joinpath(@__DIR__,"gaussian-plot2.svg"))
  p
end

# generalization error by different correlation lengths
theme2 = style(point_size=1.8px,
               key_position=:top,
               default_color=colorant"black")
Gadfly.with_theme(theme2) do
  set_default_plot_size(24cm, 18cm)
  ycols = (:CV,:BCV,:DRV,:ACTUAL)
  p1 = plot(df, x=:NoveltyFactor, y=Col.value(ycols...),
       xgroup=:rfactor, color=Col.index(ycols...),
       Guide.xlabel("Covariate shift by correlation length"),
       Guide.ylabel("Error"),
       Guide.title("Error vs. covariate shift"),
       Guide.colorkey(title="Method"),
       Scale.color_discrete_manual(colors...),
       Geom.subplot_grid(layer(Geom.point), layer(Geom.line, Stat.smooth(smoothing=1.0)),
                         layer(yintercept=[0.0], Geom.hline(color="gray", style=:dash)),
                         layer(yintercept=[0.5], Geom.hline(color="gray", style=:dash)),
                         Coord.cartesian(xmin=0.0,xmax=1.0,ymax=0.5)))
  xcols = (:CV,:BCV,:DRV)
  p2 = plot(ff, x=Col.value(xcols...), y=:ACTUAL, xgroup=Col.index(xcols...),
            Guide.xlabel("Estimated Error"), Guide.ylabel("Actual Error"),
            Guide.title("Q-Q plot for inside configuration"),
            Geom.subplot_grid(layer(Geom.point,Stat.qq),
                              layer(Geom.abline(color="grey",style=:dot))))
  p = vstack(p1, p2)
  p |> SVG(joinpath(@__DIR__,"gaussian-plot3.svg"))
  p
end
