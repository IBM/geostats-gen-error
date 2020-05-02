# instantiate environment
using Pkg
push!(empty!(LOAD_PATH), @__DIR__)
Pkg.instantiate()

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
  A1 = b * acos((d * d + b - a) / (2d*r₂))
  A2 = a * acos((d * d + a - b) / (2d*r₁))
  A3 = (1 / 2) * sqrt((-d + r₂ + r₁) * (d + r₂ - r₁) * (d - r₂ + r₁) * (d + r₂ + r₁))

  A1 + A2 - A3
end

# -----------------
# SHIFT FUNCTIONS
# -----------------
kldiv(δ, τ) = δ^2 + τ^2 - log(τ^4) - 1

function jaccard(c₁, r₁, c₂, r₂)
  a = r₁ * r₁
  b = r₂ * r₂

  I = intersection(c₁, r₁, c₂, r₂)
  U = π * a + π * b - I

  1 - I / U
end

jaccard(δ, τ) = jaccard(0, 3, 3*√2δ + 0, 3τ)

function novelty(c₁, r₁, c₂, r₂)
  a = r₁ * r₁
  b = r₂ * r₂

  # (outside - inside) / source
  I = intersection(c₁, r₁, c₂, r₂)
  O = π * b - I
  S = π * a
  R = (O - I) / S

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
df = CSV.read("gaussian.csv", missingstring="NaN")

# drop outliers due to numerical instability
df = filter(row -> row[:DRV] ≤ 0.5, df)
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
       Guide.ylabel("Error"),
       Guide.title("Error vs. covariate shift"),
       Guide.colorkey(title="Configuration"),
       Scale.color_discrete_manual(colors...),
       Geom.subplot_grid(layer(Geom.point), free_x_axis=true,
                         layer(yintercept=[0.0], Geom.hline(color=colors[1], style=:dash)),
                         layer(yintercept=[0.5], Geom.hline(color=colors[3], style=:dash)),
                         Coord.cartesian(ymax=0.5),
                         Guide.ylabel(orientation=:vertical)))
  p |> SVG("gaussian-plot1.svg")
  p
end

# generalization error by different methods
Gadfly.with_theme(theme) do
  ycols = (:CV,:BCV,:DRV,:ACTUAL)
  set_default_plot_size(24cm, 18cm)
  p1 = plot(df, x=:NoveltyFactor, y=Col.value(ycols...),
       xgroup=Col.index(ycols...), color=:rfactor,
       Guide.xlabel("Covariate shift"),
       Guide.ylabel("Error"),
       Guide.title("Error vs. covariate shift by methods"),
       Guide.colorkey(title="Correlation length"),
       Scale.color_discrete_manual(colors...),
       Geom.subplot_grid(layer(Geom.point),
                         layer(yintercept=[0.0], Geom.hline(color="gray", style=:dash)),
                         layer(yintercept=[0.5], Geom.hline(color="gray", style=:dash)),
                         Coord.cartesian(xmax=0.8, ymax=0.5)))
  p2 = plot(df, x=:rfactor, y=Col.value(ycols...),
            xgroup=Col.index(ycols...), color=:rfactor,
            Guide.xlabel("Correlation length"),
            Guide.ylabel("Error"),
            Guide.title("Error vs. correlation length by methods"),
            Guide.colorkey(title="Correlation length"),
            Scale.color_discrete_manual(colors...),
            Geom.subplot_grid(Geom.boxplot))
  p = vstack(p1, p2)
  p |> SVG("gaussian-plot2.svg")
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
       Guide.xlabel("Covariate shift"),
       Guide.ylabel("Error"),
       Guide.title("Error vs. covariate shift by correlation lengths"),
       Guide.colorkey(title="Method"),
       Scale.color_discrete_manual(colors...),
       Geom.subplot_grid(layer(Geom.point), layer(Geom.line, Stat.smooth),
                         layer(yintercept=[0.0], Geom.hline(color="gray", style=:dash)),
                         layer(yintercept=[0.5], Geom.hline(color="gray", style=:dash)),
                         Coord.cartesian(xmax=0.8,ymax=0.5)))
  xcols = (:CV,:BCV,:DRV)
  ff = filter(row -> row[:config] == "inside", df)
  p2 = plot(ff, x=Col.value(xcols...), y=:ACTUAL, xgroup=Col.index(xcols...),
            Guide.xlabel("Method"), Guide.ylabel("ACTUAL"),
            Guide.title("Q-Q plot by methods for inside configuration"),
            Geom.subplot_grid(layer(Geom.point,Stat.qq),
                              layer(Geom.abline(color="grey",style=:dot))))
  p = vstack(p1, p2)
  p |> SVG("gaussian-plot3.svg")
  p
end
