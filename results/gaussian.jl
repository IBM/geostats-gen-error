# instantiate environment
using Pkg
push!(empty!(LOAD_PATH), @__DIR__)
Pkg.instantiate()

using Gadfly
using DataFrames
using CSV

# intersection between two circles
function intersection(c₁, r₁, c₂, r₂)
  # squared radii
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
kldiv(δ, τ) = (1 / 2) * log(τ^2) + (δ^2 - τ + 1) / τ

function jaccard(c₁, r₁, c₂, r₂)
  # squared radii
  a = r₁ * r₁
  b = r₂ * r₂

  d = sqrt(2 * (c₁ - c₂) * (c₁ - c₂))

  I = intersection(c₁, r₁, c₂, r₂)
  U = π * a + π * b - I

  1 - I / U
end

jaccard(δ, τ) = jaccard(0.0, 3.0, 3*√2δ, 3τ)

function areashift(c₁, r₁, c₂, r₂)
  # squared radii
  a = r₁ * r₁
  b = r₂ * r₂

  d = sqrt(2 * (c₁ - c₂) * (c₁ - c₂))

  I = intersection(c₁, r₁, c₂, r₂)

  # distance size
  D = (π * d * d) / (π * 4a)

  # outside vs. inside
  S = (π * b - 2I) / (π * a)

  (S + D + 1) / 3
end

areashift(δ, τ) = areashift(0.0, 3.0, 3*√2δ, 3τ)

# covariate shift configuration
shiftconfig(δ, τ) = 2*√2*δ ≤ 1 - τ ? "inside" : 2*√2*δ ≥ 1 + τ ? "outside" : "neither"

# -------------
# MAIN SCRIPT
# -------------

# load results table
df = CSV.read("gaussian.csv", missingstring="NaN")
df = dropmissing(df)

# difference with actual error
df[!,:δCV]  = df[!,:CV]  - df[!,:ACTUAL]
df[!,:δBCV] = df[!,:BCV] - df[!,:ACTUAL]
df[!,:δDRV] = df[!,:DRV] - df[!,:ACTUAL]

# shift measures
for measure in [kldiv, jaccard, areashift]
  df[!,Symbol(measure)] = measure.(df[!,:δ], df[!,:τ])
end

# shift configuration
df[!,:config] = shiftconfig.(df[!,:δ], df[!,:τ])

# correlation length as factor
df[!,:rfactor] = "r=".*string.(df[!,:r])

# set plotting theme
theme = Gadfly.get_theme(Val(:dark))
Gadfly.push_theme(theme)
theme = style(point_size=2px, key_position=:top)
gcolors = ("#1b9e77","#7570b3","#d95f02")

# generalization error vs. shift function
Gadfly.with_theme(theme) do
  xcols = (:kldiv,:jaccard,:areashift)
  set_default_plot_size(28cm, 10cm)
  plot(df, x=Col.value(xcols...), y=:ACTUAL, ygroup=:MODEL,
       color=:config, xgroup=Col.index(xcols...),
       Guide.xlabel("Shift function"),
       Guide.ylabel("Generalization error"),
       Guide.title("Error vs. shift function"),
       Guide.colorkey(title="Configuration"),
       Scale.color_discrete_manual(gcolors...),
       Geom.subplot_grid(Geom.point, free_x_axis=true))
end

# generalization error by different methods
Gadfly.with_theme(theme) do
  ycols = (:CV,:BCV,:DRV,:ACTUAL)
  set_default_plot_size(28cm, 20cm)
  p1 = plot(df, x=:areashift, y=Col.value(ycols...),
       xgroup=Col.index(ycols...), color=:rfactor,
       Guide.xlabel("Covariate shift"),
       Guide.ylabel("Generalization error"),
       Guide.title("Error vs. covariate shift by methods"),
       Guide.colorkey(title="Correlation length"),
       Scale.color_discrete_manual(gcolors...),
       Geom.subplot_grid(Geom.point))
  p2 = plot(df, x=:rfactor, y=Col.value(ycols...),
            xgroup=Col.index(ycols...), color=:rfactor,
            Guide.xlabel("Correlation length"),
            Guide.ylabel("Generalization error"),
            Guide.title("Error vs. correlation length by methods"),
            Guide.colorkey(title="Correlation length"),
            Scale.color_discrete_manual(gcolors...),
            Geom.subplot_grid(Geom.boxplot))
  vstack(p1, p2)
end

# generalization error by different correlation lengths
Gadfly.with_theme(theme) do
  set_default_plot_size(28cm, 10cm)
  ycols = (:CV,:BCV,:DRV,:ACTUAL)
  plot(df, x=:areashift, y=Col.value(ycols...),
       xgroup=:rfactor, color=Col.index(ycols...),
       Guide.xlabel("Covariate shift"),
       Guide.ylabel("Generalization error"),
       Guide.title("Error vs. covariate shift by correlation lengths"),
       Guide.colorkey(title="Method"),
       Geom.subplot_grid(layer(Geom.line, Stat.smooth)))
end
