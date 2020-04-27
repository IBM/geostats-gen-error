using Gadfly

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

function jaccard(c₁, r₁, c₂, r₂)
  # squared radii
  a = r₁ * r₁
  b = r₂ * r₂

  d = sqrt(2 * (c₁ - c₂) * (c₁ - c₂))

  I = intersection(c₁, r₁, c₂, r₂)
  U = π * a + π * b - I

  1 - I / U
end

kldiv(δ, τ) = (1 / 2) * log(τ^2) + (δ^2 - τ + 1) / τ

jaccard(δ, τ) = jaccard(0.0, 3.0, 3*√2δ, 3τ)

areashift(δ, τ) = areashift(0.0, 3.0, 3*√2δ, 3τ)

shiftconfig(δ, τ) = 2*√2*δ ≤ 1 - τ ? "inside" : 2*√2*δ ≥ 1 + τ ? "outside" : "neither"
