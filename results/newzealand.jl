# instantiate environment
using Pkg; Pkg.instantiate()

using Gadfly
using CSV

# -------------
# MAIN SCRIPT
# -------------

# load results table
df = CSV.read("newzealand.csv", missingstring="NaN")
df = dropmissing(df)

# set plotting theme
theme = Gadfly.get_theme(Val(:dark))
Gadfly.push_theme(theme)
theme = style(point_size=2.5px,
              key_position=:top,
              default_color=colorant"black")
colors = ("#1b9e77","#7570b3","#d95f02")

Gadfly.with_theme(theme) do
    ycols = (:CV,:BCV,:DRV,:ACTUAL)
    plot(df, x=:k, y=Col.value(ycols...),
         xgroup=Col.index(ycols...),
         Geom.subplot_grid(Geom.boxplot))
end

Gadfly.with_theme(theme) do
    ycols = (:CV,:BCV,:DRV)
    plot(df, x=:ACTUAL, y=Col.value(ycols...), Geom.point)
end

# pretty_table(df, backend=:latex, tf=latex_simple,
             # nosubheader=true, formatters=ft_round(2,3:6))
