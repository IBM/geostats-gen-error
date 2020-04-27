# Overview

This repository contains the experiments of the paper
*Hoffimann et. al. 2020. Generalization error of learning models under covariate shift and spatial correlation*.

# Instructions

After cloning the repository, you can either run the scripts to reproduce the results in your local machine,
or plot the results that were saved in the repository by the authors.

## Running

To reproduce the environment, please go to the root folder and run the following commands:

```julia
push!(empty!(LOAD_PATH), @__DIR__)
using Pkg; Pkg.instantiate()
```

After the environment is instantiated, you can run the experiments by including them in the session:

```julia
include("gaussian.jl")   # experiment 1
include("newzealand.jl") # experiment 2
```

If you are only interested in inspecting the results, please check the `results` folder in this repository.

## Plotting

We provide a separate environment for plotting the results. Please start a session in the `results` folder,
activate the environment with the same commands shown in the previous section, and include the corresponding
plotting scripts.
