# Overview

This repository contains the experiments of the paper
*Hoffimann et. al. 2020. Generalization error of learning
models under covariate shift and spatial correlation*.

# Instructions

After cloning the repository, you can either run the scripts
to reproduce the results in your machine, or plot the results
that were saved in the repository by the authors.

## Running

Please make sure that you have Julia v1.4 or a newer version.

From the root folder of the project:

### Gaussian experiment

```bash
shell> julia --project gaussian.jl
```

### New Zealand experiment

```shell
shell> julia --project newzealand.jl
```

## Plotting

From the `results` folder of the project:

```shell
shell> julia --project gaussian.jl
shell> julia --project newzealand.jl
```
