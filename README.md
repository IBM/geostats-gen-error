# Overview

This repository contains the experiments of the paper:

*Hoffimann et. al. 2020. Geostatistical learning:
challenges and opportunities*.

# Instructions

After cloning the repository, you can either run the scripts
to reproduce the results in your machine, or plot the results
that were saved in the repository by the authors.

Please make sure that you have Julia v1.4 or a newer version of the language installed.

## Running

From the root folder of the project:

```bash
# Gaussian experiment
$ julia --project gaussian.jl

# New Zealand experiment
$ julia --project newzealand.jl
```

## Plotting

Run the same commands but from the `results` folder.

## Results

### Gaussian

![gaussian-plot1](results/gaussian-plot1.svg)
![gaussian-plot1](results/gaussian-plot2.svg)
![gaussian-plot1](results/gaussian-plot3.svg)

### New Zealand

#### Onshore -> Offshore

![newzealand-onoff](results/newzealand-onoff.png)

#### No covariate shift

![newzealand-noshift](results/newzealand-noshift.png)
