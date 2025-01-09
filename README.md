# FLTM-JM: Functional Latent Trait Model-Joint Modeling


# Introduction

FLTM-JM is a joint modeling framework for multivariate longitudinal
ordinal variables and time-to-event data with right censoring. A
Bayesian framework is utilized for parameter estimation and dynamic
prediction. The posterior sampling is done by *Stan* in *R*, utilizing
the package *rstan*. This vignette illustrates implementation of FLTM-JM
on a simulated data.
