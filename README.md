# Robust and Conjugate Gaussian Process Regression

This repository contains all code and data needed to reproduce the results in the paper "Robust and Conjugate Gaussian Process Regression". 

## Reproducing experiments

- The folder `experiments` contains notebooks to recreate all the experiments.
- The folder `rcgp` contains RCGP implementation in gpflow.

To run benchmark experiments use:

```python 
python -m experiments.uci.run full experiments/uci/experiment_results
```
