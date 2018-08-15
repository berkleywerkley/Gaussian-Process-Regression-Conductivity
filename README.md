# Gaussian-Process-Regression-Conductivity
Part of Work at MIT

## get_data.py
Function to collect data for gp.py
returns dictionary with key as tuple of eps24 and radius, and value as conductivity

## gp.py
Uses scikit-learn to perform Gaussian Process Regression on a parameter and conductivity data to suggest a new set of parameters to test
