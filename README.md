# Shark-Teeth-Raman-Analysis
Jupyter Notebook for processing and analysis of Raman spectra of shark/ray teeth.

The Notebook imports Raman spectra, does automatic baseline subtraction, background artefact subtraction, peak detection, peak fitting, and figure generation. It then saves processed data to a standardised data file format.

Written by Dr Joseph Razzell Hollis on 2025-03-10. This Notebook uses OSTRI v0.1.2-alpha, a system of standardised python classes and functions developed by Dr Joseph Razzell Hollis for handling Raman & FTIR spectra in Python. See www.github.com/Jobium/OSTRI/ for updates and documentation.

Any updates to this Notebook will be made available online at www.github.com/Jobium/Shark-Teeth-Raman-Analysis/

Python code requires Python 3.7 (or higher) and the following packages with minimum versions:
- numpy 1.19.5
- pandas 0.22.0
- matplotlib 2.1.2
- lmfit 1.0.3
- scipy 1.5.4
- scikit-learn 0.24.2
