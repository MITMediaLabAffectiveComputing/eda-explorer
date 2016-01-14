eda-explorer
============

Scripts to detect artifacts in EDA data


Version 0.4

Please also cite this project:
Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R. Automatic identification of artifacts in electrodermal activity data. In Engineering in Medicine and Biology Conference. 2015.


Required python packages: 
===
- numpy: 1.9.2 
- scipy: 0.14.0 
- pandas: 0.16.0
- sklearn: 0.16.1
- pickle 
- matplotlib: 1.3.1 
- imp
- PyWavelets: 0.2.2
- os

To run artifact detection from the command line:
==
python EDA-Artifact-Detection-Script.py

Note that PickleDirectory is the main directory

Currently there are only 2 classifiers to choose from: Binary or Multiclass

To run peak detection:
==
python EDA-Peak-Detection-Script.py

Descriptions of the algorithm settings can be found at http://eda-explorer.media.mit.edu/info/

Notes:
===

1. Currently, these files are written with the assumption that the sample rate is an integer power of 2. 

2. Keep the "classify.py" and "SVMBinary.p" and "SVMMulticlass.p" files in the same directory.

3. Please visit eda.explorer.media.mit.edu to use the web-based version
