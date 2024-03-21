# Wasserstein Geodesic Interpolation for Gradual Domain Adaptation.

## Installation and Dependencies

**GRADOT** is based on `NumPy`, `Pandas`, `Scikit-Learn` and `POT`.
So, make sure these packages are installed. For example, you can install them with `pip`:

```
pip3 install numpy pandas POT
```

All the codes for the article are tested on macOS 14.2.1


## Scripts for experiments:
To reproduce the results, the following files are provided:

 1 - `Dataset.py` to generate Moons dataset (source and rotated target)

 2 - `WassersteinGeodesic.py` implmentation of displacement interpolation between source and target domains

 3 - `GRADOT.py` to generate intermediate domains based on `WassersteinGeodesic`

 4 - `GST_GRADOT.py` to perform Gradual Domain Adaptation upon generated intermediate domains.  


 ## Scripts for figures:

● Intermediate domains for Moons datasets - `GST_GRADOT.py`

