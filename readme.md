# Overview

1. Download the data from https://www.icpsr.umich.edu/icpsrweb/ICPSR/studies/31061#
2. Run ``python3 fitmodels.py --help`` to see the options for fitting LASSO models. 
3. Then you can run ``python3 output.py`` to generate all the final results.

To fully replicate the paper, this entails running:
``python3 fitmodels.py --fitprop 1``
``python3 fitmodels.py --bootstrap True``
``python3 fitmodels.py --fitprop 0.65``
``python3 output.py``