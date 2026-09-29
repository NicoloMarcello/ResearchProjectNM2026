
# import toolboxes used for data visualization

import mne
import numpy as np
import scipy.io as sp

# load the data from the .mat file

data = sp.loadmat('exp_BN49.mat')
print(data.keys())                        # Print the keys to understand the structure of the loaded data
print('hello')
