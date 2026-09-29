
# import toolboxes used for data visualization
import mne
import numpy as np
import scipy.io as sp



data = sp.loadmat('single subject data/Kornysheva_etal_Neuron_2019_s16_ICA_corrected.mat')           # load the data from the .mat file
print(data.keys())                                                                               # print the keys of the loaded data to understand its structure
print(data['D'].shape)                                                                     # print the shape of the 'data' array to understand its dimensions
EEG_data = data['D']                                                       # extract the EEG data from the loaded data


raw = mne.io.RawArray(EEG_data)
