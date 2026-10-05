# import toolboxes used for data visualization
import mne
import numpy as np
import scipy.io as sp

# load the Epoched data from the .fif file
subject_folder = '//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01pz-dbush99/Katja Button Press Data/s05/'
subject_filename = 'Subject5_spm_mne'
filepath = subject_folder + subject_filename + '.fif'

# read data file into MNE Epochs object 'Epoch_data'
Epochs_data = mne.read_epochs(filepath,preload=True)

# print data and info about the Epoched data
print(f"Epochs data shape: {Epochs_data.get_data().shape}")
print(Epochs_data.info)

Epochs_data.plot(n_epochs=1,n_channels=4,block=True,scalings=1e3)
