#!/usr/bin/env python3
"""Plot mean MEG sensor correlations and phase slopes."""

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


# Ask for one participant's epoch file. The channel locations are consistent
# across participants, so this file is used to supply the sensor metadata.
sub_ID = input("subject ID: ").strip()

# Behavioral and MEG files follow the directory and naming conventions used in
# MEG_powerplot_subject.ipynb. Single-digit subjects have a zero-padded folder
# name, while the FIF filename uses the unpadded subject number.
subject_folder = "//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01pz-dbush99/Katja Button Press Data"
if sub_ID.startswith("0"):
    spliced_id = sub_ID[1:]
    MEG_file = "/s0" + spliced_id + "/Subject" + spliced_id + "_spm_mne-epo"
else:
    MEG_file = "/s" + sub_ID + "/Subject" + sub_ID + "_spm_mne-epo"
epochs_file = subject_folder + MEG_file + ".fif"

# Load the epochs metadata. The sensor locations are stored in epochs.info, so
# the epoch samples do not need to be loaded for these topographic plots.
epochs = mne.read_epochs(epochs_file, preload=False)

# Collect the MEG sensor names in the same order as they appear in the epochs.
# These names are used to align each CSV value with its sensor location.
MEG_sens = []
for channel in epochs.info["ch_names"]:
    if channel.startswith("M"):
        MEG_sens.append(channel)

# Keep the sensor locations from the participant's epochs file.
meg_picks = mne.pick_channels(epochs.info["ch_names"], include=MEG_sens)
group_info = mne.pick_info(epochs.info, meg_picks, copy=True)

# The results CSV has one sensor per row. Set the sensor name as the index, then
# reorder its rows to match MEG_sens so values cannot be plotted at the wrong site.
group_corr_results = pd.read_csv("group_corr_results.csv")
group_corr_results = group_corr_results.set_index("sensor").loc[MEG_sens]

# Plot both metrics stored in the CSV. Each map uses a symmetric color scale so
# positive and negative values are shown with equal strength around zero.
metrics_to_plot = [
    ("mean_corr", "Mean sensor correlation", "Mean correlation (r)"),
    (
        "mean_slope_rad_per_press",
        "Mean sensor slope per press",
        "Slope (rad/press)",
    ),
]

# Identify the MEG channel type for each sensor. The notebooks plot each type
# separately so MNE can use the correct sensor positions for each topomap.
sensor_types = [
    mne.channel_type(group_info, index)
    for index in range(len(MEG_sens))
]

for metric, title, colorbar_label in metrics_to_plot:
    metric_values = group_corr_results[metric].to_numpy()
    color_limit = np.max(np.abs(metric_values))

    for sensor_type in sorted(set(sensor_types)):
        # Select this sensor type's values and matching location metadata.
        picks = np.array(
            [
                index
                for index, kind in enumerate(sensor_types)
                if kind == sensor_type
            ]
        )
        sensor_info = mne.pick_info(group_info, picks, copy=True)

        # Draw the topomap using the same size, diverging colormap, and contour
        # setting as MEG_powerplot_all.ipynb.
        fig, ax = plt.subplots(figsize=(7, 6))
        image, _ = mne.viz.plot_topomap(
            metric_values[picks],
            sensor_info,
            ch_type=sensor_type,
            axes=ax,
            cmap="RdBu_r",
            contours=0,
            vlim=(-color_limit, color_limit),
            show=False,
        )
        ax.set_title(title + " (" + sensor_type + ")")
        fig.colorbar(image, ax=ax, label=colorbar_label)

plt.show()
