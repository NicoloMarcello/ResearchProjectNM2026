"""Shared topomap plotting for sensor-wise statistical results."""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


def plot_sensor_test_topomaps(
    results,
    sensor_info,
    output_base,
    mean_column,
    colorbar_label,
    title_prefix,
    p_value_column="p_value",
    threshold=0.01,
):
    """Plot magnetometer and gradiometer means, marking selected p-values."""
    output_base = Path(output_base)
    aligned = results.set_index("sensor").loc[sensor_info["ch_names"]]
    values = aligned[mean_column].to_numpy(dtype=float)
    p_values = aligned[p_value_column].to_numpy(dtype=float)
    marked = np.isfinite(p_values) & (p_values < threshold)
    sensor_types = [
        mne.channel_type(sensor_info, index)
        for index in range(len(sensor_info["ch_names"]))
    ]

    for sensor_type in ("mag", "grad"):
        picks = np.asarray([
            index for index, kind in enumerate(sensor_types)
            if kind == sensor_type
        ], dtype=int)
        if not len(picks):
            continue

        type_info = mne.pick_info(sensor_info, picks, copy=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        image, _ = mne.viz.plot_topomap(
            values[picks],
            type_info,
            ch_type=sensor_type,
            axes=ax,
            cmap="RdBu_r",
            contours=0,
            mask=marked[picks],
            mask_params={
                "marker": "o",
                "markerfacecolor": "yellow",
                "markeredgecolor": "black",
                "linestyle": "None",
                "linewidth": 0.7,
                "markersize": 7,
            },
            show=False,
        )
        ax.set_title(
            title_prefix + "; circles mark raw p < "
            + str(threshold) + " (" + sensor_type + ")"
        )
        fig.colorbar(image, ax=ax, label=colorbar_label)
        fig.tight_layout()
        plot_file = output_base.with_name(
            output_base.stem + "_" + sensor_type + output_base.suffix
        )
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved scalp map to:", plot_file)
