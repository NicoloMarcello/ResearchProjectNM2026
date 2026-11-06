#!/usr/bin/env python3
"""Plot sensor-wise correlation and slope metrics from the repository CSV.

The CSV is a table with one sensor per row and a ``sensor`` column for channel
names. This script plots ``mean_corr`` and ``mean_slope_rad_per_press``.

The epoch file is used only for its channel metadata and sensor locations.
Its file naming follows MEG_powerplot_subject.ipynb and MEG_powerplot_all.ipynb.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


DEFAULT_DATA_ROOT = Path(
    "//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01pz-dbush99/Katja Button Press Data"
)
MEG_CHANNEL_TYPES = {"mag", "grad", "planar1", "planar2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot sensor metrics from the repository CSV using locations from "
            "one participant's MNE epochs file."
        )
    )
    parser.add_argument(
        "subject_id",
        nargs="?",
        help="participant ID, for example 05 or 16 (prompted if omitted)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="root of the MEG data share (default follows the notebooks)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="sensor-by-row CSV; by default searches beside this script",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="optional folder in which to save the scalp-map PNG files",
    )
    parser.add_argument(
        "--show-names",
        action="store_true",
        help="label the sensors on each topomap",
    )
    return parser.parse_args()


def make_epochs_path(data_root: Path, subject_id: str) -> Path:
    subject_id = subject_id.strip()
    if not subject_id.isdigit() or int(subject_id) < 1:
        raise ValueError("Subject ID must be a positive number, such as 05 or 16.")

    number = str(int(subject_id))
    folder_id = number.zfill(2)
    return data_root / f"s{folder_id}" / f"Subject{number}_spm_mne-epo.fif"


def find_csv(csv_arg: Path | None) -> Path:
    if csv_arg is not None:
        candidates = [csv_arg.expanduser()]
        if not csv_arg.suffix:
            candidates.append(csv_arg.with_name(csv_arg.name + ".csv").expanduser())
    else:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir / "group_corr_results.csv",
            script_dir / "group_corr_results",
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the correlation CSV. Searched:\n  "
        + searched
        + "\nPass its location with --csv."
    )


def read_metrics(csv_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(csv_file)
            if any(cell.strip() for cell in row)
        ]

    if not rows:
        raise ValueError(f"The results CSV is empty: {csv_path}")

    headers = [header.casefold() for header in rows[0]]
    required_columns = ["sensor", "mean_corr", "mean_slope_rad_per_press"]
    missing_columns = [column for column in required_columns if column not in headers]
    if missing_columns:
        raise ValueError(
            f"The results CSV is missing required column(s) {missing_columns}. "
            f"Found columns: {rows[0]}"
        )

    sensor_index = headers.index("sensor")
    metric_indices = {
        "mean_corr": headers.index("mean_corr"),
        "mean_slope_rad_per_press": headers.index("mean_slope_rad_per_press"),
    }
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"The results CSV has no sensor rows: {csv_path}")
    highest_index = max([sensor_index, *metric_indices.values()])
    if any(len(row) <= highest_index for row in data_rows):
        raise ValueError("A CSV data row is missing a required sensor metric.")

    names = [row[sensor_index] for row in data_rows]
    if any(not name for name in names):
        raise ValueError("The CSV contains a blank channel name.")
    if len(set(names)) != len(names):
        raise ValueError("The CSV contains duplicate channel names.")

    metrics = {}
    for metric, column_index in metric_indices.items():
        try:
            values = np.asarray(
                [
                    float(row[column_index]) if row[column_index] else np.nan
                    for row in data_rows
                ],
                dtype=float,
            )
        except ValueError as exc:
            raise ValueError(f"Values in the {metric!r} column must be numeric.") from exc
        metrics[metric] = values

    finite_corr = np.isfinite(metrics["mean_corr"])
    if np.any(finite_corr & ((metrics["mean_corr"] < -1.0) | (metrics["mean_corr"] > 1.0))):
        raise ValueError("Finite correlation coefficients must be between -1 and 1.")

    return names, metrics


def plot_metrics(
    epochs: mne.Epochs,
    csv_names: list[str],
    metric_values: dict[str, np.ndarray],
    subject_id: str,
    output_dir: Path | None,
    show_names: bool,
) -> None:
    # Match the sensor selection used in the notebooks: MEG channel names begin
    # with M. Channel types and positions come from the participant's epochs.
    channel_indices = {
        name: index for index, name in enumerate(epochs.info["ch_names"])
    }
    bads = set(epochs.info.get("bads", []))
    meg_names = [
        name
        for index, name in enumerate(epochs.info["ch_names"])
        if name.startswith("M")
        and name not in bads
        and mne.channel_type(epochs.info, index) in MEG_CHANNEL_TYPES
    ]

    csv_not_in_epochs = [name for name in csv_names if name not in meg_names]
    if csv_not_in_epochs:
        print(
            "CSV channels not found as usable MEG sensors in this epoch file "
            f"(will skip): {csv_not_in_epochs}"
        )

    epoch_not_in_csv = [name for name in meg_names if name not in csv_names]
    if epoch_not_in_csv:
        print(
            f"{len(epoch_not_in_csv)} MEG channels in the epoch file have no "
            "CSV value and will be omitted."
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        "mean_corr": ("Mean sensor correlation", "Mean correlation (r)"),
        "mean_slope_rad_per_press": (
            "Mean sensor slope per press",
            "Slope (rad/press)",
        ),
    }
    subject_label = str(int(subject_id)).zfill(2)
    n_plots = 0

    for metric, csv_values in metric_values.items():
        value_by_name = dict(zip(csv_names, csv_values))
        nonfinite_names = [
            name
            for name in meg_names
            if name in value_by_name and not np.isfinite(value_by_name[name])
        ]
        if nonfinite_names:
            print(f"Skipping {metric} values missing for: {nonfinite_names}")

        # Preserve epoch-file order and match every value by its sensor name.
        names = [
            name
            for name in meg_names
            if name in value_by_name and np.isfinite(value_by_name[name])
        ]
        if not names:
            print(f"No usable MEG values for {metric}; skipping this metric.")
            continue

        picks = np.asarray([channel_indices[name] for name in names], dtype=int)
        sensor_info = mne.pick_info(epochs.info, picks, copy=True)
        values = np.asarray([value_by_name[name] for name in names], dtype=float)
        sensor_types = [
            mne.channel_type(sensor_info, i) for i in range(len(names))
        ]

        # MNE's standard "grad" topomap combines paired planar gradiometers
        # using RMS, which would discard signed correlation and slope values.
        if "grad" in sensor_types:
            raise ValueError(
                "This participant's epoch file contains planar gradiometers. "
                "MNE's standard ch_type='grad' topomap combines channel pairs "
                "by RMS, which is not appropriate for these signed metrics. "
                "The example epoch file in MEG_powerplot_subject.ipynb contains "
                "magnetometers; select an equivalent magnetometer epoch file "
                "or use a plotting method with individual gradiometer coordinates."
            )

        color_limit = float(np.max(np.abs(values)))
        if color_limit == 0:
            color_limit = 1.0
        print(f"{metric}: using symmetric color limits ±{color_limit:.6g}")
        title_prefix, colorbar_label = metric_labels[metric]

        for sensor_type in sorted(set(sensor_types)):
            type_picks = np.asarray(
                [i for i, kind in enumerate(sensor_types) if kind == sensor_type],
                dtype=int,
            )
            if len(type_picks) < 3:
                print(
                    f"Skipping {metric} ({sensor_type}): a topomap needs at "
                    "least three sensor locations."
                )
                continue

            type_info = mne.pick_info(sensor_info, type_picks, copy=True)
            type_names = [names[i] for i in type_picks]
            fig, ax = plt.subplots(figsize=(7, 6))
            image, _ = mne.viz.plot_topomap(
                values[type_picks],
                type_info,
                ch_type=sensor_type,
                axes=ax,
                cmap="RdBu_r",
                contours=0,
                vlim=(-color_limit, color_limit),
                names=type_names if show_names else None,
                show=False,
            )
            ax.set_title(f"{title_prefix} ({sensor_type})")
            fig.colorbar(image, ax=ax, label=colorbar_label)
            n_plots += 1

            if output_dir is not None:
                output_path = output_dir / (
                    f"group_corr_subject_{subject_label}_{metric}_{sensor_type}.png"
                )
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Saved {output_path}")

    if n_plots == 0:
        raise ValueError("No topomaps could be generated from the selected data.")

    plt.show()


def main() -> None:
    args = parse_args()
    subject_id = args.subject_id or input("Subject ID (for example 05 or 16): ")

    csv_path = find_csv(args.csv)
    epochs_path = make_epochs_path(args.data_root, subject_id)
    if not epochs_path.is_file():
        raise FileNotFoundError(
            f"Could not find the participant epochs file: {epochs_path}\n"
            "Check --data-root and the subject ID."
        )

    csv_names, metric_values = read_metrics(csv_path)
    print(f"Reading channel locations from {epochs_path}")
    epochs = mne.read_epochs(epochs_path, preload=False, verbose=False)
    print(f"Reading correlations from {csv_path}")
    plot_metrics(
        epochs,
        csv_names,
        metric_values,
        subject_id,
        args.output_dir,
        args.show_names,
    )


if __name__ == "__main__":
    main()
