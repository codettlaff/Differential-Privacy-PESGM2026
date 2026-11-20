import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

# Settings
target_appliances = ['HPE', 'FRE']
appliances_to_remove = ['RSE']

# The .mat file stores time as elapsed seconds, so real timestamps are rebuilt from these.
start_time = '2012-04-01 00:00'   # Local wall-clock time of the first sample in AMPds2
timezone = 'America/Vancouver'    # AMPds2 was recorded in Burnaby, BC

settings1 = {
    'dataset_name': 'ampds_pattern_matching_1week',
    'T_limit': 1440 * 7, # One Week
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 30,
    'chunk_size': 1440, # One Day
    'time_resolution': 60,
    'time_units': 'seconds',
    'target_apps': target_appliances,
    'apps_to_remove': appliances_to_remove,
    'start_time': start_time,
    'timezone': timezone}

settings2 = {
    'dataset_name': 'ampds_pattern_matching_2year',
    'T_limit': None, # Full Two-Years Dataset
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 30,
    'chunk_size': 1440, # One Day
    'time_resolution': 60,
    'time_units': 'seconds',
    'target_apps': target_appliances,
    'apps_to_remove': appliances_to_remove,
    'start_time': start_time,
    'timezone': timezone}

settings3 = {
    'dataset_name': 'ampds_CNN_1week',
    'T_limit': 1440 * 7, # One Week
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 1,
    'chunk_size': 1440, # One Day
    'time_resolution': 60,
    'time_units': 'seconds',
    'target_apps': target_appliances,
    'apps_to_remove': appliances_to_remove,
    'start_time': start_time,
    'timezone': timezone}

settings4 = {
    'dataset_name': 'ampds_CNN_2year',
    'T_limit': None, # Full Two-Years Dataset
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 1,
    'chunk_size': 1440, # One Day
    'time_resolution': 60,
    'time_units': 'seconds',
    'target_apps': target_appliances,
    'apps_to_remove': appliances_to_remove,
    'start_time': start_time,
    'timezone': timezone}

def build_datetimes(elapsed_seconds, start_time, timezone, time_resolution):
    """Convert elapsed seconds into absolute UTC timestamps (numpy datetime64[s]).

    The first sample is placed at `start_time` (local wall-clock time in `timezone`) and every
    later sample is offset by its elapsed time relative to the first one. Timestamps are kept in
    UTC because numpy datetime64 has no timezone, and local time repeats an hour every November
    when daylight saving ends. Convert back with:
        pd.DatetimeIndex(datetimes).tz_localize('UTC').tz_convert(timezone)
    """
    elapsed_seconds = np.asarray(elapsed_seconds, dtype=np.int64)
    start_utc = pd.Timestamp(start_time, tz=timezone).tz_convert('UTC').tz_localize(None)
    offsets = (elapsed_seconds - elapsed_seconds[0]).astype('timedelta64[s]')
    datetimes = start_utc.to_datetime64().astype('datetime64[s]') + offsets

    # Sanity check: samples should be evenly spaced at the stated resolution
    irregular = np.count_nonzero(np.diff(elapsed_seconds) != time_resolution)
    if irregular:
        print(f'  Warning: {irregular} gaps between samples differ from {time_resolution} s.')
    return datetimes

def generate_dataset(dataset_name, T_limit, train_val_test_split, window_length, stride, chunk_size, time_resolution, time_units, target_apps, apps_to_remove, start_time, timezone):

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(base_dir, 'data')
    smart_meter_data_dir = os.path.join(data_dir, 'smart_meter_data')
    processed_data_dir = os.path.join(base_dir, 'processed_data')

    os.makedirs(processed_data_dir, exist_ok=True)
    ampds_folderpath = os.path.join(smart_meter_data_dir, 'ampds')
    ampds_filepath = os.path.join(ampds_folderpath, 'ampds2.mat')
    data = loadmat(ampds_filepath)

    processed_data_filepath = os.path.join(processed_data_dir, dataset_name + '.npz')

    # Get Labels
    agg_labels, apps_labels = data['labelInp'], data['labelOut']
    electrical_quantities = [label.strip() for label in agg_labels.flatten()]
    electrical_quantities = [quantity for quantity in electrical_quantities if quantity not in ("time", "id")]
    appliance_names = [label.strip() for label in apps_labels.flatten()]
    appliance_names = [name for name in appliance_names if name not in ("time", "id")]

    # Get Data
    T = data['input'].shape[0]
    T = min(T, T_limit) if T_limit is not None else T
    num_chunks = T // chunk_size

    agg_dt, apps_dt = data['input'][:T], data['output'][:T, :, 0] # For appliances, keep only real power P.
    elapsed_seconds, agg = agg_dt[:, 0], agg_dt[:, 2:] # Column 0 = elapsed seconds, column 1 = id (discarded)
    datetimes = build_datetimes(elapsed_seconds, start_time, timezone, time_resolution)
    apps = apps_dt[:, 2:] # discard datetime and id

    # Remove appliances
    remove_idx = [appliance_names.index(name) for name in apps_to_remove]
    remove_cols = [i for i in remove_idx]  # offset for time/id columns

    # Subtract removed appliances from aggregate real power
    agg[:, 0] -= apps[:, remove_cols].sum(axis=1)

    # Drop them from apps, apps_labels and appliance_names
    keep_cols = [c for c in range(apps.shape[1]) if c not in remove_cols]
    apps = apps[:, keep_cols]
    apps_labels = np.asarray(apps_labels).flatten()[keep_cols]
    appliance_names = [n for n in appliance_names if n not in apps_to_remove]

    # Keep only target appliances
    appliance_indices = [appliance_names.index(name) for name in target_apps]

    apps = apps[:, appliance_indices]
    appliance_names = target_apps

    def precompute_indices(
    num_timesteps,
    window_length,
    stride,
    train_val_test_split,
    num_chunks,
    seed=42):

        expected_num_windows = max(0, (num_timesteps - window_length) // stride + 1)

        # Divide the timeline into contiguous, non-overlapping chunks.
        boundaries = np.linspace(0, num_timesteps, num_chunks + 1, dtype=int)
        chunk_ids = np.random.default_rng(seed).permutation(num_chunks)

        # Allocate whole chunks to splits, as close as possible to the requested ratios.
        exact_counts = np.asarray(train_val_test_split) * num_chunks
        chunk_counts = np.floor(exact_counts).astype(int)
        remainder = num_chunks - chunk_counts.sum()
        chunk_counts[np.argsort(-(exact_counts - chunk_counts))[:remainder]] += 1

        split_names = ("train", "val", "test")
        idx_dict = {}
        chunk_offset = 0

        for split_name, chunk_count in zip(split_names, chunk_counts):
            split_chunk_ids = chunk_ids[chunk_offset:chunk_offset + chunk_count]
            chunk_offset += chunk_count

            inp_parts = []

            for chunk_id in split_chunk_ids:
                chunk_start = boundaries[chunk_id]
                chunk_end = boundaries[chunk_id + 1]

                # Keep only windows fully contained in this chunk.
                starts = np.arange(
                    chunk_start,
                    chunk_end - window_length + 1,
                    stride,
                )
                inp_parts.append(starts)

            inp_idx = np.concatenate(inp_parts) if inp_parts else np.array([], dtype=int)

            # Shuffle examples within each split.
            perm = np.random.default_rng(seed + len(idx_dict)).permutation(len(inp_idx))
            idx_dict[split_name + '_indices'] = inp_idx[perm]

        num_windows = sum(len(indices) for indices in idx_dict.values())
        return expected_num_windows, num_windows, idx_dict

    expected_num_windows, num_windows, idx_dict = precompute_indices(T, window_length, stride, train_val_test_split, num_chunks, seed=42)

    # Recompute the train/validation/test proportions from the assigned windows.
    split_names = ("train_indices", "val_indices", "test_indices")
    total = sum(len(idx_dict[name]) for name in split_names)

    real_train_val_test_split = [
        len(idx_dict[name]) / total for name in split_names]

    # calculate real number of unique timestamps used in whole dataset
    all_starts = np.concatenate([idx_dict[name] for name in split_names])
    all_timestamps = np.unique(
        np.concatenate([
            np.arange(start, start + window_length)
            for start in all_starts]))
    real_num_unique_timestamps = len(all_timestamps)

    def normalize_data(x, train_indices, window_length):
        # All timesteps covered by any training window
        train_timesteps = np.unique((train_indices[:, None] + np.arange(window_length)).ravel())
        x_train = x[train_timesteps]

        x_min = x_train.min(axis=0, keepdims=True)
        x_max = x_train.max(axis=0, keepdims=True)
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        return x_norm, x_min.ravel(), x_max.ravel()

    # Normalize using min/max from training data only, then reattach time/id columns
    # Normalize each feature separately using min/max from training data only, then reattach time/id columns
    train_indices = idx_dict['train_indices']

    agg_norm, agg_min_train, agg_max_train = normalize_data(agg, train_indices, window_length)
    apps_norm, apps_min_train, apps_max_train = normalize_data(apps, train_indices, window_length)

    scaling_factors = {
        'agg_min': agg_min_train,    # shape (n_agg_features,)
        'agg_max': agg_max_train,
        'apps_min': apps_min_train,  # shape (n_apps,)
        'apps_max': apps_max_train}

    np.savez_compressed(
        processed_data_filepath,
        electrical_quantities=electrical_quantities,
        appliance_names=appliance_names,
        datetimes=datetimes,              # datetime64[s], UTC
        elapsed_seconds=elapsed_seconds,
        timezone=timezone,
        agg_data=agg,
        apps_data=apps,
        agg_norm=agg_norm,
        apps_norm=apps_norm,
        **scaling_factors,
        **idx_dict,
        train_val_test_split=real_train_val_test_split,
        T=real_num_unique_timestamps,
        window_length=window_length,
        stride=stride,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        time_resolution=time_resolution,
        time_units=time_units)

if __name__ == "__main__":
    for settings in (settings1, settings2, settings3, settings4):
        print(f"Generating {settings['dataset_name']}...")
        generate_dataset(**settings)
    print("Done.")
