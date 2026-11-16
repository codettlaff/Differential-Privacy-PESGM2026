import os
import numpy as np
from scipy.io import loadmat

# Data Generation Settings

target_appliances = ['FRE', 'HPE']
apps_to_remove = ['RSE']

pattern_matching_mini_settings = {
    'T_limit': 3000,
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 30,
    'chunk_size': 300,
    'target_appliances': target_appliances,
    'apps_to_remove': apps_to_remove,
    'rand_seed': 42,
    'time_resolution': 60,
    'time_units': 'seconds',
    'dataset_name': 'ampds_pattern-matching_mini'}

pattern_matching_full_settings = {
    'T_limit': None,
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 30,
    'chunk_size': 300,
    'target_appliances': target_appliances,
    'apps_to_remove': apps_to_remove,
    'rand_seed': 42,
    'time_resolution': 60,
    'time_units': 'seconds',
    'dataset_name': 'ampds_pattern-matching_full'}

cnn_mini_settings = {
    'T_limit': 3000,
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 1,
    'chunk_size': 300,
    'target_appliances': target_appliances,
    'apps_to_remove': apps_to_remove,
    'rand_seed': 42,
    'time_resolution': 60,
    'time_units': 'seconds',
    'dataset_name': 'ampds_cnn_mini'}

cnn_full_settings = {
    'T_limit': None,
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 1,
    'chunk_size': 300,
    'target_appliances': target_appliances,
    'apps_to_remove': apps_to_remove,
    'rand_seed': 42,
    'time_resolution': 60,
    'time_units': 'seconds',
    'dataset_name': 'ampds_cnn_full'}

## Directories

def generate_data(T_limit, train_val_test_split, window_length, stride, chunk_size, target_appliances, apps_to_remove, rand_seed, time_resolution, time_units, dataset_name):

    # Directories
    scripts_dir = os.path.dirname(__file__)
    base_dir = os.path.join(scripts_dir, '..')
    data_dir = os.path.join(base_dir, 'data')
    smart_meter_data_dir = os.path.join(data_dir, 'smart_meter_data')
    processed_data_dir = os.path.join(base_dir, 'processed_data')
    os.makedirs(processed_data_dir, exist_ok=True)

    ampds_folderpath = os.path.join(smart_meter_data_dir, 'ampds')
    ampds_filepath = os.path.join(ampds_folderpath, 'ampds2.mat')

    data = loadmat(ampds_filepath)

    agg, apps = data['input'], data['output']
    agg_labels, apps_labels = data['labelInp'], data['labelOut']

    electrical_quantities = [label.strip() for label in agg_labels.flatten()]
    electrical_quantities = [quantity for quantity in electrical_quantities if quantity not in ("time", "id")]
    appliance_names = [label.strip() for label in apps_labels.flatten()]
    appliance_names = [name for name in appliance_names if name not in ("time", "id")]

    # Trim Data
    T = agg.shape[0]
    T = min(T, T_limit) if T_limit is not None else T
    num_chunks = T // chunk_size
    agg, apps = agg[:T], apps[:T] # Trim number of timestamps for all data.
    apps = apps[:,:,0] # For appliances, keep only real power P

    # Remove Appliances
    remove_idx = [appliance_names.index(name) for name in apps_to_remove]
    remove_cols = [i + 2 for i in remove_idx]  # offset for time/id columns

    # Subtract removed appliances from aggregate real power
    agg[:, 2] -= apps[:, remove_cols].sum(axis=1)

    # Drop them from apps, apps_labels and appliance_names
    keep_cols = [c for c in range(apps.shape[1]) if c not in remove_cols]
    apps = apps[:, keep_cols]
    apps_labels = np.asarray(apps_labels).flatten()[keep_cols]
    appliance_names = [n for n in appliance_names if n not in apps_to_remove]

    # Keep only target appliances
    appliance_indices = [appliance_names.index(name) for name in target_appliances]
    column_indices = [0, 1] + [i + 2 for i in appliance_indices] # Select appliance columns, preserving the time and id columns

    apps = apps[:, column_indices]
    apps_labels = apps_labels[column_indices]
    appliance_names = target_appliances

    # Split Dataset
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

    expected_num_windows, num_windows, idx_dict = precompute_indices(T, window_length, stride, train_val_test_split, num_chunks, seed=rand_seed)

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

    # Normalize Data
    def normalize_data(x, train_indices, window_length):
        # All timesteps covered by any training window
        train_timesteps = np.unique((train_indices[:, None] + np.arange(window_length)).ravel())
        x_train = x[train_timesteps]

        x_min = x_train.min(axis=0, keepdims=True)
        x_max = x_train.max(axis=0, keepdims=True)
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        return x_norm, x_min.ravel(), x_max.ravel()

    # Normalize using min/max from training data only, then reattach time/id columns
    train_indices = idx_dict['train_indices']

    agg_p_norm, agg_min, agg_max = normalize_data(agg[:, 2:], train_indices, window_length)
    apps_p_norm, apps_min_train, apps_max_train = normalize_data(apps[:, 2:], train_indices, window_length)

    scaling_factors = {
        'apps_min': apps_min_train,
        'apps_max': apps_max_train}

    agg_norm = np.column_stack((agg[:, :2], agg_p_norm))
    apps_norm = np.column_stack((apps[:, :2], apps_p_norm))

    dataset_path = os.path.join(processed_data_dir, dataset_name + '.npz')
    np.savez_compressed(
    dataset_path,
    agg_labels=agg_labels,
    apps_labels=apps_labels,
    electrical_quantities=electrical_quantities,
    appliance_names=appliance_names,
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

if __name__ == '__main__':
    for s in [pattern_matching_mini_settings, pattern_matching_full_settings, cnn_mini_settings, cnn_full_settings]:
        generate_data(**s)
        print(f'Completed Data Generation: {s['dataset_name']}')