import os
import numpy as np
from scipy.io import loadmat

# Data Generation Settings

# Need to add apps_to_remove and set it to 'apps_to_remove'

target_appliances = ['FRE', 'HPE']

pattern_matching_mini_settings = {
    'T_limit': 3000,
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 30,
    'chunk_size': 300,
    'target_appliances': target_appliances,
    'rand_seed': 42,
    'dataset_name': 'ampds_pattern-matching_mini'}

pattern_matching_full_settings = {
    'T_limit': None,
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 30,
    'chunk_size': 300,
    'target_appliances': target_appliances,
    'rand_seed': 42,
    'dataset_name': 'ampds_pattern-matching_full'}

cnn_mini_settings = {
    'T_limit': 3000,
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 1,
    'chunk_size': 300,
    'target_appliances': target_appliances,
    'rand_seed': 42,
    'dataset_name': 'ampds_cnn_mini'}

cnn_full_settings = {
    'T_limit': None,
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 1,
    'chunk_size': 300,
    'target_appliances': target_appliances,
    'rand_seed': 42,
    'dataset_name': 'ampds_cnn_full'}

def generate_data(T_limit, train_val_test_split, window_length, stride, chunk_size, target_appliances, rand_seed, dataset_name):

    # Directories
    scripts_dir = os.path.dirname(__file__)
    base_dir = os.path.join(scripts_dir, '..')
    data_dir = os.path.join(base_dir, 'data')
    processed_data_dir = os.path.join(base_dir, 'processed_data')

    smart_meter_data_folderpath = os.path.join(data_dir, 'smart_meter_data')
    ampds_folderpath = os.path.join(smart_meter_data_folderpath, 'ampds')
    ampds_filepath = os.path.join(ampds_folderpath, 'ampds2.mat')

    # Fixed Settings - Constant for AMPds2 Data
    time_resolution = 60
    time_units = 'seconds'

    # Load Data
    data = loadmat(ampds_filepath)
    agg_data, apps_data = data['input'], data['output']
    agg_labels, apps_labels = data['labelInp'], data['labelOut']

    electrical_quantities = [label.strip() for label in agg_labels.flatten()]
    electrical_quantities = [quantity for quantity in electrical_quantities if quantity not in ("time", "id")]
    appliance_names = [label.strip() for label in apps_labels.flatten()]
    appliance_names = [name for name in appliance_names if name not in ("time", "id")]

    # Trim Data
    T = agg_data.shape[0]
    T = min(T, T_limit) if T_limit is not None else T
    agg_data, apps_data = agg_data[:T_limit], apps_data[:T_limit]
    apps_data = apps_data[:, :, 0]  # For appliances, keep only real power P
    apps_labels = np.asarray(apps_labels).flatten()

    # Remove appliances (e.g. rental suite) and subtract them from aggregate P
    remove_cols = [appliance_names.index(n) + 2 for n in apps_to_remove]
    agg_data[:, 2] -= apps_data[:, remove_cols].sum(axis=1)
    keep_cols = [c for c in range(apps_data.shape[1]) if c not in remove_cols]
    apps_data, apps_labels = apps_data[:, keep_cols], apps_labels[keep_cols]
    appliance_names = [n for n in appliance_names if n not in apps_to_remove]

    # Select Target Appliances Only (look up indices BEFORE overwriting names)
    if target_appliances is not None:
        column_indices = [0, 1] + [appliance_names.index(n) + 2 for n in target_appliances]
        apps_data, apps_labels = apps_data[:, column_indices], apps_labels[column_indices]
        appliance_names = list(target_appliances)

    def get_appliance_metrics(apps):
        return {
            "apps_min": np.min(apps, axis=0),
            "apps_max": np.max(apps, axis=0),
            "apps_mean": np.mean(apps, axis=0),
            "apps_median": np.median(apps, axis=0),
            "apps_std": np.std(apps, axis=0),
            "apps_energy": np.sum(apps, axis=0)}

    # Normalize Data
    def normalize_data(x):
        x_min = x.min(axis=0, keepdims=True)
        x_max = x.max(axis=0, keepdims=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    # Separate ID and time columns, normalize appliance columns, then reattach.
    agg_norm = np.column_stack((agg_data[:, :2], normalize_data(agg_data[:, 2:])))
    apps_norm = np.column_stack((apps_data[:, :2], normalize_data(apps_data[:, 2:])))

    # Create Training, Validation, and Testing Datasets
    num_chunks = T // chunk_size

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
            out_parts = []

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
                out_parts.append(starts + window_length // 2)

            inp_idx = np.concatenate(inp_parts) if inp_parts else np.array([], dtype=int)
            out_idx = np.concatenate(out_parts) if out_parts else np.array([], dtype=int)

            # Shuffle examples within each split.
            perm = np.random.default_rng(seed + len(idx_dict)).permutation(len(inp_idx))
            idx_dict[split_name + '_indices'] = inp_idx[perm]

        num_windows = sum(len(indices) for indices in idx_dict.values())
        return expected_num_windows, num_windows, idx_dict

    expected_num_windows, num_windows, idx_dict = precompute_indices(T, window_length, stride, train_val_test_split, num_chunks, seed=rand_seed)

    # Recompute the train/validation/test proportions from the assigned windows.
    split_names = ("train_indices", "val_indices", "test_indices")
    total = sum(len(idx_dict[name]) for name in split_names)

    real_train_val_test_split = [len(idx_dict[name]) / total for name in split_names]

    # calculate real number of unique timestamps used in whole dataset
    all_starts = np.concatenate([idx_dict[name] for name in split_names])
    all_timestamps = np.unique(
        np.concatenate([
            np.arange(start, start + window_length)
            for start in all_starts]))
    real_num_unique_timestamps = len(all_timestamps)

    # Save Dataset
    dataset_path = os.path.join(processed_data_dir, dataset_name + '.npz')

    np.savez_compressed(
        dataset_path,
        agg_labels=agg_labels,
        apps_labels=apps_labels,
        electrical_quantities=electrical_quantities,
        appliance_names=appliance_names,
        agg_data=agg_data,
        apps_data=apps_data,
        agg_norm=agg_norm,
        apps_norm=apps_norm,
        **appliance_metrics,
        **idx_dict,
        train_val_test_split=real_train_val_test_split,
        T=real_num_unique_timestamps,
        time_resolution=time_resolution,
        time_units=time_units,
        window_length=window_length,
        stride=stride,
        chunk_size=chunk_size,
        num_chunks=num_chunks)

generate_data(**pattern_matching_full_settings)