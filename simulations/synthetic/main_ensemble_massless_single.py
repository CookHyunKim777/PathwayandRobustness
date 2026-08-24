"""
Unified Simulation - Massless Kuramoto Model, Single-Peak (K sweep)
Usage: python -m simulations.synthetic.main_ensemble_massless_single degree_exponent exponent seed
"""

from core.fully_rk4_degree_massless import *
from core.regular_sampling import *
from core.detect_cluster_GCC import *
import numpy as np
import os
import sys
import tempfile
import shutil
import time
import fcntl
import json
import random
from pathlib import Path

startup_delay = random.uniform(0, 30)
print(f"Starting with {startup_delay:.1f}s delay to reduce I/O contention...")
time.sleep(startup_delay)

# ==================== [SINGLE PEAK] Positive-only power-law sampling ====================

def sample_powerlaw_positive(power_exp, N, seed, MIN=1.0, MAX=4.0):
    """
    Use the positive-half formula from the original ``PowerLaw_distribution``
    (upper bound, regular sampling), extended from N/2 samples to N samples.
    """
    np.random.seed(seed)
    x    = np.arange(1, N + 1)[::-1]
    mgp1 = -power_exp + 1
    km   = MIN ** mgp1
    kM   = MAX ** mgp1
    omega = (kM + (x / N) * (km - kM)) ** (1.0 / mgp1) - MIN
    np.random.shuffle(omega)
    return omega  # All values are positive; shape (N,).


# ==================== Utility functions ====================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SYNC_PATHS_DATA_DIR", str(REPO_ROOT / "results"))).expanduser()
BASE_PATH = str(DATA_ROOT / "DF_single_peak")


def safe_save_npy(filepath, data, expected_shape=None, max_retries=7, retry_delay=1.0):
    lock_path = filepath + '.lock'
    is_object_array = data.dtype == object

    for attempt in range(max_retries):
        try:
            if expected_shape and data.shape != expected_shape:
                print(f"Warning: Shape mismatch for {filepath}")
                print(f"Expected: {expected_shape}, Got: {data.shape}")

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(lock_path, 'w') as lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    temp_dir = os.path.dirname(filepath)
                    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix='.npy') as tmp_file:
                        np.save(tmp_file.name, data)
                        tmp_filename = tmp_file.name
                        tmp_file.flush()
                        os.fsync(tmp_file.fileno())

                    test_data = np.load(tmp_filename, allow_pickle=is_object_array)

                    if not is_object_array:
                        if not np.array_equal(data, test_data):
                            os.remove(tmp_filename)
                            raise ValueError("Data corruption detected after save")

                    if expected_shape and test_data.shape != expected_shape:
                        os.remove(tmp_filename)
                        raise ValueError(f"Shape verification failed")

                    shutil.move(tmp_filename, filepath)

                    final_data = np.load(filepath, allow_pickle=is_object_array)
                    if not is_object_array:
                        if not np.array_equal(data, final_data):
                            raise ValueError("Final verification failed")

                    return True

                except (IOError, OSError) as lock_error:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Lock failed for {filepath}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise lock_error

        except Exception as e:
            if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
                try:
                    os.remove(tmp_filename)
                except:
                    pass

            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed for {filepath}: {e}")
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"All attempts failed for {filepath}: {e}")
                return False

        finally:
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except:
                pass

    return False


def check_disk_space(path, min_gb=10):
    statvfs = os.statvfs(path)
    free_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
    if free_gb < min_gb:
        print(f"Warning: Low disk space ({free_gb:.1f} GB remaining)")
        return False
    return True


def get_checkpoint_paths(degree_exponent, exponent, m, seed):
    base = Path(BASE_PATH)
    checkpoint_dir = base / "unified_checkpoints" / f"degree_exponent_{degree_exponent:.1f}" / f"exponent_{exponent:.2f}" / f"m_{m:.1f}" / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        'checkpoint_file':   checkpoint_dir / "checkpoint.json",
        'failed_log_file':   checkpoint_dir / "failed_Ks.txt",
        'progress_log_file': checkpoint_dir / "progress.log",
        'summary_file':      checkpoint_dir / "summary.txt"
    }


def save_checkpoint(K_idx, degree_exponent, exponent, m, seed, failed_K_indices):
    paths = get_checkpoint_paths(degree_exponent, exponent, m, seed)
    checkpoint_data = {
        'last_completed_K_idx': int(K_idx),
        'failed_K_indices': [int(i) for i in failed_K_indices],
        'timestamp': time.time(),
        'parameters': {
            'degree_exponent': float(degree_exponent),
            'exponent':        float(exponent),
            'm':               float(m),
            'seed':            int(seed),
        },
        'completed_count': K_idx + 1,
        'failed_count':    len(failed_K_indices),
    }
    try:
        with open(paths['checkpoint_file'], 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        with open(paths['progress_log_file'], 'a') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint: K_idx={K_idx}, failed={len(failed_K_indices)}\n")
        return True
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")
        return False


def log_progress(message, degree_exponent, exponent, m, seed):
    paths = get_checkpoint_paths(degree_exponent, exponent, m, seed)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(message)
    with open(paths['progress_log_file'], 'a') as f:
        f.write(f"[{timestamp}] {message}\n")


def save_final_summary(degree_exponent, exponent, m, seed, K_list, failed_K_indices, trivial_K_indices, start_time):
    paths = get_checkpoint_paths(degree_exponent, exponent, m, seed)
    runtime_hours = (time.time() - start_time) / 3600
    total_Ks = len(K_list)
    completed = total_Ks - len(failed_K_indices)
    success_rate = (completed / total_Ks * 100) if total_Ks > 0 else 0
    trivial_rate = (len(trivial_K_indices) / total_Ks * 100) if total_Ks > 0 else 0
    failed_Ks  = [float(K_list[i]) for i in failed_K_indices]
    trivial_Ks = [float(K_list[i]) for i in trivial_K_indices]
    summary = f"""
MASSLESS Single-Peak Unified Simulation (K sweep)
====================================================================
Parameters: degree_exponent={degree_exponent}, exponent={exponent}, m={m} (MASSLESS), seed={seed}
Omega setup: positive-only power-law regular sampling, mean-centered (sum=0)
K range: {K_list[0]:.2f} ~ {K_list[-1]:.2f}, total {total_Ks} points
Energy Rate: (E_final_avg - E_initial) / T_convergence, E = -0.5*K*N*R^2
Results: {completed}/{total_Ks} completed ({success_rate:.1f}%), {len(trivial_K_indices)} trivial ({trivial_rate:.1f}%)
Runtime: {runtime_hours:.2f} hours
Failed Ks : {failed_Ks  if failed_Ks  else 'None'}
Trivial Ks: {trivial_Ks if trivial_Ks else 'None'}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    try:
        with open(paths['summary_file'], 'w') as f:
            f.write(summary)
        print(summary)
    except Exception as e:
        print(f"Failed to save summary: {e}")


# ==================== Cluster comparison functions ====================

def compare_cluster_structures(clusters1, clusters2):
    if len(clusters1) != len(clusters2):
        return False
    if len(clusters1) == 0:
        return True
    sets1 = [frozenset(c) for c in clusters1]
    sets2 = [frozenset(c) for c in clusters2]
    return set(sets1) == set(sets2)


def find_cluster_emergence_time(window_times, window_clusters, final_clusters):
    if len(window_clusters) < 2:
        return window_times[0] if window_times else 0.0
    last_different_idx = -1
    for i in range(len(window_clusters) - 1, -1, -1):
        if not compare_cluster_structures(window_clusters[i], final_clusters):
            last_different_idx = i
            break
    if last_different_idx == -1:
        return window_times[0]
    if last_different_idx + 1 < len(window_times):
        return window_times[last_different_idx + 1]
    else:
        return window_times[-1]


# ==================== Observable calculations ====================

def compute_order_parameter(phase, degree_per_total):
    return np.abs(np.sum(degree_per_total * np.exp(1j * phase)))


def compute_potential_energy(phase, degree_per_total, K, N):
    """E = -0.5*K*N*R²"""
    R = np.abs(np.sum(degree_per_total * np.exp(1j * phase)))
    return -0.5 * K * N * R**2


# ==================== Main simulation function ====================

def run_unified_simulation_massless(seed, degree_exponent, exponent, K, N=1024):
    m = 0.0

    results = {
        'order_parameter': {},
        'arnold_tongue': {},
        'melnikov_domain': {},
        'convergence_info': {},
        'energy': {},
    }

    # ========== System initialization ==========
    omega_raw        = sample_powerlaw_positive(degree_exponent, N, seed, MIN=1.0, MAX=4.0)
    degree           = np.power(np.abs(omega_raw) + 1.0, exponent)
    mean_degree      = np.mean(degree)
    total_degree     = np.sum(degree)
    degree_per_mean  = degree / mean_degree
    degree_per_total = degree / total_degree
    omega_raw       -= np.mean(omega_raw)
    natural_freq     = omega_raw.copy()

    adj   = K * degree_per_mean
    phase = Uniform_distribution(0, 2*np.pi, N, True, seed + 1)
    dt    = 1e-2

    T_init         = 1.0
    n_init_steps   = int(T_init / dt)
    E_init_sum     = 0.0
    E_init_counter = 0

    MAX_TIME            = 90000
    MIN_CLUSTER_SIZE    = 32
    EARLY_CHECK_TIME    = 5000
    MEASURE_WINDOW      = 5000
    COARSE_MEASURE_TIME = 1000
    FINE_WINDOW         = 1000.0
    fine_window_steps   = int(FINE_WINDOW / dt)

    all_window_clusters  = []
    window_times         = []
    window_velocity_sum  = np.zeros(N)
    window_counter       = 0
    prev_clusters_coarse = None
    curr_clusters_coarse = None

    total_simulated_time = 0.0
    converged            = False
    trivial_state        = False
    early_checked        = False
    iteration            = 0

    print(f"  Phase 1: MASSLESS Single-Peak steady-state detection (max {MAX_TIME}s)...")

    while total_simulated_time < MAX_TIME and not converged and not trivial_state:
        iteration         += 1
        segment_start_time = total_simulated_time
        coarse_start_time  = segment_start_time + MEASURE_WINDOW - COARSE_MEASURE_TIME

        print(f"    Iteration {iteration}: {segment_start_time}s -> {segment_start_time + MEASURE_WINDOW}s")

        coarse_velocity_sum = np.zeros(N)
        coarse_counter      = 0
        t_list_segment      = np.arange(0, MEASURE_WINDOW, dt)

        for idx, t in enumerate(t_list_segment):
            current_time     = segment_start_time + t
            current_velocity = get_velocity(adj, degree_per_total, natural_freq, phase)

            window_velocity_sum += current_velocity
            window_counter      += 1

            if window_counter == fine_window_steps:
                window_mean_velocity = window_velocity_sum / fine_window_steps
                clusters_window, _   = cluster_index(window_mean_velocity, threshold=1e-8, minimum_size=MIN_CLUSTER_SIZE)
                all_window_clusters.append(clusters_window)
                window_times.append(current_time + dt)
                window_velocity_sum = np.zeros(N)
                window_counter      = 0

            if current_time >= coarse_start_time:
                coarse_velocity_sum += current_velocity
                coarse_counter      += 1

            if E_init_counter < n_init_steps:
                E_init_sum     += compute_potential_energy(phase, degree_per_total, K, N)
                E_init_counter += 1
                if E_init_counter == n_init_steps:
                    E_initial = E_init_sum / n_init_steps
                    print(f"  Initial potential energy (0-1s avg): {E_initial:.6e}")

            phase = rk4_massless(adj, degree_per_total, natural_freq, phase, dt)

        total_simulated_time += MEASURE_WINDOW

        coarse_mean_velocity    = coarse_velocity_sum / coarse_counter
        curr_clusters_coarse, _ = cluster_index(coarse_mean_velocity, threshold=1e-8, minimum_size=MIN_CLUSTER_SIZE)
        cluster_sizes           = [len(c) for c in curr_clusters_coarse]
        print(f"      Found {len(curr_clusters_coarse)} clusters (sizes: {cluster_sizes})")

        if total_simulated_time >= EARLY_CHECK_TIME and not early_checked:
            early_checked    = True
            max_cluster_size = max(cluster_sizes) if cluster_sizes else 0

            if len(curr_clusters_coarse) == 0:
                trivial_state = True
                print(f"  *** TRIVIAL STATE DETECTED ***")

                E_current = compute_potential_energy(phase, degree_per_total, K, N)

                results['convergence_info'] = {
                    'converged': False, 'trivial_state': True,
                    'total_simulated_time':     total_simulated_time,
                    'early_termination_time':   total_simulated_time,
                    'iterations':               iteration,
                    'precise_convergence_time': 0.0,
                    'num_clusters':             0,
                    'cluster_sizes':            [],
                    'max_cluster_size_at_check': max_cluster_size,
                }
                results['order_parameter']['mean_R'] = compute_order_parameter(phase, degree_per_total)
                results['order_parameter']['var_R']  = 0.0
                results['arnold_tongue']['mean_dphase']                  = coarse_mean_velocity
                results['arnold_tongue']['var_dphase']                   = np.zeros(N)
                results['melnikov_domain']['cluster_list']               = []
                results['melnikov_domain']['remaining']                  = np.arange(N)
                results['melnikov_domain']['cluster_mean_dphases']       = []
                results['melnikov_domain']['cluster_Rs']                 = []
                results['melnikov_domain']['cluster_var_Rs']             = []
                results['melnikov_domain']['cluster_mean_natural_freqs'] = []
                results['energy']['energy_rate'] = (E_current - E_initial) / total_simulated_time
                results['energy']['E_initial']   = E_initial
                results['energy']['E_final_avg'] = E_current

                return results
            else:
                print(f"    Non-trivial (max cluster: {max_cluster_size}), continuing...")

        if prev_clusters_coarse is not None:
            if compare_cluster_structures(curr_clusters_coarse, prev_clusters_coarse):
                converged = True
                print(f"  Cluster structure converged at {total_simulated_time}s!")
            else:
                print(f"      Cluster structure changed, continuing...")

        prev_clusters_coarse = curr_clusters_coarse

    if not converged and not trivial_state:
        print(f"  WARNING: Max time ({MAX_TIME}s) reached without convergence")

    final_clusters           = curr_clusters_coarse
    precise_convergence_time = find_cluster_emergence_time(window_times, all_window_clusters, final_clusters)
    print(f"    Cluster structure first appeared at: {precise_convergence_time:.1f}s")

    results['convergence_info'] = {
        'converged':               converged,
        'trivial_state':           False,
        'total_simulated_time':    total_simulated_time,
        'precise_convergence_time': precise_convergence_time,
        'iterations':              iteration,
        'num_clusters':            len(final_clusters),
        'cluster_sizes':           [len(c) for c in final_clusters],
    }

    results['melnikov_domain']['cluster_list'] = final_clusters
    clustered = set()
    for c in final_clusters:
        clustered.update(c)
    results['melnikov_domain']['remaining'] = np.array([i for i in range(N) if i not in clustered])

    # Phase 2: 500 seconds.
    num_clusters = len(final_clusters)
    T2           = 500
    t_list_2     = np.arange(0, T2, dt)
    num_steps_2  = len(t_list_2)

    temp_mean_R = 0.0
    temp_var_R  = 0.0
    temp_mean_E = 0.0

    arnold_window_time  = 2.0
    arnold_window_steps = int(arnold_window_time / dt)
    dphase_window_averages = []
    arnold_velocity_sum    = np.zeros(N)
    arnold_step_counter    = 0

    if num_clusters > 0:
        cluster_mean_dphases       = [0.0 for _ in final_clusters]
        cluster_Rs                 = [0.0 for _ in final_clusters]
        cluster_Rs_sq              = [0.0 for _ in final_clusters]
        cluster_mean_natural_freqs = [np.mean(natural_freq[cluster]) for cluster in final_clusters]
    else:
        cluster_mean_dphases, cluster_Rs, cluster_Rs_sq, cluster_mean_natural_freqs = [], [], [], []

    for idx, t in enumerate(t_list_2):
        R_t = compute_order_parameter(phase, degree_per_total)
        temp_mean_R += R_t
        temp_var_R  += R_t**2
        temp_mean_E += compute_potential_energy(phase, degree_per_total, K, N)

        current_velocity = get_velocity(adj, degree_per_total, natural_freq, phase)

        arnold_velocity_sum += current_velocity
        arnold_step_counter += 1
        if arnold_step_counter == arnold_window_steps:
            dphase_window_averages.append((arnold_velocity_sum / arnold_window_steps).copy())
            arnold_velocity_sum = np.zeros(N)
            arnold_step_counter = 0

        if num_clusters > 0:
            for i, cluster in enumerate(final_clusters):
                cluster_mean_dphases[i] += np.mean(current_velocity[cluster])
                R_cluster_t = np.abs(np.sum(degree_per_total[cluster] * np.exp(1j * phase[cluster])))
                cluster_Rs[i]    += R_cluster_t
                cluster_Rs_sq[i] += R_cluster_t ** 2

        phase = rk4_massless(adj, degree_per_total, natural_freq, phase, dt)

    temp_mean_R /= num_steps_2
    temp_var_R   = temp_var_R / num_steps_2 - temp_mean_R**2
    E_final_avg  = temp_mean_E / num_steps_2

    results['order_parameter']['mean_R'] = temp_mean_R
    results['order_parameter']['var_R']  = temp_var_R

    dphase_window_averages = np.array(dphase_window_averages)
    results['arnold_tongue']['mean_dphase'] = np.mean(dphase_window_averages, axis=0)
    results['arnold_tongue']['var_dphase']  = np.var(dphase_window_averages,  axis=0)

    cluster_var_Rs = []
    if num_clusters > 0:
        for i in range(num_clusters):
            cluster_mean_dphases[i] /= num_steps_2
            cluster_Rs[i]           /= num_steps_2
            cluster_Rs_sq[i]        /= num_steps_2
            cluster_var_Rs.append(cluster_Rs_sq[i] - cluster_Rs[i] ** 2)

    results['melnikov_domain']['cluster_mean_dphases']       = cluster_mean_dphases
    results['melnikov_domain']['cluster_Rs']                 = cluster_Rs
    results['melnikov_domain']['cluster_var_Rs']             = cluster_var_Rs
    results['melnikov_domain']['cluster_mean_natural_freqs'] = cluster_mean_natural_freqs

    T_conv      = precise_convergence_time if precise_convergence_time > 0 else total_simulated_time
    energy_rate = (E_final_avg - E_initial) / T_conv
    results['energy']['energy_rate'] = energy_rate
    results['energy']['E_initial']   = E_initial
    results['energy']['E_final_avg'] = E_final_avg

    print(f"  Phase 2 complete: R={temp_mean_R:.6f}")
    print(f"  Energy: E_init={E_initial:.6e}, E_final_avg={E_final_avg:.6e}, rate={energy_rate:.6e}")

    return results


def save_all_results(results, seed, degree_exponent, exponent, m, K, N=1024):
    base_path     = BASE_PATH
    success_count = 0
    total_count   = 0

    # 1. Order Parameter
    order_dir = f"{base_path}/order_parameter/degree_exponent_{degree_exponent:.1f}/exponent_{exponent:.2f}/m_{m:.1f}/K_{K:.2f}/seed_{seed}"
    os.makedirs(order_dir, exist_ok=True)
    total_count += 2
    if safe_save_npy(f"{order_dir}/mean_R.npy", np.array([results['order_parameter']['mean_R']])): success_count += 1
    if safe_save_npy(f"{order_dir}/var_R.npy",  np.array([results['order_parameter']['var_R']])): success_count += 1

    # 2. Arnold Tongue
    arnold_dir = f"{base_path}/arnold_tongue/degree_exponent_{degree_exponent:.1f}/exponent_{exponent:.2f}/m_{m:.1f}/K_{K:.2f}/seed_{seed}"
    os.makedirs(arnold_dir, exist_ok=True)
    total_count += 2
    if safe_save_npy(f"{arnold_dir}/mean_dphase.npy", results['arnold_tongue']['mean_dphase'], expected_shape=(N,)): success_count += 1
    if safe_save_npy(f"{arnold_dir}/var_dphase.npy",  results['arnold_tongue']['var_dphase'],  expected_shape=(N,)): success_count += 1

    # 3. Melnikov Domain
    melnikov_dir = f"{base_path}/melnikov_domain/degree_exponent_{degree_exponent:.1f}/exponent_{exponent:.2f}/m_{m:.1f}/K_{K:.2f}/seed_{seed}"
    os.makedirs(melnikov_dir, exist_ok=True)
    cluster_list = results['melnikov_domain']['cluster_list']
    remaining    = results['melnikov_domain']['remaining']

    total_count += 1
    if safe_save_npy(f"{melnikov_dir}/num_clusters.npy", np.array(len(cluster_list))): success_count += 1
    for i, cluster in enumerate(cluster_list):
        total_count += 1
        if safe_save_npy(f"{melnikov_dir}/cluster_{i}_indices.npy", cluster): success_count += 1

    total_count += 5
    if safe_save_npy(f"{melnikov_dir}/cluster_sizes.npy",              np.array([len(c) for c in cluster_list]) if cluster_list else np.array([])): success_count += 1
    if safe_save_npy(f"{melnikov_dir}/cluster_mean_dphases.npy",       np.array(results['melnikov_domain']['cluster_mean_dphases'])       if results['melnikov_domain']['cluster_mean_dphases']       else np.array([])): success_count += 1
    if safe_save_npy(f"{melnikov_dir}/cluster_Rs.npy",                 np.array(results['melnikov_domain']['cluster_Rs'])                 if results['melnikov_domain']['cluster_Rs']                 else np.array([])): success_count += 1
    if safe_save_npy(f"{melnikov_dir}/cluster_var_Rs.npy",             np.array(results['melnikov_domain']['cluster_var_Rs'])             if results['melnikov_domain']['cluster_var_Rs']             else np.array([])): success_count += 1
    if safe_save_npy(f"{melnikov_dir}/cluster_mean_natural_freqs.npy", np.array(results['melnikov_domain']['cluster_mean_natural_freqs']) if results['melnikov_domain']['cluster_mean_natural_freqs'] else np.array([])): success_count += 1

    total_count += 1
    if safe_save_npy(f"{melnikov_dir}/remaining_nodes.npy", remaining): success_count += 1

    # 4. Convergence Info
    conv_dir  = f"{base_path}/convergence_info/degree_exponent_{degree_exponent:.1f}/exponent_{exponent:.2f}/m_{m:.1f}/K_{K:.2f}/seed_{seed}"
    os.makedirs(conv_dir, exist_ok=True)
    conv_info = results['convergence_info']
    total_count += 7
    if safe_save_npy(f"{conv_dir}/converged.npy",                np.array([conv_info['converged']])): success_count += 1
    if safe_save_npy(f"{conv_dir}/trivial_state.npy",            np.array([conv_info['trivial_state']])): success_count += 1
    if safe_save_npy(f"{conv_dir}/total_simulated_time.npy",     np.array([conv_info['total_simulated_time']])): success_count += 1
    if safe_save_npy(f"{conv_dir}/precise_convergence_time.npy", np.array([conv_info['precise_convergence_time']])): success_count += 1
    if safe_save_npy(f"{conv_dir}/num_clusters.npy",             np.array([conv_info['num_clusters']])): success_count += 1
    if safe_save_npy(f"{conv_dir}/cluster_sizes.npy",            np.array(conv_info['cluster_sizes']) if conv_info['cluster_sizes'] else np.array([])): success_count += 1
    if safe_save_npy(f"{conv_dir}/iterations.npy",               np.array([conv_info['iterations']])): success_count += 1

    # 5. Energy
    energy_dir = f"{base_path}/energy/degree_exponent_{degree_exponent:.1f}/exponent_{exponent:.2f}/m_{m:.1f}/K_{K:.2f}/seed_{seed}"
    os.makedirs(energy_dir, exist_ok=True)
    total_count += 1
    energy_data = np.array([results['energy']['energy_rate'], results['energy']['E_initial'], results['energy']['E_final_avg']])
    if safe_save_npy(f"{energy_dir}/energy_rate.npy", energy_data, expected_shape=(3,)): success_count += 1

    return success_count, total_count


# ==================== Main entry point ====================

if __name__ == "__main__":
    N = 1024
    m = 0.0

    degree_exponent = float(sys.argv[1])
    exponent        = float(sys.argv[2])
    seed            = int(sys.argv[3])

    K_list   = np.round(np.arange(0.0, 20.0, 0.1), 2)
    total_Ks = len(K_list)

    start_time = time.time()

    os.makedirs(BASE_PATH, exist_ok=True)

    if not check_disk_space(BASE_PATH):
        log_progress("WARNING: Low disk space detected!", degree_exponent, exponent, m, seed)

    failed_K_indices  = []
    trivial_K_indices = []
    log_progress(
        f"[single_peak MASSLESS] K sweep started: degree_exponent={degree_exponent}, "
        f"exponent={exponent}, seed={seed}, K: {K_list[0]:.2f}~{K_list[-1]:.2f}",
        degree_exponent, exponent, m, seed
    )

    for K_idx, K in enumerate(K_list):
        try:
            log_progress(
                f"Processing K={K:.2f} (idx {K_idx+1}/{total_Ks})...",
                degree_exponent, exponent, m, seed
            )
            K_start_time = time.time()

            results = run_unified_simulation_massless(seed, degree_exponent, exponent, K, N)

            if results['convergence_info']['trivial_state']:
                trivial_K_indices.append(K_idx)

            success_count, total_count = save_all_results(results, seed, degree_exponent, exponent, m, K, N)

            K_wall       = time.time() - K_start_time
            conv_time    = results['convergence_info']['total_simulated_time']
            precise_time = results['convergence_info']['precise_convergence_time']
            num_clusters = results['convergence_info']['num_clusters']

            if results['convergence_info']['trivial_state']:
                status_str = "TRIVIAL"
            elif results['convergence_info']['converged']:
                status_str = f"converged ({num_clusters} clusters)"
            else:
                status_str = f"max_time ({num_clusters} clusters)"

            log_progress(
                f"K={K:.2f} done: {success_count}/{total_count} files, "
                f"coarse={conv_time}s, precise={precise_time:.0f}s ({status_str}), "
                f"dE/dt={results['energy']['energy_rate']:.4e}, wall={K_wall:.1f}s",
                degree_exponent, exponent, m, seed
            )

            if success_count < total_count:
                log_progress(f"WARNING: Some files failed to save for K={K:.2f}", degree_exponent, exponent, m, seed)

            if K_idx % 10 == 0:
                checkpoint_delay = random.uniform(0, 5)
                time.sleep(checkpoint_delay)
                save_checkpoint(K_idx, degree_exponent, exponent, m, seed, failed_K_indices)

        except Exception as e:
            log_progress(f"Error processing K={K:.2f}: {e}", degree_exponent, exponent, m, seed)
            import traceback
            log_progress(traceback.format_exc(), degree_exponent, exponent, m, seed)
            failed_K_indices.append(K_idx)
            continue

    save_checkpoint(total_Ks - 1, degree_exponent, exponent, m, seed, failed_K_indices)
    save_final_summary(degree_exponent, exponent, m, seed, K_list, failed_K_indices, trivial_K_indices, start_time)
    log_progress("[single_peak MASSLESS] K sweep ended", degree_exponent, exponent, m, seed)
    print("end")
