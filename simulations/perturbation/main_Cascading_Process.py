import numpy as np
import os
import sys
import time
from pathlib import Path

from core.regular_sampling import *
from core.fully_rk4_degree import *
from core.detect_cluster_GCC import *

# ==================== Parameters ====================
N = 1024
degree_exponent = float(sys.argv[1])
exponent        = float(sys.argv[2])
m               = float(sys.argv[3])
seed            = int(sys.argv[4])

K_list = np.round(np.arange(0.0, 20.0, 0.1), 2)

gamma = 1.0
dt    = 1e-2

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SYNC_PATHS_DATA_DIR", str(REPO_ROOT / "results"))).expanduser()
BASE_PATH = str(DATA_ROOT / "DF_double_peak")

# ==================== Cluster comparison ====================

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
    return window_times[-1]


def compute_order_parameter(phase, degree_per_total):
    return np.abs(np.sum(degree_per_total * np.exp(1j * phase)))


def compute_total_energy(phase, dphase, degree_per_total, m, K, N):
    E_kin = 0.5 * m * np.sum(dphase**2)
    R = np.abs(np.sum(degree_per_total * np.exp(1j * phase)))
    return E_kin - 0.5 * K * N * R**2


# ==================== Perturbed-state calculation ====================

def compute_perturbed_state(phase, degree_per_total, natural_freq, initial_clusters, perturb_indices):
    perturb_set = set(perturb_indices)
    perturbed_R = compute_order_parameter(phase, degree_per_total)

    surviving_cluster_ids = []
    surviving_remaining_sizes = []
    surviving_original_sizes = []
    surviving_removed_counts = []
    surviving_Rs = []
    surviving_mean_freqs = []
    surviving_std_freqs = []
    surviving_mean_abs_freqs = []
    surviving_remaining_indices_list = []

    for i, cluster in enumerate(initial_clusters):
        cluster_set = set(cluster)
        remaining = np.array(sorted(cluster_set - perturb_set))
        if len(remaining) == 0:
            continue
        surviving_cluster_ids.append(i)
        surviving_original_sizes.append(len(cluster))
        surviving_remaining_sizes.append(len(remaining))
        surviving_removed_counts.append(len(cluster_set & perturb_set))
        R_cl = np.abs(np.sum(degree_per_total[remaining] * np.exp(1j * phase[remaining])))
        surviving_Rs.append(R_cl)
        surviving_mean_freqs.append(np.mean(natural_freq[remaining]))
        surviving_std_freqs.append(np.std(natural_freq[remaining]))
        surviving_mean_abs_freqs.append(np.mean(np.abs(natural_freq[remaining])))
        surviving_remaining_indices_list.append(remaining)

    return {
        'perturbed_R': perturbed_R,
        'surviving_cluster_ids': np.array(surviving_cluster_ids),
        'surviving_original_sizes': np.array(surviving_original_sizes),
        'surviving_remaining_sizes': np.array(surviving_remaining_sizes),
        'surviving_removed_counts': np.array(surviving_removed_counts),
        'surviving_Rs': np.array(surviving_Rs),
        'surviving_mean_freqs': np.array(surviving_mean_freqs),
        'surviving_std_freqs': np.array(surviving_std_freqs),
        'surviving_mean_abs_freqs': np.array(surviving_mean_abs_freqs),
        'surviving_remaining_indices_list': surviving_remaining_indices_list,
    }


# ==================== Perturbation experiment ====================

def run_perturbation_experiment(case_name, perturb_indices, steady_phase, steady_dphase,
                                 natural_freq, initial_clusters, gamma, adj, degree_per_total,
                                 params, seed, K):
    print(f"\n--- Starting {case_name} experiment ---")
    print(f"  Perturbed oscillators: {len(perturb_indices)}")

    phase  = steady_phase.copy()
    dphase = steady_dphase.copy()

    dphase[perturb_indices] = natural_freq[perturb_indices] / gamma
    phase[perturb_indices]  = Uniform_distribution(0, 2*np.pi, len(perturb_indices), True, seed + 1)

    perturbed_state = compute_perturbed_state(phase, degree_per_total, natural_freq, initial_clusters, perturb_indices)
    print(f"  R immediately after perturbation: {perturbed_state['perturbed_R']:.6f}")
    for j, cid in enumerate(perturbed_state['surviving_cluster_ids']):
        print(f"    cluster {cid}: {perturbed_state['surviving_original_sizes'][j]} -> "
              f"{perturbed_state['surviving_remaining_sizes'][j]} "
              f"(removed {perturbed_state['surviving_removed_counts'][j]}), "
              f"R={perturbed_state['surviving_Rs'][j]:.4f}, "
              f"<omega>={perturbed_state['surviving_mean_freqs'][j]:.4f}, "
              f"<|omega|>={perturbed_state['surviving_mean_abs_freqs'][j]:.4f}")

    MAX_TIME            = 50000
    MIN_CLUSTER_SIZE    = 32
    MEASURE_WINDOW      = 10000
    COARSE_MEASURE_TIME = 1000
    FINE_WINDOW         = 1000.0
    fine_window_steps   = int(FINE_WINDOW / dt)

    T_init_energy       = 1.0
    n_init_energy_steps = int(T_init_energy / dt)
    E_init_sum          = 0.0
    E_init_counter      = 0
    E_initial           = None

    all_window_clusters  = []
    window_times         = []
    window_dphase_sum    = np.zeros(N)
    window_counter       = 0
    prev_clusters_coarse = None
    curr_clusters_coarse = None
    total_simulated_time = 0.0
    converged  = False
    iteration  = 0

    print(f"  Simulating for at most {MAX_TIME} seconds...")

    while total_simulated_time < MAX_TIME and not converged:
        iteration += 1
        segment_start_time = total_simulated_time
        coarse_start_time  = segment_start_time + MEASURE_WINDOW - COARSE_MEASURE_TIME

        coarse_dphase_sum = np.zeros(N)
        coarse_counter    = 0
        t_list_segment    = np.arange(0, MEASURE_WINDOW, dt)

        for idx, t in enumerate(t_list_segment):
            current_time = segment_start_time + t

            window_dphase_sum += dphase
            window_counter    += 1

            if window_counter == fine_window_steps:
                window_mean_dphase = window_dphase_sum / fine_window_steps
                clusters_window, _ = cluster_index(window_mean_dphase, threshold=1e-8, minimum_size=MIN_CLUSTER_SIZE)
                all_window_clusters.append(clusters_window)
                window_times.append(current_time + dt)
                window_dphase_sum = np.zeros(N)
                window_counter    = 0

            if current_time >= coarse_start_time:
                coarse_dphase_sum += dphase
                coarse_counter    += 1

            if E_init_counter < n_init_energy_steps:
                E_init_sum    += compute_total_energy(phase, dphase, degree_per_total, m, K, N)
                E_init_counter += 1
                if E_init_counter == n_init_energy_steps:
                    E_initial = E_init_sum / n_init_energy_steps
                    print(f"  Initial energy (0-1 s average): {E_initial:.6e}")

            phase, dphase = rk4(adj, degree_per_total, params, phase, dphase, dt)

        total_simulated_time += MEASURE_WINDOW

        coarse_mean_dphase      = coarse_dphase_sum / coarse_counter
        curr_clusters_coarse, _ = cluster_index(coarse_mean_dphase, threshold=1e-8, minimum_size=MIN_CLUSTER_SIZE)
        cluster_sizes = [len(c) for c in curr_clusters_coarse]
        print(f"    {total_simulated_time:.0f}s: {len(curr_clusters_coarse)} clusters, sizes={cluster_sizes}")

        if prev_clusters_coarse is not None:
            if compare_cluster_structures(curr_clusters_coarse, prev_clusters_coarse):
                converged = True
                print(f"  *** Convergence detected at {total_simulated_time}s ***")

        prev_clusters_coarse = curr_clusters_coarse

    final_clusters = curr_clusters_coarse

    if converged:
        precise_convergence_time = find_cluster_emergence_time(window_times, all_window_clusters, final_clusters)
        print(f"  Refined convergence time: {precise_convergence_time:.1f}s")
    else:
        precise_convergence_time = -1
        print(f"  *** Not converged within {MAX_TIME} seconds ***")

    print("  Simulating an additional 1000 seconds to measure the order parameter...")

    T_measure      = 1000
    t_list_measure = np.arange(0, T_measure, dt)
    num_steps      = len(t_list_measure)

    total_R_sum    = 0.0
    total_R_sq_sum = 0.0
    total_E_sum    = 0.0

    num_clusters = len(final_clusters)
    if num_clusters > 0:
        cluster_R_sums    = [0.0 for _ in final_clusters]
        cluster_R_sq_sums = [0.0 for _ in final_clusters]

    for idx, t in enumerate(t_list_measure):
        R_t = compute_order_parameter(phase, degree_per_total)
        total_R_sum    += R_t
        total_R_sq_sum += R_t ** 2
        total_E_sum    += compute_total_energy(phase, dphase, degree_per_total, m, K, N)

        if num_clusters > 0:
            for i, cluster in enumerate(final_clusters):
                R_cluster_t = np.abs(np.sum(degree_per_total[cluster] * np.exp(1j * phase[cluster])))
                cluster_R_sums[i]    += R_cluster_t
                cluster_R_sq_sums[i] += R_cluster_t ** 2

        phase, dphase = rk4(adj, degree_per_total, params, phase, dphase, dt)

    mean_R = total_R_sum / num_steps
    var_R  = (total_R_sq_sum / num_steps) - mean_R ** 2

    cluster_mean_Rs = []
    cluster_var_Rs  = []
    if num_clusters > 0:
        for i in range(num_clusters):
            mean_R_i = cluster_R_sums[i] / num_steps
            var_R_i  = (cluster_R_sq_sums[i] / num_steps) - mean_R_i ** 2
            cluster_mean_Rs.append(mean_R_i)
            cluster_var_Rs.append(var_R_i)

    print(f"  Global R: {mean_R:.6f} (variance: {var_R:.2e})")

    E_final_avg = total_E_sum / num_steps
    T_conv      = precise_convergence_time if precise_convergence_time > 0 else total_simulated_time
    energy_rate = (E_final_avg - E_initial) / T_conv
    print(f"  Energy: E_init={E_initial:.6e}, E_final_avg={E_final_avg:.6e}, rate={energy_rate:.6e}")

    return {
        'case_name': case_name,
        'n_perturbed': len(perturb_indices),
        'perturb_indices': perturb_indices,
        'converged': converged,
        'total_simulated_time': total_simulated_time,
        'precise_convergence_time': precise_convergence_time,
        'mean_R': mean_R,
        'var_R': var_R,
        'num_clusters': num_clusters,
        'cluster_sizes': [len(c) for c in final_clusters],
        'cluster_indices': final_clusters,
        'cluster_mean_Rs': cluster_mean_Rs,
        'cluster_var_Rs': cluster_var_Rs,
        'energy': {
            'E_initial': E_initial,
            'E_final_avg': E_final_avg,
            'energy_rate': energy_rate,
        },
        'perturbed_state': perturbed_state,
    }


# ==================== Result output ====================

def save_case_results(case_dir, results):
    os.makedirs(case_dir, exist_ok=True)
    np.save(f"{case_dir}/converged.npy",                np.array([results['converged']]))
    np.save(f"{case_dir}/total_simulated_time.npy",     np.array([results['total_simulated_time']]))
    np.save(f"{case_dir}/precise_convergence_time.npy", np.array([results['precise_convergence_time']]))
    np.save(f"{case_dir}/mean_R.npy",                   np.array([results['mean_R']]))
    np.save(f"{case_dir}/var_R.npy",                    np.array([results['var_R']]))
    np.save(f"{case_dir}/perturb_indices.npy",          results['perturb_indices'])
    np.save(f"{case_dir}/num_clusters.npy",             np.array([results['num_clusters']]))
    np.save(f"{case_dir}/cluster_sizes.npy",            np.array(results['cluster_sizes']) if results['cluster_sizes'] else np.array([]))
    np.save(f"{case_dir}/cluster_mean_Rs.npy",          np.array(results['cluster_mean_Rs']) if results['cluster_mean_Rs'] else np.array([]))
    np.save(f"{case_dir}/cluster_var_Rs.npy",           np.array(results['cluster_var_Rs']) if results['cluster_var_Rs'] else np.array([]))
    for i, cluster in enumerate(results['cluster_indices']):
        np.save(f"{case_dir}/cluster_{i}_indices.npy", cluster)
    energy = results['energy']
    np.save(f"{case_dir}/energy_rate.npy",
            np.array([energy['energy_rate'], energy['E_initial'], energy['E_final_avg']]))
    ps = results['perturbed_state']
    np.save(f"{case_dir}/perturbed_R.npy",                         np.array([ps['perturbed_R']]))
    np.save(f"{case_dir}/perturbed_surviving_cluster_ids.npy",     ps['surviving_cluster_ids'])
    np.save(f"{case_dir}/perturbed_surviving_original_sizes.npy",  ps['surviving_original_sizes'])
    np.save(f"{case_dir}/perturbed_surviving_remaining_sizes.npy", ps['surviving_remaining_sizes'])
    np.save(f"{case_dir}/perturbed_surviving_removed_counts.npy",  ps['surviving_removed_counts'])
    np.save(f"{case_dir}/perturbed_surviving_Rs.npy",              ps['surviving_Rs'])
    np.save(f"{case_dir}/perturbed_surviving_mean_freqs.npy",      ps['surviving_mean_freqs'])
    np.save(f"{case_dir}/perturbed_surviving_std_freqs.npy",       ps['surviving_std_freqs'])
    np.save(f"{case_dir}/perturbed_surviving_mean_abs_freqs.npy",  ps['surviving_mean_abs_freqs'])
    for j, remaining in enumerate(ps['surviving_remaining_indices_list']):
        np.save(f"{case_dir}/perturbed_surviving_cluster_{j}_indices.npy", remaining)


# ==================== System initialization (fixed seed, independent of K) ====================

print(f"{'='*80}")
print(f"[double_peak] Perturbation Experiment (K sweep)")
print(f"Parameters: degree_exponent={degree_exponent}, exponent={exponent}, m={m}, seed={seed}")
print(f"K range: {K_list[0]:.2f} ~ {K_list[-1]:.2f}  ({len(K_list)} points)")
print(f"{'='*80}")

params_base = np.zeros([3, N])
params_base[0] = PowerLaw_distribution(degree_exponent, N, True, seed, MIN=1.0, MAX=3.0+1.0)
natural_freq = params_base[0].copy()

degree           = np.power(np.abs(params_base[0]) + 1.0, exponent)
mean_degree      = np.mean(degree)
total_degree     = np.sum(degree)
degree_per_mean  = degree / mean_degree
degree_per_total = degree / total_degree

positive_indices = np.where(natural_freq >= 0)[0]
negative_indices = np.where(natural_freq <  0)[0]
positive_sorted  = positive_indices[np.argsort(np.abs(natural_freq[positive_indices]))]
negative_sorted  = negative_indices[np.argsort(np.abs(natural_freq[negative_indices]))]

perturbation_cases_indices = {}
for step in range(4):
    s, e = step * 128, (step + 1) * 128
    perturbation_cases_indices[f'step{step+1}'] = np.concatenate([
        positive_sorted[s:e], negative_sorted[s:e]
    ])

print(f"Positive ω: {len(positive_sorted)}, negative ω: {len(negative_sorted)}")


# ==================== K sweep ====================

for K in K_list:
    print(f"\n{'='*80}")
    print(f"K = {K:.2f}")
    print(f"{'='*80}")

    sentinel = (
        f"{BASE_PATH}/perturbation_experiment/"
        f"degree_exponent_{degree_exponent:.1f}/exponent_{exponent:.2f}/"
        f"m_{m:.1f}/K_{K:.2f}/seed_{seed}/step1/mean_R.npy"
    )
    if os.path.exists(sentinel):
        print(f"  Already done, skipping.")
        continue

    adj = K * degree_per_mean

    params = np.zeros([3, N])
    params[0] = natural_freq
    params[1] = gamma
    params[2] = m

    print("\n=== Initial relaxation (10000 seconds) ===")
    dphase = np.zeros(N)
    phase  = np.zeros(N)
    for i in range(N):
        omega = natural_freq[i]
        if np.abs(omega) <= K:
            phase[i] = np.arctan(omega / K) if K != 0 else 0.0
        else:
            phase[i] = np.pi / 2 if omega > 0 else -np.pi / 2
    phase = np.mod(phase, 2 * np.pi)

    T_initial = 10000
    for idx, t in enumerate(np.arange(0, T_initial, dt)):
        if idx % 200000 == 0:
            print(f"  Progress: {t:.0f}/{T_initial}s")
        phase, dphase = rk4(adj, degree_per_total, params, phase, dphase, dt)
    print("Initial relaxation complete")

    print("\n=== Initial stationary-state measurement (1000 seconds) ===")
    T_measure_initial      = 1000
    t_list_measure_initial = np.arange(0, T_measure_initial, dt)
    num_steps_initial      = len(t_list_measure_initial)

    initial_R_sum      = 0.0
    initial_R_sq_sum   = 0.0
    initial_dphase_sum = np.zeros(N)
    initial_E_sum      = 0.0

    for idx, t in enumerate(t_list_measure_initial):
        R_t = compute_order_parameter(phase, degree_per_total)
        initial_R_sum    += R_t
        initial_R_sq_sum += R_t ** 2
        initial_dphase_sum += dphase
        initial_E_sum    += compute_total_energy(phase, dphase, degree_per_total, m, K, N)
        phase, dphase = rk4(adj, degree_per_total, params, phase, dphase, dt)

    initial_mean_R      = initial_R_sum / num_steps_initial
    initial_var_R       = (initial_R_sq_sum / num_steps_initial) - initial_mean_R ** 2
    initial_mean_E      = initial_E_sum / num_steps_initial
    initial_mean_dphase = initial_dphase_sum / num_steps_initial
    initial_clusters, _ = cluster_index(initial_mean_dphase, threshold=1e-8, minimum_size=32)
    initial_cluster_sizes = [len(c) for c in initial_clusters]

    phase_fc  = phase.copy()
    dphase_fc = dphase.copy()
    initial_cluster_R_sums    = [0.0 for _ in initial_clusters]
    initial_cluster_R_sq_sums = [0.0 for _ in initial_clusters]
    if len(initial_clusters) > 0:
        for idx, t in enumerate(t_list_measure_initial):
            for i, cluster in enumerate(initial_clusters):
                R_cl = np.abs(np.sum(degree_per_total[cluster] * np.exp(1j * phase_fc[cluster])))
                initial_cluster_R_sums[i]    += R_cl
                initial_cluster_R_sq_sums[i] += R_cl ** 2
            phase_fc, dphase_fc = rk4(adj, degree_per_total, params, phase_fc, dphase_fc, dt)

    initial_cluster_mean_Rs = [initial_cluster_R_sums[i] / num_steps_initial for i in range(len(initial_clusters))]
    initial_cluster_var_Rs  = [(initial_cluster_R_sq_sums[i] / num_steps_initial) - initial_cluster_mean_Rs[i]**2
                                for i in range(len(initial_clusters))]

    print(f"  Initial R: {initial_mean_R:.6f} (variance: {initial_var_R:.2e})")
    print(f"  Initial energy: {initial_mean_E:.6e}")
    print(f"  Initial clusters: {len(initial_clusters)}, sizes={initial_cluster_sizes}")

    steady_phase  = phase.copy()
    steady_dphase = dphase.copy()

    print("\n=== Perturbation setup (four steps, 256 oscillators each) ===")
    for step in range(4):
        s, e = step * 128, (step + 1) * 128
        pos_freqs = natural_freq[positive_sorted[s:e]]
        neg_freqs = natural_freq[negative_sorted[s:e]]
        print(f"  step{step+1}: positive ω [{pos_freqs.min():.3f}, {pos_freqs.max():.3f}], "
              f"negative ω [{neg_freqs.min():.3f}, {neg_freqs.max():.3f}]")

    all_results = {}
    for case_name, perturb_indices in perturbation_cases_indices.items():
        results = run_perturbation_experiment(
            case_name, perturb_indices, steady_phase, steady_dphase,
            natural_freq, initial_clusters, gamma, adj, degree_per_total, params, seed, K
        )
        all_results[case_name] = results

    print("\n=== Saving results ===")
    base_dir = f"{BASE_PATH}/perturbation_experiment"
    save_dir = (
        f"{base_dir}/degree_exponent_{degree_exponent:.1f}/exponent_{exponent:.2f}/"
        f"m_{m:.1f}/K_{K:.2f}/seed_{seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    basic_params = {
        'N': N, 'K': K, 'gamma': gamma, 'm': m,
        'degree_exponent': degree_exponent, 'exponent': exponent, 'seed': seed,
        'perturbation_method': 'dphase_and_phase_reset',
        'perturbation_size': 256,
        'initial_condition': 'dphase=0, phase=arctan(omega/K) or pm_pi/2',
        'initial_steady_time': 10000, 'max_resim_time': 50000, 'measure_time': 1000,
    }
    np.save(f"{save_dir}/basic_params.npy", basic_params)
    np.save(f"{save_dir}/natural_freq.npy",     natural_freq)
    np.save(f"{save_dir}/degree_per_total.npy", degree_per_total)
    np.save(f"{save_dir}/steady_phase.npy",     steady_phase)
    np.save(f"{save_dir}/steady_dphase.npy",    steady_dphase)
    np.save(f"{save_dir}/initial_mean_R.npy",   np.array([initial_mean_R]))
    np.save(f"{save_dir}/initial_var_R.npy",    np.array([initial_var_R]))
    np.save(f"{save_dir}/initial_mean_E.npy",   np.array([initial_mean_E]))
    np.save(f"{save_dir}/initial_num_clusters.npy",   np.array([len(initial_clusters)]))
    np.save(f"{save_dir}/initial_cluster_sizes.npy",  np.array(initial_cluster_sizes) if initial_cluster_sizes else np.array([]))
    np.save(f"{save_dir}/initial_cluster_mean_Rs.npy", np.array(initial_cluster_mean_Rs) if initial_cluster_mean_Rs else np.array([]))
    np.save(f"{save_dir}/initial_cluster_var_Rs.npy",  np.array(initial_cluster_var_Rs)  if initial_cluster_var_Rs  else np.array([]))
    for i, cluster in enumerate(initial_clusters):
        np.save(f"{save_dir}/initial_cluster_{i}_indices.npy", cluster)

    for case_name, results in all_results.items():
        save_case_results(f"{save_dir}/{case_name}", results)

    summary_lines = [
        f"seed={seed}, K={K:.2f}, m={m}",
        f"degree_exponent={degree_exponent}, exponent={exponent}",
        "Perturbation: 256 oscillators (128 per side), four intervals", "",
        "[Initial stationary state]",
        f"  R = {initial_mean_R:.6f}",
        f"  E = {initial_mean_E:.6e}",
        f"  Clusters: {len(initial_clusters)}, sizes={initial_cluster_sizes}", "",
    ]
    for case_name, results in all_results.items():
        conv_str = (f"Converged at {results['precise_convergence_time']:.0f}s" if results['converged']
                    else f"Not converged (>{results['total_simulated_time']:.0f}s)")
        ps = results['perturbed_state']
        en = results['energy']
        summary_lines += [
            f"{case_name}:",
            f"  R after perturbation = {ps['perturbed_R']:.6f} -> recovered R = {results['mean_R']:.6f}",
            f"  Energy: E_init={en['E_initial']:.6e}, E_final={en['E_final_avg']:.6e}, dE/dt={en['energy_rate']:.6e}",
            f"  {conv_str}",
            f"  Clusters: {results['num_clusters']}, sizes={results['cluster_sizes']}",
        ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(f"{save_dir}/summary.txt", 'w') as f:
        f.write(summary_text)
    np.save(f"{save_dir}/all_results.npy", all_results)
    print(f"\nSaved results to: {save_dir}")

print(f"\n{'='*80}")
print(f"[double_peak] Completed all K values (seed={seed})")
print(f"{'='*80}")
