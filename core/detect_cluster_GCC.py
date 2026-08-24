"""Utilities for detecting synchronized clusters from mean angular velocities."""

import numpy as np
import networkx as nx
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def R(phase, N):
    return np.abs(np.sum(np.exp(1j * phase))) / N

def cluster_index(mean_dphases, threshold=1e-8, minimum_size=16):
    """Detect synchronized clusters from time-averaged angular velocities.

    The procedure sorts the oscillators by angular velocity, constructs
    connected components from neighboring values, merges sufficiently close
    components, and assigns remaining oscillators to compatible clusters.
    """
    # 1) Sort the angular velocities and retain the original indices.
    sorted_idx = np.argsort(mean_dphases)
    sd = mean_dphases[sorted_idx]
    N = len(sd)

    # 2) Connect neighboring values that satisfy the velocity threshold.
    adj = np.zeros((N, N), int)
    for i in range(N-1):
        if (sd[i+1] - sd[i])**2 < threshold:
            adj[i, i+1] = adj[i+1, i] = 1

    # 3) Extract the initial clusters as connected components.
    g = nx.from_numpy_array(adj)
    raw_clusters = [set(c) for c in nx.connected_components(g) if len(c) >= minimum_size]

    # 4) Repeatedly merge clusters with sufficiently close mean velocities.
    merged = True
    while merged:
        merged = False
        # compute means
        means = [np.mean(sd[list(c)]) for c in raw_clusters]
        for i in range(len(raw_clusters)):
            for j in range(i+1, len(raw_clusters)):
                # Apply the squared-difference criterion used in the simulations.
                if (means[i] - means[j])**2 < np.sqrt(threshold):
                    # Merge the two components.
                    raw_clusters[i] |= raw_clusters[j]
                    del raw_clusters[j]
                    merged = True
                    break
            if merged:
                break

    # 5) Assign remaining oscillators to compatible clusters.
    already_clustered = set()
    for c in raw_clusters:
        already_clustered |= c

    node_left = set(range(N)) - already_clustered

    final_clusters = []
    for c in raw_clusters:
        # Mean angular velocity of the current cluster.
        mu = np.mean(sd[list(c)])
        to_add = {n for n in list(node_left)
                  if (mean_dphases[sorted_idx[n]] - mu)**2 < np.sqrt(threshold)}
        c |= to_add
        node_left -= to_add
        # Map the sorted indices back to the original oscillator indices.
        final_clusters.append(np.array([sorted_idx[idx] for idx in sorted(c)]))

    # 6) Collect oscillators that remain unassigned.
    remaining = np.array([sorted_idx[n] for n in node_left])
    
    # Return clusters in descending order of size.
    final_clusters_sorted = sorted(final_clusters, key=len, reverse=True)
    return final_clusters_sorted, remaining

# class devil_stair:
#     def __init__(self, dphase):
#         # dphase: time average dphase
#         # cluster_list: node list of each cluster
#         # etc: node list of single node
#         self.dphase = dphase.copy()
#         self.cluster_list, self.etc = cluster_index(dphase)
        
#     def height(self):
#         if len(self.cluster_list) < 1:
#             return np.array([])
#         elif len(self.cluster_list) == 1:
#             return np.array([0.0])
#         elif len(self.cluster_list) > 1:
#             base_dphase = np.mean(self.dphase[self.cluster_list[0]])    # mean dphase of giant cluster
#             secondary_cluster_height = np.mean(self.dphase[self.cluster_list[1]]) - base_dphase   # difference between mean dphase of giant cluster and secondary cluster 
#             height_ratios = []
#             for i in range(len(self.cluster_list)):
#                 height_ratio = (np.mean(self.dphase[self.cluster_list[i]]) - base_dphase) / secondary_cluster_height
#                 height_ratios.append(height_ratio)
#             return np.array(height_ratios)
    
#     def cluster(self, heights = [1.0, 1.5, 2.0]):
#         if len(self.cluster_list) < 1 :
#             return [], []
#         elif len(self.cluster_list[0]) < 9:
#             return [], []
            
#         if heights == None:
#             return self.cluster_list, self.height()
#         else:
#             height_ratio = np.round(self.height(), 1)
#             cluster_list = [self.cluster_list[0]]
#             # heights = [1.0] + heights
#             new_height_ratio = [self.height()[0]]
#             for height in heights:
#                 idxs = np.where(height_ratio == height)[0]
#                 cluster_list_temp = []
#                 new_height_ratio_temp = []
#                 for idx in idxs:
#                     cluster_list_temp.append(self.cluster_list[idx])
#                     new_height_ratio_temp.append(self.height()[idx])
#                 if len(cluster_list_temp) == 0 : pass
#                 else:
#                     cluster_list.append(np.concatenate(cluster_list_temp))
#                     new_height_ratio.append(np.mean(np.array(new_height_ratio_temp)))
#             return cluster_list, new_height_ratio
            
class devil_stair:
    def __init__(self, dphase, cluster_list):
        self.dphase = dphase
        self.cluster_list = cluster_list
        
    def height(self):
        if len(self.cluster_list) < 2:
            return np.array([])

        base_cluster = self.cluster_list[0]
        base_mean = np.mean(self.dphase[base_cluster])
        second_cluster = self.cluster_list[1]
        second_mean = np.mean(self.dphase[second_cluster])
        reference_gap = second_mean - base_mean
        
        if abs(reference_gap) < 1e-9:
            return np.array([])
 
        height_ratios = []
        for i in range(len(self.cluster_list)):
            cidx = self.cluster_list[i]
            cmean = np.mean(self.dphase[cidx])
            ratio = (cmean - base_mean) / reference_gap
            height_ratios.append(ratio)

        return np.array(height_ratios)
    
def cluster_by_kde_more_peaks(dphase_values, bandwidth=0.05, x_grid_points=2000,
                              min_peak_height=0.01, cluster_margin=0.1, min_cluster_size=30):
    dphase_values = np.array(dphase_values)
    N_val = len(dphase_values)
    if N_val == 0:
        return [], [], np.array([])
    
    kde = gaussian_kde(dphase_values, bw_method=bandwidth)
    val_min, val_max = dphase_values.min(), dphase_values.max()
    x_grid = np.linspace(val_min - 0.5, val_max + 0.5, x_grid_points)
    kde_values = kde(x_grid)
    
    peaks, properties = find_peaks(kde_values, height=min_peak_height)
    peak_positions = x_grid[peaks]
    
    cluster_list = []
    cluster_centers = []
    assigned_mask = np.zeros(N_val, dtype=bool)
    indices = np.arange(N_val)
    
    for p in peak_positions:
        left_bound = p - cluster_margin
        right_bound = p + cluster_margin
        in_cluster = ((dphase_values >= left_bound) &
                      (dphase_values <= right_bound) &
                      (~assigned_mask))
        csize = np.sum(in_cluster)
        if csize >= min_cluster_size:
            cluster_indices = indices[in_cluster]
            cluster_list.append(cluster_indices)
            cluster_centers.append(p)
            assigned_mask[in_cluster] = True
    unassigned = indices[~assigned_mask]
    return cluster_list, np.array(cluster_centers), unassigned        
