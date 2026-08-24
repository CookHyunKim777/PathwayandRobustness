#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const char *get_output_root(void) {
    const char *root = getenv("SYNC_PATHS_DATA_DIR");
    return (root != NULL && root[0] != '\0') ? root : "./results";
}

/* ─── xoshiro256** RNG ─── */
static uint64_t xs[4];
static uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t xoshiro_next(void) {
    const uint64_t result = rotl64(xs[1] * 5, 7) * 9;
    const uint64_t t = xs[1] << 17;
    xs[2] ^= xs[0]; xs[3] ^= xs[1];
    xs[1] ^= xs[2]; xs[0] ^= xs[3];
    xs[2] ^= t; xs[3] = rotl64(xs[3], 45);
    return result;
}
static double xoshiro_double(void) { return (xoshiro_next() >> 11) * (1.0 / (UINT64_C(1) << 53)); }
static void xoshiro_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        xs[i] = z ^ (z >> 31);
    }
}

/* ─── Global RK4 workspace for the first-order system (theta only) ─── */
static double *k1_t, *k2_t, *k3_t, *k4_t;
static double *dth_buf, *tmp_t_buf;
static double *sin_buf, *cos_buf;

void alloc_rk4(int N) {
    k1_t = malloc(N*sizeof(double)); k2_t = malloc(N*sizeof(double));
    k3_t = malloc(N*sizeof(double)); k4_t = malloc(N*sizeof(double));
    dth_buf   = malloc(N*sizeof(double));
    tmp_t_buf = malloc(N*sizeof(double));
    sin_buf   = malloc(N*sizeof(double));
    cos_buf   = malloc(N*sizeof(double));
}
void free_rk4(void) {
    free(k1_t); free(k2_t); free(k3_t); free(k4_t);
    free(dth_buf); free(tmp_t_buf);
    free(sin_buf); free(cos_buf);
}

/* ─── Equations of motion for the first-order annealed mean field ─── */
static void get_dtheta(const double *theta,
                       const double *power, const double *gamma,
                       const double *wdeg, double total_wdeg, double lambda,
                       double *dth, int N) {
    double sw = 0.0, cw = 0.0;
    for (int i = 0; i < N; i++) {
        double s = sin(theta[i]);
        double c = cos(theta[i]);
        sin_buf[i] = s; cos_buf[i] = c;
        sw += wdeg[i] * s; cw += wdeg[i] * c;
    }
    double s_norm = sw / total_wdeg;
    double c_norm = cw / total_wdeg;
    for (int i = 0; i < N; i++) {
        double coupling = lambda * wdeg[i] * (s_norm * cos_buf[i] - c_norm * sin_buf[i]);
        dth[i] = (power[i] + coupling) / gamma[i];
    }
}

static void rk4_step(const double *power, const double *gamma,
                     const double *wdeg, double total_wdeg, double lambda,
                     const double *theta, double dt, double *th_out, int N) {
    get_dtheta(theta, power, gamma, wdeg, total_wdeg, lambda, dth_buf, N);
    for (int i = 0; i < N; i++) k1_t[i] = dth_buf[i];
    for (int i = 0; i < N; i++) tmp_t_buf[i] = theta[i] + 0.5*dt*k1_t[i];
    get_dtheta(tmp_t_buf, power, gamma, wdeg, total_wdeg, lambda, dth_buf, N);
    for (int i = 0; i < N; i++) k2_t[i] = dth_buf[i];
    for (int i = 0; i < N; i++) tmp_t_buf[i] = theta[i] + 0.5*dt*k2_t[i];
    get_dtheta(tmp_t_buf, power, gamma, wdeg, total_wdeg, lambda, dth_buf, N);
    for (int i = 0; i < N; i++) k3_t[i] = dth_buf[i];
    for (int i = 0; i < N; i++) tmp_t_buf[i] = theta[i] + dt*k3_t[i];
    get_dtheta(tmp_t_buf, power, gamma, wdeg, total_wdeg, lambda, dth_buf, N);
    for (int i = 0; i < N; i++) k4_t[i] = dth_buf[i];
    for (int i = 0; i < N; i++) {
        th_out[i] = theta[i] + dt*(k1_t[i] + 2.0*k2_t[i] + 2.0*k3_t[i] + k4_t[i]) / 6.0;
        th_out[i] = fmod(th_out[i], 2.0*M_PI);
        if (th_out[i] < 0.0) th_out[i] += 2.0*M_PI;
    }
}

/* ═══ Synchronized-cluster detection ═══ */
typedef struct { int *members; int size; double mean_omega; } Cluster;
static double *g_sort_vals;
static int cmp_by_omega(const void *a, const void *b) {
    double da = g_sort_vals[*(const int *)a], db = g_sort_vals[*(const int *)b];
    return (da > db) - (da < db);
}
static int cmp_cluster_desc(const void *a, const void *b) {
    int sa = ((const Cluster *)a)->size, sb = ((const Cluster *)b)->size;
    return (sa < sb) - (sa > sb);
}

int detect_sync_clusters(const double *avg_omega, int N,
                         double threshold, double merge_thr, int min_size,
                         int *node_cluster, Cluster **clusters_out, int *num_clusters_out) {
    const double merge_sq = merge_thr;
    int *sorted = malloc(N * sizeof(int));
    if (!sorted) return -1;
    for (int i = 0; i < N; i++) sorted[i] = i;
    g_sort_vals = (double *)avg_omega;
    qsort(sorted, N, sizeof(int), cmp_by_omega);
    double *sd = malloc(N * sizeof(double));
    if (!sd) { free(sorted); return -1; }
    for (int i = 0; i < N; i++) sd[i] = avg_omega[sorted[i]];
    int *run_id = malloc(N * sizeof(int));
    if (!run_id) { free(sorted); free(sd); return -1; }
    int num_runs = 0; run_id[0] = 0;
    for (int i = 1; i < N; i++) {
        double d = sd[i] - sd[i-1];
        run_id[i] = (d*d < threshold) ? run_id[i-1] : ++num_runs;
    }
    num_runs++;
    int *run_size = calloc(num_runs, sizeof(int));
    if (!run_size) { free(sorted); free(sd); free(run_id); return -1; }
    for (int i = 0; i < N; i++) run_size[run_id[i]]++;
    int *run_to_cls = malloc(num_runs * sizeof(int));
    if (!run_to_cls) { free(sorted); free(sd); free(run_id); free(run_size); return -1; }
    int ncls = 0;
    for (int r = 0; r < num_runs; r++)
        run_to_cls[r] = (run_size[r] >= min_size) ? ncls++ : -1;
    if (ncls == 0) {
        for (int i = 0; i < N; i++) node_cluster[i] = -1;
        *clusters_out = NULL; *num_clusters_out = 0;
        free(sorted); free(sd); free(run_id); free(run_size); free(run_to_cls);
        return 0;
    }
    Cluster *cls = calloc(ncls, sizeof(Cluster));
    if (!cls) { free(sorted); free(sd); free(run_id); free(run_size); free(run_to_cls); return -1; }
    for (int i = 0; i < N; i++) { int k = run_to_cls[run_id[i]]; if (k >= 0) cls[k].size++; }
    for (int k = 0; k < ncls; k++) {
        cls[k].members = malloc(cls[k].size * sizeof(int));
        if (!cls[k].members) return -1;
        cls[k].size = 0;
    }
    for (int i = 0; i < N; i++) { int k = run_to_cls[run_id[i]]; if (k >= 0) cls[k].members[cls[k].size++] = sorted[i]; }
    for (int k = 0; k < ncls; k++) {
        double s = 0.0;
        for (int j = 0; j < cls[k].size; j++) s += avg_omega[cls[k].members[j]];
        cls[k].mean_omega = s / cls[k].size;
    }
    int merged = 1;
    while (merged) {
        merged = 0; int i = 0;
        while (i < ncls) {
            if (cls[i].size == 0) { i++; continue; }
            int j = i + 1;
            while (j < ncls && cls[j].size == 0) j++;
            if (j >= ncls) break;
            double d = cls[i].mean_omega - cls[j].mean_omega;
            if (d*d < merge_sq) {
                int new_sz = cls[i].size + cls[j].size;
                int *tmp = realloc(cls[i].members, new_sz * sizeof(int));
                if (!tmp) return -1;
                cls[i].members = tmp;
                memcpy(cls[i].members + cls[i].size, cls[j].members, cls[j].size * sizeof(int));
                cls[i].size = new_sz;
                free(cls[j].members); cls[j].members = NULL; cls[j].size = 0;
                double s = 0.0;
                for (int x = 0; x < cls[i].size; x++) s += avg_omega[cls[i].members[x]];
                cls[i].mean_omega = s / cls[i].size;
                merged = 1;
            } else { i = j; }
        }
    }
    int new_ncls = 0;
    for (int k = 0; k < ncls; k++) if (cls[k].size > 0) cls[new_ncls++] = cls[k];
    ncls = new_ncls;
    char *is_clustered = calloc(N, sizeof(char));
    if (!is_clustered) return -1;
    for (int k = 0; k < ncls; k++)
        for (int j = 0; j < cls[k].size; j++) is_clustered[cls[k].members[j]] = 1;
    int *add_to = malloc(N * sizeof(int));
    int *add_count = calloc(ncls, sizeof(int));
    if (!add_to || !add_count) return -1;
    for (int i = 0; i < N; i++) add_to[i] = -1;
    for (int i = 0; i < N; i++) {
        if (is_clustered[i]) continue;
        double best_sq = 1e300; int best_k = -1;
        for (int k = 0; k < ncls; k++) {
            double d = avg_omega[i] - cls[k].mean_omega, dsq = d*d;
            if (dsq < merge_sq && dsq < best_sq) { best_sq = dsq; best_k = k; }
        }
        if (best_k >= 0) { add_to[i] = best_k; add_count[best_k]++; }
    }
    for (int k = 0; k < ncls; k++) {
        if (add_count[k] > 0) {
            int *tmp = realloc(cls[k].members, (cls[k].size + add_count[k]) * sizeof(int));
            if (!tmp) return -1;
            cls[k].members = tmp;
        }
    }
    int *add_fill = calloc(ncls, sizeof(int));
    if (!add_fill) return -1;
    for (int i = 0; i < N; i++) { int k = add_to[i]; if (k >= 0) cls[k].members[cls[k].size + add_fill[k]++] = i; }
    for (int k = 0; k < ncls; k++) cls[k].size += add_count[k];
    qsort(cls, ncls, sizeof(Cluster), cmp_cluster_desc);
    for (int i = 0; i < N; i++) node_cluster[i] = -1;
    for (int k = 0; k < ncls; k++)
        for (int j = 0; j < cls[k].size; j++) node_cluster[cls[k].members[j]] = k;
    free(sorted); free(sd); free(run_id); free(run_size); free(run_to_cls);
    free(is_clustered); free(add_to); free(add_count); free(add_fill);
    *clusters_out = cls; *num_clusters_out = ncls;
    return ncls;
}

void free_clusters(Cluster *cls, int ncls) {
    if (!cls) return;
    for (int k = 0; k < ncls; k++) free(cls[k].members);
    free(cls);
}

static int is_valid_country(const char *s) {
    static const char *allowed[] = {"DE", "ES", "FR", "UK"};
    for (size_t i = 0; i < sizeof(allowed)/sizeof(allowed[0]); i++)
        if (strcmp(s, allowed[i]) == 0) return 1;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
   Complete simulation for one lambda (Phases 1-3) and result output
   ═══════════════════════════════════════════════════════════════ */
static void run_one_lambda(const char *country, int seed, double lambda,
                           const char *alpha_str,
                           const double *power, const double *gamma,
                           const double *wdeg, double total_wdeg,
                           int N, double dt, int transient_steps, int sample_steps,
                           double *theta, double *th2) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Initialize theta independently for each lambda. */
    for (int i = 0; i < N; i++)
        theta[i] = (i + xoshiro_double()) / N * 2.0 * M_PI;
    for (int i = N-1; i > 0; i--) {
        int j = (int)(xoshiro_double() * (i+1));
        double tmp = theta[i]; theta[i] = theta[j]; theta[j] = tmp;
    }

    /* Phase 1: Transient */
    for (int step = 0; step < transient_steps; step++) {
        rk4_step(power, gamma, wdeg, total_wdeg, lambda, theta, dt, th2, N);
        for (int i = 0; i < N; i++) theta[i] = th2[i];
    }

    /* Phase 2: avg_omega */
    double *avg_omega = calloc(N, sizeof(double));
    for (int step = 0; step < sample_steps; step++) {
        get_dtheta(theta, power, gamma, wdeg, total_wdeg, lambda, dth_buf, N);
        for (int i = 0; i < N; i++) avg_omega[i] += dth_buf[i];
        rk4_step(power, gamma, wdeg, total_wdeg, lambda, theta, dt, th2, N);
        for (int i = 0; i < N; i++) theta[i] = th2[i];
    }
    for (int i = 0; i < N; i++) avg_omega[i] /= sample_steps;

    /* Cluster detection */
    int     *node_cluster = malloc(N * sizeof(int));
    Cluster *clusters = NULL; int num_clusters = 0;
    int ret = detect_sync_clusters(avg_omega, N, 1e-2, 1.0, 4,
                                   node_cluster, &clusters, &num_clusters);
    if (ret < 0) { fprintf(stderr, "Cluster detection failed (lambda=%.2f)\n", lambda);
                   free(avg_omega); free(node_cluster); return; }

    /* Phase 3: order parameters, dphase, and nodewise mean dtheta for F. */
    double total_R = 0.0, total_Rw = 0.0;
    double *cls_R_sum      = calloc(num_clusters, sizeof(double));
    double *cls_R_unw_sum  = calloc(num_clusters, sizeof(double));
    double *cls_dphase_sum = calloc(num_clusters, sizeof(double));
    double *p3_avg_dth     = calloc(N, sizeof(double));

    for (int step = 0; step < sample_steps; step++) {
        get_dtheta(theta, power, gamma, wdeg, total_wdeg, lambda, dth_buf, N);
        for (int i = 0; i < N; i++) p3_avg_dth[i] += dth_buf[i];
        rk4_step(power, gamma, wdeg, total_wdeg, lambda, theta, dt, th2, N);
        for (int i = 0; i < N; i++) theta[i] = th2[i];

        double s_sum = 0.0, c_sum = 0.0, sw = 0.0, cw = 0.0;
        for (int i = 0; i < N; i++) {
            double si = sin(theta[i]), ci = cos(theta[i]);
            s_sum += si; c_sum += ci; sw += wdeg[i]*si; cw += wdeg[i]*ci;
        }
        total_R  += sqrt(s_sum*s_sum + c_sum*c_sum) / N;
        total_Rw += sqrt(sw*sw + cw*cw) / total_wdeg;

        for (int k = 0; k < num_clusters; k++) {
            double sc = 0.0, cc = 0.0, sc_unw = 0.0, cc_unw = 0.0, om_sum = 0.0;
            for (int j = 0; j < clusters[k].size; j++) {
                int nd = clusters[k].members[j];
                double w_norm = wdeg[nd] / total_wdeg;
                sc += w_norm*sin(theta[nd]); cc += w_norm*cos(theta[nd]);
                sc_unw += sin(theta[nd]); cc_unw += cos(theta[nd]);
                om_sum += dth_buf[nd];
            }
            cls_R_sum[k]      += sqrt(sc*sc + cc*cc);
            cls_R_unw_sum[k]  += sqrt(sc_unw*sc_unw + cc_unw*cc_unw) / N;
            cls_dphase_sum[k] += om_sum / clusters[k].size;
        }
    }
    double final_R = total_R / sample_steps, final_Rw = total_Rw / sample_steps;
    for (int k = 0; k < num_clusters; k++) {
        cls_R_sum[k] /= sample_steps; cls_R_unw_sum[k] /= sample_steps;
        cls_dphase_sum[k] /= sample_steps;
    }

    for (int i = 0; i < N; i++) p3_avg_dth[i] /= sample_steps;

    /* Global F. */
    double m1 = 0.0, m2 = 0.0, mw1 = 0.0, mw2 = 0.0;
    for (int i = 0; i < N; i++) {
        double x = p3_avg_dth[i];
        m1 += x; m2 += x*x;
        double w = wdeg[i] / total_wdeg;
        mw1 += w*x; mw2 += w*x*x;
    }
    m1 /= N; m2 /= N;
    double F   = m2 - m1*m1;
    double F_w = mw2 - mw1*mw1;

    /* Cluster-resolved F. */
    double *cls_F   = calloc(num_clusters, sizeof(double));
    double *cls_F_w = calloc(num_clusters, sizeof(double));
    for (int k = 0; k < num_clusters; k++) {
        double cm1 = 0.0, cm2 = 0.0, cw1 = 0.0, cw2 = 0.0;
        int sz = clusters[k].size;
        for (int j = 0; j < sz; j++) {
            int nd = clusters[k].members[j];
            double x = p3_avg_dth[nd];
            cm1 += x; cm2 += x*x;
            double w = wdeg[nd] / total_wdeg;
            cw1 += w*x; cw2 += w*x*x;
        }
        cm1 /= sz; cm2 /= sz;
        cls_F[k]   = cm2 - cm1*cm1;
        cls_F_w[k] = cw2 - cw1*cw1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;

    /* Save results, including the alpha layer. */
    char out_dir[1024], out_name[2048], tmp_cmd[1100];
    snprintf(out_dir, sizeof(out_dir),
             "%s/DF/%s/Annealed_massless/alpha_%s/%.2f/%d",
             get_output_root(), country, alpha_str, lambda, seed);
    snprintf(tmp_cmd, sizeof(tmp_cmd), "mkdir -p %s", out_dir);
    if (system(tmp_cmd) != 0) fprintf(stderr, "Warning: mkdir failed\n");
    snprintf(out_name, sizeof(out_name),
             "%s/%s_forward_lmd%.3f_seed%d.txt", out_dir, country, lambda, seed);
    FILE *of = fopen(out_name, "w");
    fprintf(of, "# country\tlambda\tR\tR_weighted\tF\tF_weighted\tTime(s)\n");
    fprintf(of, "%s\t%g\t%g\t%g\t%g\t%g\t%.3f\n", country, lambda, final_R, final_Rw, F, F_w, elapsed);
    fprintf(of, "\n# Node_Index\tAvg_omega\tCluster_ID\n");
    for (int i = 0; i < N; i++) fprintf(of, "%d\t%g\t%d\n", i, avg_omega[i], node_cluster[i]);
    fprintf(of, "\n# Num_Clusters: %d\n", num_clusters);
    fprintf(of, "# Cluster_idx\tSize\tMean_Omega\tR_cluster_weighted\tR_cluster_unweighted\tDphase_Phase3\tF_cluster\tF_cluster_weighted\n");
    for (int k = 0; k < num_clusters; k++)
        fprintf(of, "%d\t%d\t%g\t%g\t%g\t%g\t%g\t%g\n", k, clusters[k].size, clusters[k].mean_omega,
                cls_R_sum[k], cls_R_unw_sum[k], cls_dphase_sum[k], cls_F[k], cls_F_w[k]);
    fclose(of);

    printf("  [lmd=%.2f] R=%.4f R_w=%.4f  clusters=%d  (%.1fs)\n",
           lambda, final_R, final_Rw, num_clusters, elapsed);

    free(avg_omega); free(node_cluster);
    free_clusters(clusters, num_clusters);
    free(cls_R_sum); free(cls_R_unw_sum); free(cls_dphase_sum);
    free(p3_avg_dth); free(cls_F); free(cls_F_w);
}

/* ─── main ─── */
int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <country> <seed> <alpha> [d_file]\n", argv[0]);
        fprintf(stderr, "  country: DE | ES | FR | UK\n");
        fprintf(stderr, "  seed   : RNG seed\n");
        fprintf(stderr, "  alpha  : label string used in output path (e.g. 0.0, 3.0)\n");
        fprintf(stderr, "  lambda : internal loop 0.00 ~ 2.00 step 0.01 (201 values)\n");
        fprintf(stderr, "  d_file : optional per-node coupling weights (wdeg override)\n");
        return 1;
    }
    const char *country = argv[1];
    if (!is_valid_country(country)) {
        fprintf(stderr, "Error: invalid country '%s'\n", country); return 1;
    }
    int seed = atoi(argv[2]);
    const char *alpha_str = argv[3];
    const char *d_file = (argc >= 5) ? argv[4] : NULL;

    const int LMD_START_MILLI = 0;
    const int LMD_STOP_MILLI  = 2010;   /* Include 2.00 in the range 0.00-2.00. */
    const int LMD_STEP_MILLI  = 10;     /* step 0.01 */

    char fname[256];
    snprintf(fname, sizeof(fname), "%s_node_rescale.txt", country);
    FILE *nf = fopen(fname, "r");
    if (!nf) { fprintf(stderr, "Cannot open %s\n", fname); return 1; }
    int N = 0; double m_v, g_v, p_v;
    while (fscanf(nf, "%lf %lf %lf", &m_v, &g_v, &p_v) == 3) N++;
    rewind(nf);
    double *mass_unused = malloc(N * sizeof(double));
    double *gamma = malloc(N * sizeof(double));
    double *power = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
        if (fscanf(nf, "%lf %lf %lf", &mass_unused[i], &gamma[i], &power[i]) != 3) {
            fprintf(stderr, "Error reading node data at line %d\n", i); return 1;
        }
    fclose(nf); free(mass_unused);
    for (int i = 0; i < N; i++) gamma[i] = 1.0;   /* massless: gamma=1 */

    double *wdeg = calloc(N, sizeof(double));
    if (d_file) {
        FILE *df = fopen(d_file, "r");
        if (!df) { fprintf(stderr, "Cannot open d_file %s\n", d_file); return 1; }
        int cnt = 0; double dv;
        while (cnt < N && fscanf(df, "%lf", &dv) == 1) wdeg[cnt++] = dv;
        fclose(df);
        if (cnt != N) { fprintf(stderr, "Error: d_file has %d values, expected N=%d\n", cnt, N); return 1; }
        printf("[%s] wdeg loaded from d_file: %s (%d nodes)\n", country, d_file, cnt);
    } else {
        snprintf(fname, sizeof(fname), "%s_link_data.txt", country);
        FILE *ef = fopen(fname, "r");
        if (!ef) { fprintf(stderr, "Cannot open %s\n", fname); return 1; }
        int u, v; double w;
        while (fscanf(ef, "%d %d %lf", &u, &v, &w) == 3)
            if (u < N && v < N) { wdeg[u] += w; wdeg[v] += w; }
        fclose(ef);
    }
    double total_wdeg = 0.0;
    for (int i = 0; i < N; i++) total_wdeg += wdeg[i];

    xoshiro_seed((uint64_t)seed);
    alloc_rk4(N);
    double *theta = malloc(N * sizeof(double));
    double *th2   = malloc(N * sizeof(double));

    /* Massless timing: dt=2e-3, transient=1500 s, sample=500 s. */
    const double dt = 2e-3;
    const int transient_steps = (int)(1500.0 / dt);   /* 750000 */
    const int sample_steps    = (int)(500.0  / dt);   /* 250000 */

    printf("[%s seed=%d alpha=%s] massless lambda sweep %.2f~%.2f (N=%d, dt=%g, T=%d/%d/%d)\n",
           country, seed, alpha_str, LMD_START_MILLI/1000.0, (LMD_STOP_MILLI-LMD_STEP_MILLI)/1000.0,
           N, dt, transient_steps, sample_steps, sample_steps);

    for (int lm = LMD_START_MILLI; lm < LMD_STOP_MILLI; lm += LMD_STEP_MILLI) {
        double lambda = lm / 1000.0;
        run_one_lambda(country, seed, lambda, alpha_str,
                       power, gamma, wdeg, total_wdeg,
                       N, dt, transient_steps, sample_steps, theta, th2);
    }

    free_rk4();
    free(gamma); free(power); free(wdeg);
    free(theta); free(th2);
    printf("[%s seed=%d alpha=%s] done.\n", country, seed, alpha_str);
    return 0;
}
