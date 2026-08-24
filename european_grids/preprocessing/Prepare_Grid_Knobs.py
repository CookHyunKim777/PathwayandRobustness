#!/usr/bin/env python3
"""
prepare_grid_knobs.py
─────────────────────
Reparametrize a real (annealed) grid onto the model's coupling
    d_i = (1+|omega|)^alpha * weight,  normalized to <d>=1,
matching the model definition in main_ensemble_*_single_peak.py:

  - degree(=d) is built from omega BEFORE mean-centering  (positive raw scale)
  - the natural frequency fed to the EOM is mean-centered afterwards

The real P has both signs (generators >0, consumers <0), so P is remapped
to the model's POSITIVE and NEGATIVE min/max SEPARATELY.

Procedure
  TARGET  run model's sample_powerlaw_positive(deg_exp,N,seed), mean-center,
          read pos/neg [min,max] as the remap targets.
  STEP 1  sign-split remap of real P:
            P>0  -> [pos_min, pos_max]   (shape kept within positives)
            P<0  -> [neg_min, neg_max]   (shape kept within negatives)
          -> omega_new  (this is the "raw"/pre-centering coupling scale)
  STEP 2  weight w from real K, using ORIGINAL real P (pre-remap):
            w_i = (K_i / sum K) * sum_j (1+|P_old_j|)^alpha
  STEP 3  reassemble with remapped omega_new, SAME alpha:
            d_i = (1+|omega_new_i|)^alpha * w_i
  STEP 4  d_i /= mean(d)
  OMEGA-OUT  natural frequency for the EOM = omega_new - mean(omega_new)
             (mean-centered, matching the model's omega_raw -= mean)

alpha is a single fixed external input (identical in step 2 and step 3).
"""
import argparse, os
import numpy as np


def sample_powerlaw_positive(power_exp, N, seed, MIN=1.0, MAX=4.0):
    """Verbatim copy of the model's sampler."""
    np.random.seed(seed)
    x    = np.arange(1, N + 1)[::-1]
    mgp1 = -power_exp + 1
    km   = MIN ** mgp1
    kM   = MAX ** mgp1
    omega = (kM + (x / N) * (km - kM)) ** (1.0 / mgp1) - MIN
    np.random.shuffle(omega)
    return omega


def model_target_ranges(deg_exp, N, seed, MIN=1.0, MAX=4.0):
    """Return (pos_min,pos_max,neg_min,neg_max) from the model, mean-centered."""
    o = sample_powerlaw_positive(deg_exp, N, seed, MIN, MAX)
    o = o - o.mean()
    pos = o[o > 0]; neg = o[o < 0]
    return pos.min(), pos.max(), neg.min(), neg.max()


def load_node(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    return arr[:, 0], arr[:, 1], arr[:, 2]   # mass, gamma, power(=P)


def load_link_K(path, N):
    K = np.zeros(N)
    with open(path) as f:
        for line in f:
            s = line.split()
            if len(s) < 3: continue
            u, v, w = int(s[0]), int(s[1]), float(s[2])
            if u < N and v < N: K[u]+=w; K[v]+=w
    return K


def remap_signed(P, pos_min, pos_max, neg_min, neg_max):
    """
    STEP 1: sign-split min/max remap. Shape kept WITHIN each sign block.
    Zeros (if any) are left at 0.
    """
    out = np.zeros_like(P)
    pm = P > 0
    nm = P < 0
    if pm.any():
        p = P[pm]
        if p.max() > p.min():
            u = (p - p.min()) / (p.max() - p.min())
        else:
            u = np.zeros_like(p)
        out[pm] = pos_min + u * (pos_max - pos_min)
    if nm.any():
        n = P[nm]
        # keep shape: map most-negative -> neg_min, least-negative -> neg_max
        if n.max() > n.min():
            u = (n - n.min()) / (n.max() - n.min())   # 0 at most-neg, 1 at least-neg
        else:
            u = np.zeros_like(n)
        out[nm] = neg_min + u * (neg_max - neg_min)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--alpha", type=float, required=True,
                    help="fixed correlation exponent, used in BOTH step 2 and step 3")
    ap.add_argument("--node", required=True)
    ap.add_argument("--link", required=True)
    # model target selection
    ap.add_argument("--deg-exp", type=float, required=True,
                    help="model degree_exponent (power-law) defining target ranges")
    ap.add_argument("--model-N", type=int, default=1024)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--MIN", type=float, default=1.0)
    ap.add_argument("--MAX", type=float, default=4.0)
    ap.add_argument("--outdir", default="./prepared")
    ap.add_argument("--mass", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _, _, P_old = load_node(args.node)          # original real P (pre-remap)
    N = len(P_old)
    K_real = load_link_K(args.link, N)
    a = args.alpha

    # ---- TARGET: model pos/neg min/max ----
    pmin, pmax, nmin, nmax = model_target_ranges(
        args.deg_exp, args.model_N, args.model_seed, args.MIN, args.MAX)

    # ---- STEP 1: sign-split remap of real P (pre-centering coupling scale) ----
    omega_new = remap_signed(P_old, pmin, pmax, nmin, nmax)

    # ---- STEP 2: weight w from real K, using ORIGINAL real P ----
    #     w_i = (K_i / sum K) * sum_j (1+|P_old_j|)^alpha
    Knorm = K_real / K_real.sum()
    base_old = np.power(1.0 + np.abs(P_old), a)
    w = Knorm * base_old.sum()

    # ---- STEP 3: reassemble with remapped omega_new, SAME alpha ----
    base_new = np.power(1.0 + np.abs(omega_new), a)
    d = base_new * w

    # ---- STEP 4: normalize <d>=1 ----
    d /= d.mean()

    # ---- OMEGA-OUT: mean-center for the EOM (matches model omega_raw -= mean) ----
    omega_out = omega_new - omega_new.mean()

    # ---- Write node and edge data using the conventional filenames. ----
    node_out = os.path.join(args.outdir, f"{args.country}_node_rescale.txt")
    edge_out = os.path.join(args.outdir, f"{args.country}_link_rescale.txt")
    with open(node_out, "w") as f:
        for i in range(N):
            f.write(f"{args.mass:.6g}\t{args.gamma:.6g}\t{omega_out[i]:.10g}\n")
    with open(edge_out, "w") as f:
        for i in range(N):
            f.write(f"{d[i]:.10g}\n")

    # ---- report ----
    def corr(x, y): return np.corrcoef(x, y)[0, 1]
    npos = (P_old > 0).sum(); nneg = (P_old < 0).sum()
    print(f"[{args.country}] N={N}  alpha={a}  deg_exp={args.deg_exp}")
    print(f"  TARGET(model): pos[{pmin:+.4f},{pmax:+.4f}]  neg[{nmin:+.4f},{nmax:+.4f}]")
    print(f"  real P: {npos} positive, {nneg} negative"
          f"   old range[{P_old.min():+.3f},{P_old.max():+.3f}]")
    print(f"  STEP1 omega_new: [{omega_new.min():+.4f},{omega_new.max():+.4f}]"
          f"  (pre-centering, raw scale for d)")
    print(f"  STEP2 w  (from real K, original P): sum(1+|P|)^a={base_old.sum():.4e}")
    print(f"  STEP4 d  : mean={d.mean():.6f}  std={d.std():.3e}"
          f"  max/min={d.max()/max(d.min(),1e-300):.2f}")
    print(f"  OMEGA-OUT (EOM natural freq): mean={omega_out.mean():+.2e}"
          f"  range[{omega_out.min():+.4f},{omega_out.max():+.4f}]")
    print(f"  corr(|omega_new|,d)={corr(np.abs(omega_new),d):+.3f}")
    if abs(a) < 1e-12:
        print(f"  sanity alpha=0: corr(K_real,d)={corr(K_real,d):+.6f} (expect +1.000000)")
    print(f"  wrote: {node_out}")
    print(f"         {edge_out}")


if __name__ == "__main__":
    main()
