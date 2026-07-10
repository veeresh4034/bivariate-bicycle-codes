import stim, pymatching, numpy as np
from brittleness import make_hot_circuit, NOISE_1Q, NOISE_2Q

def estimate_matching(believed_circuit, det_samples):
    """Re-estimate edge probabilities of the believed matching graph from
    unlabeled detection-event samples (two-point correlation method)."""
    dem = believed_circuit.detector_error_model(decompose_errors=True)
    base = pymatching.Matching.from_detector_error_model(dem)
    D = det_samples.astype(np.float64)
    mu = D.mean(axis=0)
    N = D.shape[0]
    edges = base.edges()  # list of (u, v_or_None, {fault_ids, weight, error_probability})
    pair_p = {}
    new = pymatching.Matching()
    # pass 1: bulk edges
    for u, v, attrs in edges:
        if v is None:
            continue
        xij = float(np.dot(D[:, u], D[:, v]) / N)
        num = xij - mu[u] * mu[v]
        den = 1.0 - 2.0 * mu[u] - 2.0 * mu[v] + 4.0 * xij
        if den <= 1e-9:
            p = 0.25
        else:
            inner = 1.0 - 4.0 * num / den
            p = 0.5 - 0.5 * np.sqrt(max(inner, 0.0))
        p = float(np.clip(p, 1e-7, 0.49))
        pair_p[(u, v)] = p
    # pass 2: boundary edges via residual
    for u, v, attrs in edges:
        w_attrs = dict(fault_ids=attrs['fault_ids'])
        if v is None:
            prod = 1.0
            for (a, b), p in pair_p.items():
                if a == u or b == u:
                    prod *= (1.0 - 2.0 * p)
            r = (1.0 - 2.0 * mu[u]) / max(prod, 1e-9)
            p_b = float(np.clip(0.5 * (1.0 - r), 1e-7, 0.49))
            new.add_boundary_edge(u, weight=np.log((1 - p_b) / p_b),
                                  error_probability=p_b, **w_attrs)
        else:
            p = pair_p[(u, v)]
            new.add_edge(u, v, weight=np.log((1 - p) / p),
                         error_probability=p, **w_attrs)
    return new

def run_adaptive(d, p_base, mult, hot_frac, calib_shots_list, test_shots, seed=999):
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p_base,
        before_round_data_depolarization=p_base,
        before_measure_flip_probability=p_base,
        after_reset_flip_probability=p_base)
    qubits = sorted({t.value for inst in c.flattened() if inst.name in NOISE_1Q | NOISE_2Q
                     for t in inst.targets_copy()})
    rng = np.random.default_rng(42)
    hot = set(rng.choice(qubits, size=max(1, int(round(hot_frac * len(qubits)))),
                         replace=False).tolist())
    true_c = make_hot_circuit(c, hot, mult)

    # fixed test set
    dets, obs = true_c.compile_detector_sampler(seed=seed).sample(
        test_shots, separate_observables=True)

    def ler(matcher):
        preds = matcher.decode_batch(dets)
        return float(np.sum(np.any(preds != obs, axis=1))) / test_shots

    m_mis = pymatching.Matching.from_detector_error_model(
        c.detector_error_model(decompose_errors=True))
    m_orc = pymatching.Matching.from_detector_error_model(
        true_c.detector_error_model(decompose_errors=True))
    out = dict(mismatched=ler(m_mis), oracle=ler(m_orc), adapted={})
    # calibration data (unlabeled, separate seed = fresh device data)
    calib_sampler = true_c.compile_detector_sampler(seed=seed + 1)
    max_k = max(calib_shots_list)
    calib_all = calib_sampler.sample(max_k)
    for k in calib_shots_list:
        m_ad = estimate_matching(c, calib_all[:k])
        out['adapted'][k] = ler(m_ad)
    return out

if __name__ == "__main__":
    import json
    cfgs = [dict(d=7, p_base=0.003, mult=10.0, hot_frac=0.05),
            dict(d=7, p_base=0.003, mult=8.0,  hot_frac=0.2)]
    all_out = []
    for cfg in cfgs:
        r = run_adaptive(**cfg, calib_shots_list=[1_000, 10_000, 100_000],
                         test_shots=150_000)
        all_out.append(dict(cfg=cfg, **r))
        print(cfg)
        print(f"  mismatched LER = {r['mismatched']:.5f}")
        print(f"  oracle     LER = {r['oracle']:.5f}")
        for k, v in r['adapted'].items():
            rec = (r['mismatched'] - v) / (r['mismatched'] - r['oracle']) * 100 \
                  if r['mismatched'] > r['oracle'] else float('nan')
            print(f"  adapted@{k:>6} shots LER = {v:.5f}  (gap recovered: {rec:.0f}%)",
                  flush=True)
    json.dump(all_out, open("/home/claude/adaptive_results.json", "w"), indent=1)
