import stim, pymatching, numpy as np, json, sys

NOISE_1Q = {"DEPOLARIZE1", "X_ERROR", "Z_ERROR"}
NOISE_2Q = {"DEPOLARIZE2"}

def make_hot_circuit(circuit: stim.Circuit, hot_qubits: set, mult: float) -> stim.Circuit:
    """Rewrite noise instructions so ops touching hot qubits have mult x noise."""
    out = stim.Circuit()
    for inst in circuit.flattened():
        name = inst.name
        if name in NOISE_1Q:
            p = inst.gate_args_copy()[0]
            cold, hot = [], []
            for t in inst.targets_copy():
                (hot if t.value in hot_qubits else cold).append(t.value)
            if cold:
                out.append(name, cold, p)
            if hot:
                out.append(name, hot, min(mult * p, 0.75))
        elif name in NOISE_2Q:
            p = inst.gate_args_copy()[0]
            ts = [t.value for t in inst.targets_copy()]
            cold, hot = [], []
            for i in range(0, len(ts), 2):
                pair = ts[i:i+2]
                (hot if (pair[0] in hot_qubits or pair[1] in hot_qubits) else cold).extend(pair)
            if cold:
                out.append(name, cold, p)
            if hot:
                out.append(name, hot, min(mult * p, 0.9375))
        else:
            out.append(inst)
    return out

def logical_error_rate(sample_circuit, decoder_circuit, shots, seed):
    dem = decoder_circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = sample_circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    preds = matcher.decode_batch(dets)
    errs = np.sum(np.any(preds != obs, axis=1))
    return errs / shots, int(errs)

def run(d, p_base, mult, hot_frac, shots, seed=1234):
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d, rounds=d,
        after_clifford_depolarization=p_base,
        before_round_data_depolarization=p_base,
        before_measure_flip_probability=p_base,
        after_reset_flip_probability=p_base)
    qubits = sorted({t.value for inst in c.flattened() if inst.name in NOISE_1Q | NOISE_2Q
                     for t in inst.targets_copy()})
    rng = np.random.default_rng(42)  # fixed hot-spot pattern across all runs
    n_hot = max(1, int(round(hot_frac * len(qubits))))
    hot = set(rng.choice(qubits, size=n_hot, replace=False).tolist())
    true_c = make_hot_circuit(c, hot, mult) if mult != 1.0 else c
    ler_mis, e1 = logical_error_rate(true_c, c,      shots, seed)   # believes uniform
    ler_orc, e2 = logical_error_rate(true_c, true_c, shots, seed)   # knows truth
    return dict(d=d, p=p_base, mult=mult, hot_frac=hot_frac, shots=shots,
                n_qubits=len(qubits), n_hot=n_hot,
                ler_mismatched=ler_mis, ler_oracle=ler_orc,
                errs_mismatched=e1, errs_oracle=e2,
                ratio=(ler_mis / ler_orc if ler_orc > 0 else float('inf')))

if __name__ == "__main__":
    results = []
    for d in [3, 5, 7]:
        for mult in [1.0, 2.0, 4.0, 8.0]:
            shots = 200_000 if d <= 5 else 100_000
            r = run(d=d, p_base=0.003, mult=mult, hot_frac=0.2, shots=shots)
            results.append(r)
            print(f"d={d} mult={mult:>3}: LER mismatched={r['ler_mismatched']:.5f} "
                  f"({r['errs_mismatched']} errs)  oracle={r['ler_oracle']:.5f} "
                  f"({r['errs_oracle']} errs)  ratio={r['ratio']:.2f}", flush=True)
    json.dump(results, open("/home/claude/brittleness_results.json", "w"), indent=1)
