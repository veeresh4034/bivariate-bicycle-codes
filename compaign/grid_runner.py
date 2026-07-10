import sys, json, os, time
import numpy as np, stim
from stimbposd import BPOSD
from codes_q import create_bivariate_bicycle_codes
from build_circuit import build_circuit
from bb_mismatch import make_hot_circuit, NOISE_1Q, NOISE_2Q

STATE = "/home/bb_grid_state.json"

def get_circuits(mult, frac, pattern_seed):
    code, A_list, B_list = create_bivariate_bicycle_codes(6,6,[3],[1,2],[1,2],[3])
    believed = build_circuit(code, A_list, B_list, p=0.003, num_repeat=6, z_basis=True)
    qubits = sorted({t.value for inst in believed.flattened()
                     if inst.name in NOISE_1Q | NOISE_2Q for t in inst.targets_copy()})
    rng = np.random.default_rng(pattern_seed)
    hot = set(rng.choice(qubits, size=max(1,int(round(frac*len(qubits)))),
                         replace=False).tolist())
    return believed, make_hot_circuit(believed, hot, mult)

def run_chunk(mult, frac, pattern_seed, chunk_shots):
    key = f"m{mult}_f{frac}_s{pattern_seed}"
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    rec = st.get(key, dict(shots=0, errs_mis=0, errs_orc=0))
    believed, true_c = get_circuits(mult, frac, pattern_seed)
    shot_seed = 100000 + pattern_seed * 1000 + rec["shots"]  # deterministic, non-repeating
    dets, obs = true_c.compile_detector_sampler(seed=shot_seed).sample(
        chunk_shots, separate_observables=True)
    t0 = time.time()
    for label, circ, field in [("mis", believed, "errs_mis"), ("orc", true_c, "errs_orc")]:
        dec = BPOSD(circ.detector_error_model(), max_bp_iters=30,
                    osd_method='osd_cs', osd_order=7)
        preds = dec.decode_batch(dets)
        rec[field] += int(np.sum(np.any(preds != obs, axis=1)))
    rec["shots"] += chunk_shots
    st[key] = rec
    json.dump(st, open(STATE, "w"), indent=1)
    lm, lo = rec['errs_mis']/rec['shots'], rec['errs_orc']/rec['shots']
    print(f"{key}: {rec['shots']} shots | mis {rec['errs_mis']} ({lm:.4f}) | "
          f"orc {rec['errs_orc']} ({lo:.4f}) | ratio {lm/lo if lo else float('nan'):.2f} "
          f"| +{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    mult, frac, pseed, shots = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    run_chunk(mult, frac, pseed, shots)
