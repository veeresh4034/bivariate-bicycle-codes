"""
EM adaptive-decoding upgrade run 
Regenerates the headline dataset deterministically (identical seeds to
the paper's pilot), then reruns the EM shot-budget sweep with a larger
15,000-shot evaluation set and extended budgets.

Run: python em_upgrade.py           
Output: em_report.json  

Requires: codes_q.py, utils.py, build_circuit.py, bb_mismatch.py
pip install stim stimbposd ldpc numpy scipy
"""
import json, time
import numpy as np, stim
from ldpc import BpOsdDecoder
from codes_q import create_bivariate_bicycle_codes
from build_circuit import build_circuit, dem_to_check_matrices
from bb_mismatch import make_hot_circuit, NOISE_1Q, NOISE_2Q

BUDGETS = [250, 500, 1000, 2000, 5000, 10000]
N_EVAL = 15000
FLOOR = 5e-5

def key_of(ins):
    dets, obs = [], []
    for t in ins.targets_copy():
        if t.is_relative_detector_id(): dets.append(t.val)
        elif t.is_logical_observable_id(): obs.append(t.val)
    return " ".join([f"D{s}" for s in sorted(set(dets))] +
                    [f"L{s}" for s in sorted(set(obs))])

def main():
    code, A, B = create_bivariate_bicycle_codes(6, 6, [3], [1, 2], [1, 2], [3])
    believed = build_circuit(code, A, B, p=0.003, num_repeat=6, z_basis=True)
    qubits = sorted({t.value for inst in believed.flattened()
                     if inst.name in NOISE_1Q | NOISE_2Q
                     for t in inst.targets_copy()})
    hot = set(np.random.default_rng(42).choice(qubits, 7, replace=False).tolist())
    true_c = make_hot_circuit(believed, hot, 10.0)

    chk, obs_m, p0, col_dict = dem_to_check_matrices(
        believed.detector_error_model(), return_col_dict=True)
    p0 = np.asarray(p0, float)
    OM = obs_m.toarray().astype(np.uint8)
    # oracle priors mapped by DEM signature
    kp = {}
    for ins in true_c.detector_error_model().flattened():
        if ins.type == "error":
            k = key_of(ins); kp[k] = kp.get(k, 0.0) + ins.args_copy()[0]
    p_orc = np.zeros(len(col_dict))
    for k, i in col_dict.items(): p_orc[i] = kp[k]

    # data: calib (seed 99 stream, same as pilot) + big eval (seed 1234)
    calib, _ = true_c.compile_detector_sampler(seed=99).sample(
        max(BUDGETS), separate_observables=True)
    dets_e, obs_e = true_c.compile_detector_sampler(seed=1234).sample(
        N_EVAL, separate_observables=True)

    def make(priors):
        return BpOsdDecoder(chk, channel_probs=list(priors), max_iter=30,
                            bp_method='product_sum', osd_method='osd_cs',
                            osd_order=7)
    def evaluate(priors, label):
        dec = make(priors); t0 = time.time()
        E = np.stack([dec.decode(dets_e[i].astype(np.uint8))
                      for i in range(N_EVAL)])
        errs = int(np.sum(np.any((E @ OM.T) % 2 != obs_e, axis=1)))
        print(f"{label:24} {errs}/{N_EVAL}  LER={errs/N_EVAL:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return errs

    out = {"n_eval": N_EVAL}
    out["mismatched"] = evaluate(p0, "mismatched(p0)")
    out["oracle"] = evaluate(p_orc, "oracle")
    dec0 = make(p0)
    out["adapted"] = {}
    for k in BUDGETS:
        E = np.stack([dec0.decode(calib[i].astype(np.uint8)) for i in range(k)])
        p_em = np.clip(E.mean(axis=0).astype(float), FLOOR, 0.45)
        out["adapted"][k] = evaluate(p_em, f"EM @ {k} calib shots")
    json.dump(out, open("em_report.json", "w"), indent=1)
    print("saved em_report.json")

if __name__ == "__main__":
    main()
