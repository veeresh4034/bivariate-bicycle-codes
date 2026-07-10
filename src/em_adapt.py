import sys, time, json
import numpy as np, stim
from ldpc import BpOsdDecoder
from build_circuit import dem_to_check_matrices

CAL = slice(0, 5000)      # unlabeled calibration shots (observables never touched)
EVA = slice(15000, 20000) # held-out evaluation shots
FLOOR, CAP = 5e-5, 0.45

def key_of(instruction):
    dets, obs = [], []
    for t in instruction.targets_copy():
        if t.is_relative_detector_id(): dets.append(t.val)
        elif t.is_logical_observable_id(): obs.append(t.val)
    return " ".join([f"D{s}" for s in sorted(set(dets))] +
                    [f"L{s}" for s in sorted(set(obs))])

def dem_key_priors(circ_file):
    dem = stim.Circuit.from_file(circ_file).detector_error_model()
    kp = {}
    for ins in dem.flattened():
        if ins.type == "error":
            k = key_of(ins)
            kp[k] = kp.get(k, 0.0) + ins.args_copy()[0]
    return kp

def setup():
    c = stim.Circuit.from_file('bb_believed.stim')
    dem = c.detector_error_model()
    chk, obs_m, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)
    return chk, obs_m, np.asarray(priors, float), col_dict

def make_dec(chk, priors):
    return BpOsdDecoder(chk, channel_probs=list(priors), max_iter=30,
                        bp_method='product_sum', osd_method='osd_cs', osd_order=7)

def decode_all(dec, dets):
    return np.stack([dec.decode(dets[i].astype(np.uint8))
                     for i in range(dets.shape[0])])

def evaluate(priors, label):
    chk, obs_m, _, _ = setup()
    dets = np.load('bb_dets.npy')[EVA]
    obs = np.load('bb_obs.npy')[EVA]
    dec = make_dec(chk, priors)
    t0 = time.time()
    E = decode_all(dec, dets)
    pred = (E @ obs_m.T.toarray().astype(np.uint8)) % 2 if hasattr(obs_m, 'toarray') \
           else (E @ obs_m.T) % 2
    errs = int(np.sum(np.any(pred != obs, axis=1)))
    n = dets.shape[0]
    print(f"EVAL {label}: {errs}/{n}  LER={errs/n:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return errs, n

def em_iteration(priors_in):
    chk, obs_m, _, _ = setup()
    dets = np.load('bb_dets.npy')[CAL]
    dec = make_dec(chk, priors_in)
    t0 = time.time()
    E = decode_all(dec, dets)
    usage = E.mean(axis=0).astype(float)
    p_new = np.clip(usage, FLOOR, CAP)
    print(f"EM iter: {dets.shape[0]} calib shots, mean usage {usage.mean():.5f}, "
          f"({time.time()-t0:.0f}s)", flush=True)
    return p_new

if __name__ == "__main__":
    cmd = sys.argv[1]
    chk, obs_m, p0, col_dict = setup()
    if cmd == "eval_mismatched":
        evaluate(p0, "mismatched(p0)")
    elif cmd == "eval_oracle":
        true_kp = dem_key_priors('bb_true.stim')
        missing = [k for k in col_dict if k not in true_kp]
        assert not missing, f"{len(missing)} DEM keys missing in true model"
        p_orc = np.zeros(len(col_dict))
        for k, idx in col_dict.items():
            p_orc[idx] = true_kp[k]
        np.save('p_oracle.npy', p_orc)
        print(f"oracle priors mapped: {len(col_dict)} cols, "
              f"range [{p_orc.min():.2e}, {p_orc.max():.2e}]")
        evaluate(p_orc, "oracle(true priors)")
    elif cmd == "em1":
        p1 = em_iteration(p0)
        np.save('p_em1.npy', p1)
    elif cmd == "eval_em1":
        evaluate(np.load('p_em1.npy'), "adapted(EM iter1)")
    elif cmd == "em2":
        p2 = em_iteration(np.load('p_em1.npy'))
        np.save('p_em2.npy', p2)
    elif cmd == "eval_em2":
        evaluate(np.load('p_em2.npy'), "adapted(EM iter2)")
