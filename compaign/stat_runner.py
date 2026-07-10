import sys, json, os, time
import numpy as np, stim
import em_adapt as em

STATE = "/home/stats2.json"
CHUNK_BASE_SEED = 556  # drift chunks: seed = base + chunk_index

def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}

def save_state(st):
    json.dump(st, open(STATE, "w"), indent=1)

def oracle43_priors():
    chk, obs_m, p0, col_dict = em.setup()
    kp = em.dem_key_priors('bb_true43.stim')
    p = np.zeros(len(col_dict))
    for k, idx in col_dict.items():
        p[idx] = kp[k]
    return p

DRIFT_PRIORS = {
    "none":   lambda: em.setup()[2],
    "stale":  lambda: np.load('p_em1.npy'),
    "recal":  lambda: np.load('p_recal.npy'),
    "oracle": oracle43_priors,
}

def drift_chunk(arm, chunk_shots):
    st = load_state()
    key = f"drift_{arm}"
    rec = st.get(key, dict(shots=0, errs=0, chunks=0))
    chk, obs_m, _, _ = em.setup()
    true43 = stim.Circuit.from_file('bb_true43.stim')
    seed = CHUNK_BASE_SEED + rec["chunks"]   # same seed sequence for all arms => paired
    dets, obs = true43.compile_detector_sampler(seed=seed).sample(
        chunk_shots, separate_observables=True)
    dec = em.make_dec(chk, DRIFT_PRIORS[arm]())
    t0 = time.time()
    E = em.decode_all(dec, dets)
    OM = obs_m.toarray().astype(np.uint8) if hasattr(obs_m, 'toarray') else obs_m
    pred = (E @ OM.T) % 2
    errs = int(np.sum(np.any(pred != obs, axis=1)))
    rec["errs"] += errs; rec["shots"] += chunk_shots; rec["chunks"] += 1
    st[key] = rec; save_state(st)
    print(f"{key}: +{errs}/{chunk_shots} -> total {rec['errs']}/{rec['shots']} "
          f"LER={rec['errs']/rec['shots']:.4f} ({time.time()-t0:.0f}s)", flush=True)

def bb144_chunk(arm, chunk_shots):
    from stimbposd import BPOSD
    st = load_state()
    key = f"bb144_{arm}"
    rec = st.get(key, dict(shots=0, errs=0, chunks=0))
    circ_file = 'bb144_believed.stim' if arm == 'mis' else 'bb144_true.stim'
    true_c = stim.Circuit.from_file('bb144_true.stim')
    seed = 700 + rec["chunks"]
    dets, obs = true_c.compile_detector_sampler(seed=seed).sample(
        chunk_shots, separate_observables=True)
    dec = BPOSD(stim.Circuit.from_file(circ_file).detector_error_model(),
                max_bp_iters=30, osd_method='osd_cs', osd_order=7)
    t0 = time.time()
    preds = dec.decode_batch(dets)
    errs = int(np.sum(np.any(preds != obs, axis=1)))
    rec["errs"] += errs; rec["shots"] += chunk_shots; rec["chunks"] += 1
    st[key] = rec; save_state(st)
    print(f"{key}: +{errs}/{chunk_shots} -> total {rec['errs']}/{rec['shots']} "
          f"LER={rec['errs']/rec['shots']:.4f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    task = sys.argv[1]
    if task.startswith("drift_"):
        drift_chunk(task.split("_")[1], int(sys.argv[2]))
    elif task.startswith("bb144_"):
        bb144_chunk(task.split("_")[1], int(sys.argv[2]))
