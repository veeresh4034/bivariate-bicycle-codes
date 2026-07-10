"""

Run:      python campaign.py --workers 8          (use your core count)


Each condition runs until its ORACLE arm accumulates `target` logical
errors (the oracle arm is always the statistics bottleneck), in paired
chunks: both decoder arms see identical syndrome samples.

Requires in the same folder: codes_q.py, utils.py, build_circuit.py,
bb_mismatch.py  (all provided in the qec_mismatch/bb package).

pip install stim stimbposd ldpc numpy scipy
"""
import argparse, json, os, time
import multiprocessing as mp
import numpy as np, stim

STATE = "campaign_state.json"

# name: (code, rounds, p, kind, mult, frac, pattern_seed, target_oracle_errs, chunk_shots)
CONDITIONS = {
    # --- P1:
    "r12_144_k10":  ("144", 12, 0.003, "hot", 10.0, 0.05, 42, 500, 250),
    "r6_144_k10":   ("144",  6, 0.003, "hot", 10.0, 0.05, 42, 600, 1000),
    "p002_k10":     ("72",   6, 0.002, "hot", 10.0, 0.05, 42, 400, 2500),
    "p004_k10":     ("72",   6, 0.004, "hot", 10.0, 0.05, 42, 1200, 1000),
    # --- P2: strengthen severity curve & patterns ---
    "sev2_s42":     ("72",   6, 0.003, "hot",  2.0, 0.05, 42, 600, 4000),
    "sev4_s43":     ("72",   6, 0.003, "hot",  4.0, 0.05, 43, 300, 4000),
    "sev4_s44":     ("72",   6, 0.003, "hot",  4.0, 0.05, 44, 300, 4000),
    "sev8_s42":     ("72",   6, 0.003, "hot",  8.0, 0.05, 42, 600, 2000),
    "sev8_s43":     ("72",   6, 0.003, "hot",  8.0, 0.05, 43, 300, 2000),
    "sev8_s44":     ("72",   6, 0.003, "hot",  8.0, 0.05, 44, 300, 2000),
    "sev10_s42":    ("72",   6, 0.003, "hot", 10.0, 0.05, 42, 700, 2000),
    "sev10_s43":    ("72",   6, 0.003, "hot", 10.0, 0.05, 43, 350, 2000),
    "sev10_s44":    ("72",   6, 0.003, "hot", 10.0, 0.05, 44, 350, 2000),
    "f02_k4":       ("72",   6, 0.003, "hot",  4.0, 0.20, 42, 600, 1500),
    "spam3":        ("72",   6, 0.003, "spam", 3.0, 0.0,  42, 300, 4000),
}

# ---------- circuit construction 
_CACHE = {}

def build_pair(cond):
    from codes_q import create_bivariate_bicycle_codes
    from build_circuit import build_circuit
    from bb_mismatch import make_hot_circuit, NOISE_1Q, NOISE_2Q
    code_name, rounds, p, kind, mult, frac, pseed, _, _ = cond
    args = (6, 6, [3], [1, 2], [1, 2], [3]) if code_name == "72" \
        else (12, 6, [3], [1, 2], [1, 2], [3])
    code, A, B = create_bivariate_bicycle_codes(*args)
    believed = build_circuit(code, A, B, p=p, num_repeat=rounds, z_basis=True)
    if kind == "hot":
        qubits = sorted({t.value for inst in believed.flattened()
                         if inst.name in NOISE_1Q | NOISE_2Q
                         for t in inst.targets_copy()})
        rng = np.random.default_rng(pseed)
        hot = set(rng.choice(qubits, size=max(1, int(round(frac * len(qubits)))),
                             replace=False).tolist())
        true_c = make_hot_circuit(believed, hot, mult)
    elif kind == "spam":
        out = stim.Circuit()
        for inst in believed.flattened():
            if inst.name in {"X_ERROR", "Z_ERROR"}:
                pr = inst.gate_args_copy()[0]
                out.append(inst.name, [t.value for t in inst.targets_copy()],
                           min(mult * pr, 0.6))
            else:
                out.append(inst)
        true_c = out
    return believed, true_c

def get_decoders(name, cond):
    if name not in _CACHE:
        from stimbposd import BPOSD
        believed, true_c = build_pair(cond)
        dec_mis = BPOSD(believed.detector_error_model(), max_bp_iters=30,
                        osd_method='osd_cs', osd_order=7)
        dec_orc = BPOSD(true_c.detector_error_model(), max_bp_iters=30,
                        osd_method='osd_cs', osd_order=7)
        _CACHE[name] = (true_c, dec_mis, dec_orc)
    return _CACHE[name]

def run_chunk(job):
    name, chunk_idx = job
    cond = CONDITIONS[name]
    chunk_shots = cond[8]
    true_c, dec_mis, dec_orc = get_decoders(name, cond)
    seed = 10_000_000 + hash(name) % 1_000_000 + chunk_idx  # deterministic
    dets, obs = true_c.compile_detector_sampler(seed=seed).sample(
        chunk_shots, separate_observables=True)
    e_mis = int(np.sum(np.any(dec_mis.decode_batch(dets) != obs, axis=1)))
    e_orc = int(np.sum(np.any(dec_orc.decode_batch(dets) != obs, axis=1)))
    return name, chunk_shots, e_mis, e_orc

# ---------- main loop ----------
def load(): return json.load(open(STATE)) if os.path.exists(STATE) else {}

def done(st, name):
    rec = st.get(name)
    return rec is not None and rec["errs_orc"] >= CONDITIONS[name][7]

def main(workers, hours):
    st = load()
    t_end = time.time() + hours * 3600
    pool = mp.Pool(workers)
    print(f"campaign: {workers} workers, {hours}h budget")
    while time.time() < t_end:
        pending = [n for n in CONDITIONS if not done(st, n)]
        if not pending:
            print("ALL TARGETS MET."); break
        # one batch: a chunk for each pending condition (round-robin keeps arms paired)
        jobs = []
        for n in pending:
            rec = st.get(n, dict(shots=0, errs_mis=0, errs_orc=0, chunks=0))
            st[n] = rec
            jobs.append((n, rec["chunks"]))
            rec["chunks"] += 1
        for name, shots, e_mis, e_orc in pool.imap_unordered(run_chunk, jobs):
            r = st[name]
            r["shots"] += shots; r["errs_mis"] += e_mis; r["errs_orc"] += e_orc
            tgt = CONDITIONS[name][7]
            print(f"  {name:14} {r['shots']:>7} shots | mis {r['errs_mis']:>5} "
                  f"| orc {r['errs_orc']:>5}/{tgt}", flush=True)
        json.dump(st, open(STATE, "w"), indent=1)
    pool.close(); pool.join()
    report()

def report():
    st = load()
    print("\n===== REPORT (95% CI) =====")
    for n, cond in CONDITIONS.items():
        r = st.get(n)
        if not r or r["errs_orc"] == 0:
            print(f"{n:14} -- no data"); continue
        lm, lo = r["errs_mis"] / r["shots"], r["errs_orc"] / r["shots"]
        ratio = lm / lo
        ci = 1.96 * ratio * np.sqrt(1 / r["errs_mis"] + 1 / r["errs_orc"])
        flag = "TARGET MET" if done(st, n) else f"{r['errs_orc']}/{cond[7]} errs"
        print(f"{n:14} LER {lm:.4f}/{lo:.4f}  ratio {ratio:.3f} +/- {ci:.3f}  [{flag}]")
    json.dump({n: st.get(n) for n in CONDITIONS},
              open("campaign_report.json", "w"), indent=1)
    print("saved campaign_report.json - send this file back for the paper update")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        main(a.workers, a.hours)
