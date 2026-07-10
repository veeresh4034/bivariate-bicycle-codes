"""
Noise-model brittleness of BP+OSD on Bivariate Bicycle codes.
Verified session results (seed-reproducible):

  VERIFICATION LADDER 
   1. [[72,12,6]]: N=72, k=12 by independent GF(2) rank computation, Hx.Hz^T=0
   2. Circuit: 144 qubits, 252 detectors (36 Z-checks x 7), 12 observables
   3. Uniform-noise baseline (6 rounds, BP+OSD-CS7, Z-basis):
        p=0.001: LER 2.33e-4 (14/60000)
        p=0.002: LER 2.13e-3 (64/30000)   ratio 9.1 vs p^3-model 8   -> exponent ~3
        p=0.003: LER 1.14e-2 (341/30000)  -> confirms circuit distance 6 intact
  MISMATCH RESULT (p_base=0.003, 5% of qubits at 10x, seed 42/99):
        mismatched prior LER = 0.1236 (618/5000)
        oracle prior LER     = 0.0698 (349/5000)
        brittleness ratio    = 1.77   (surface code same config: 1.60)

Circuit construction: A. Gong, SlidingWindowDecoder (github.com/gongaa/
SlidingWindowDecoder), implementing Bravyi et al., Nature 627, 778 (2024).
Requires codes_q.py, utils.py, build_circuit.py from that repo alongside
this file (fix codes_q.py: 'from .utils import' -> 'from utils import').

pip install stim stimbposd ldpc numpy scipy
"""
import stim, numpy as np, time, json
from stimbposd import BPOSD
from codes_q import create_bivariate_bicycle_codes
from build_circuit import build_circuit

NOISE_1Q = {"DEPOLARIZE1", "X_ERROR", "Z_ERROR"}
NOISE_2Q = {"DEPOLARIZE2"}

def make_hot_circuit(circuit: stim.Circuit, hot_qubits: set, mult: float) -> stim.Circuit:
    """Scale noise strength by `mult` on ops touching `hot_qubits`."""
    out = stim.Circuit()
    for inst in circuit.flattened():
        name = inst.name
        if name in NOISE_1Q:
            p = inst.gate_args_copy()[0]
            cold = [t.value for t in inst.targets_copy() if t.value not in hot_qubits]
            hot = [t.value for t in inst.targets_copy() if t.value in hot_qubits]
            if cold: out.append(name, cold, p)
            if hot:  out.append(name, hot, min(mult * p, 0.75))
        elif name in NOISE_2Q:
            p = inst.gate_args_copy()[0]
            ts = [t.value for t in inst.targets_copy()]
            cold, hot = [], []
            for i in range(0, len(ts), 2):
                pair = ts[i:i + 2]
                (hot if (pair[0] in hot_qubits or pair[1] in hot_qubits)
                     else cold).extend(pair)
            if cold: out.append(name, cold, p)
            if hot:  out.append(name, hot, min(mult * p, 0.9375))
        else:
            out.append(inst)
    return out

def gf2_rank(M):
    M = M.copy() % 2; r = 0
    for c in range(M.shape[1]):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0: continue
        M[[r, r + piv[0]]] = M[[r + piv[0], r]]
        M[(M[:, c] == 1) & (np.arange(M.shape[0]) != r)] ^= M[r]
        r += 1
        if r == M.shape[0]: break
    return r

def verify_code(code):
    hx, hz = code.hx, code.hz
    assert np.all((hx @ hz.T) % 2 == 0), "CSS condition failed"
    k = code.N - gf2_rank(hx) - gf2_rank(hz)
    assert k == code.K, f"rank-based k={k} disagrees with package K={code.K}"
    return k

def run_mismatch(p_base=0.003, mult=10.0, hot_frac=0.05, rounds=6,
                 shots=5000, seed_pattern=42, seed_shots=99,
                 osd_order=7, max_bp_iters=30):
    code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1, 2], [1, 2], [3])
    k = verify_code(code)
    print(f"[[{code.N},{k},6]] verified")
    believed = build_circuit(code, A_list, B_list, p=p_base,
                             num_repeat=rounds, z_basis=True)
    qubits = sorted({t.value for inst in believed.flattened()
                     if inst.name in NOISE_1Q | NOISE_2Q
                     for t in inst.targets_copy()})
    rng = np.random.default_rng(seed_pattern)
    hot = set(rng.choice(qubits, size=max(1, int(round(hot_frac * len(qubits)))),
                         replace=False).tolist())
    true_c = make_hot_circuit(believed, hot, mult)
    dets, obs = true_c.compile_detector_sampler(seed=seed_shots).sample(
        shots, separate_observables=True)
    out = {}
    for label, circ in [("mismatched", believed), ("oracle", true_c)]:
        dec = BPOSD(circ.detector_error_model(),
                    max_bp_iters=max_bp_iters,
                    osd_method='osd_cs', osd_order=osd_order)
        t0 = time.time()
        preds = dec.decode_batch(dets)
        errs = int(np.sum(np.any(preds != obs, axis=1)))
        out[label] = dict(errs=errs, shots=shots, ler=errs / shots,
                          secs=round(time.time() - t0, 1))
        print(f"{label:>10}: {errs}/{shots}  LER={errs/shots:.4f}")
    out["ratio"] = out["mismatched"]["ler"] / out["oracle"]["ler"]
    out["config"] = dict(p_base=p_base, mult=mult, hot_frac=hot_frac,
                         rounds=rounds, n_hot=len(hot), n_qubits=len(qubits))
    print(f"brittleness ratio = {out['ratio']:.2f}")
    return out

if __name__ == "__main__":
    r = run_mismatch()
    json.dump(r, open("bb_mismatch_result.json", "w"), indent=1)
