"""T027: which part of pi05 tracks what — RSA, semantics-vs-motion pair regression and time-resolved decoding on the
already-extracted activation / attention trajectories (CPU only, no re-extraction). Pre-registration:
scripts/shortcut/analysis/rsa_decoding_PREREG.md (committed before any result was computed).

Data (pi05 lerobot/pi05_libero_finetuned; per policy call = every 10 env steps; see PREREG §0):
  U40     40 training tasks × init 0–9: S/O/L libero3 run, G cand31 train10 run (A100)
  U90     LIBERO-90 90 tasks × 10 (A100)
  cand    cand31 31 Goal-scene candidates × 10 (A100)
  basket  A100 opp / near (+ ctl = U40 Object), 4090 ctl / far20 / far25
Features per call (T021 definitions + libero90 gemma_task): expert, gemma, gemma_task, siglip, attention,
patch_attention, action. expert / gemma / gemma_task / siglip are per-layer MAGNITUDE summaries (norm, mean |MLP|), not
hidden vectors.

Parts (--parts, default all):
  load   episodes → _cache/eps_<set>.pkl (per-call features, T025 events, motion, first frame)
  emb    (lerobot venv: transformers-free, safetensors + sentencepiece) mean Gemma input embedding per instruction
  sem    semantics.csv (verb / object / destination / qualifier sets from instruction + BDDL goal)
  A      task-level RSA (U40, U90): feature RDM × model RDM Spearman, partial, noise ceiling, Mantel p
  B      episode-pair regression on dissociation sets, variance partitioning, 2×2 tables, basket graded distance
  C      decoding: time-resolved task / suite / verb / object / destination, phase, success, confusion,
         cross-generalisation to cand31 / L90 fallback / basket, basket condition
  D      figures + numbers.json
Run: OPENBLAS_CORETYPE=ZEN $W/cand31/.venv-viz/bin/python rsa_decoding.py [--parts load,sem,A,B,C,D]
     $W/lerobot/.venv/bin/python rsa_decoding.py --parts emb
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # parallelism = NPROC worker processes (task rule: ≤ 8 CPU workers)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

NPROC = 8
K = Path("/outputs/VLA-shortcut-exps/kubectl")
OUT = K / "rsa_decoding"
CACHE = OUT / "_cache"
L3 = K / "libero-spatial-object-long"
GC = K / "libero_goal_composition"
L9 = K / "libero90"
BF = K / "basket/far"
J3 = L3 / "_raw/bench-pi05lib-libero3/jobs"
CKPT = Path("/cache/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/"
            "7152f13e34c6c115d808aea0e5d08aca74eac14b/model.safetensors")
SPM = Path("/cache/pid_demo_dl/paligemma_tok_hf/tokenizer.model")

# set → list of (group, npz dir, job dir, obs dir, variant filter)
SRC = {
    "train": [("S", L3 / "_raw/attn/out/ours-spat-custom-train10", J3 / "ours-spat-custom-train10",
               L3 / "_raw/attn/obs/ours-spat-custom-train10", None),
              ("O", L3 / "_raw/attn/out/ours-obj-custom-train10", J3 / "ours-obj-custom-train10",
               L3 / "_raw/attn/obs/ours-obj-custom-train10", None),
              ("G", GC / "_raw/attn/out/ours-goal-custom-train10",
               GC / "_raw/runs/bench-pi05lib-train10/jobs/ours-goal-custom-train10",
               GC / "_raw/attn/obs/ours-goal-custom-train10", None),
              ("L", L3 / "_raw/attn/out/ours-l10-custom-train10", J3 / "ours-l10-custom-train10",
               L3 / "_raw/attn/obs/ours-l10-custom-train10", None)],
    "cand": [("cand", GC / "_raw/attn/out/ours-goal-custom-cand31",
              GC / "_raw/runs/bench-pi05lib-cand31/jobs/ours-goal-custom-cand31",
              GC / "_raw/attn/obs/ours-goal-custom-cand31", None)],
    "l90": [("l90", L9 / "_raw/attn/out/ours-l90-custom-ctl10", L9 / "_raw/bench-pi05lib-libero90/jobs/ours-l90-custom-ctl10",
             L9 / "_raw/attn/obs/ours-l90-custom-ctl10", None)],
    "bskA": [("opp", L3 / "_raw/attn/out/ours-obj-custom-basket", J3 / "ours-obj-custom-basket",
              L3 / "_raw/attn/obs/ours-obj-custom-basket", "bsk_opp_"),
             ("near", L3 / "_raw/attn/out/ours-obj-custom-basket", J3 / "ours-obj-custom-basket",
              L3 / "_raw/attn/obs/ours-obj-custom-basket", "bsk_near_")],
    "bskV": [("ctlV", BF / "_raw/attn/out/ours-obj-custom-train10",
              BF / "_raw/runs/bench-pi05lib-basket_far/jobs/ours-obj-custom-train10",
              BF / "_raw/attn/obs/ours-obj-custom-train10", None),
             ("far20", BF / "_raw/attn/out/ours-obj-custom-basket",
              BF / "_raw/runs/bench-pi05lib-basket_far/jobs/ours-obj-custom-basket",
              BF / "_raw/attn/obs/ours-obj-custom-basket", "bsk_c20y21_"),
             ("far25", BF / "_raw/attn/out/ours-obj-custom-basket",
              BF / "_raw/runs/bench-pi05lib-basket_far/jobs/ours-obj-custom-basket",
              BF / "_raw/attn/obs/ours-obj-custom-basket", "bsk_c25y20_")],
}
JUDGE = GC / "1_tables/task_judgment.csv"
L9EPS = L9 / "1_results/episodes.csv"
L9META = L9 / "1_results/task_meta.json"

SEGS = ["img_agentview", "img_wrist", "bos", "task", "state", "template", "action"]
TASKSEG = SEGS.index("task")
FEAT_ORDER = ["gemma_task", "gemma", "siglip", "patch_attention", "attention", "expert", "action"]
FEATS = {  # per call; T021 l3_analysis.py FEATS + libero90 l90_analysis.py gemma_task
    "expert": lambda d: np.log(np.concatenate([d["exp_mlp"].mean(1), d["exp_h"].mean(1)], 1)),
    "gemma": lambda d: np.log(np.concatenate([d["vlm_mlp"].reshape(len(d["vlm_mlp"]), -1),
                                              d["vlm_h"].reshape(len(d["vlm_h"]), -1)], 1)),
    "gemma_task": lambda d: np.log(np.concatenate([d["vlm_mlp"][..., TASKSEG], d["vlm_h"][..., TASKSEG]], 1)),
    "siglip": lambda d: np.log(d["sig_h"][:, :2].reshape(len(d["sig_h"]), -1)),
    "attention": lambda d: d["attn_mass"].mean(1).reshape(len(d["attn_mass"]), -1),
    "patch_attention": lambda d: np.sqrt(d["attn_img"][:, :, :512].astype(np.float32).reshape(len(d["attn_img"]), -1)),
    "action": lambda d: d["act_pred"].reshape(len(d["act_pred"]), -1),
}
NPZ_KEYS = ("attn_mass", "attn_img", "exp_mlp", "exp_h", "vlm_h", "vlm_mlp", "sig_h", "act_pred", "t")
LGROUPS = ["all", "g1", "g2", "g3"]  # g1–g3 = L0–5 / 6–11 / 12–17 (siglip 0–8 / 9–17 / 18–26)
LGROUP_NAME = {"all": "all layers", "g1": "early (L0–5)", "g2": "mid (L6–11)", "g3": "late (L12–17)"}


def layer_cols(feat, g):
    """column indices of layer group g (1..3) in the per-call feature vector; None = no layer axis."""
    r = [range(0, 6), range(6, 12), range(12, 18)][g - 1]
    if feat == "expert" or feat == "gemma_task":
        return np.array([l for l in r] + [18 + l for l in r])
    if feat == "gemma":
        return np.array([l * 6 + s for l in r for s in range(6)] + [108 + l * 6 + s for l in r for s in range(6)])
    if feat == "attention":
        return np.array([l * 7 + s for l in r for s in range(7)])
    if feat == "siglip":
        rs = [range(0, 9), range(9, 18), range(18, 27)][g - 1]
        return np.array([i * 27 + l for i in range(2) for l in rs])
    if feat == "patch_attention":
        return np.arange((g - 1) * 512, g * 512)
    return None


# ---------------------------------------------------------------- events (T025 suite_phase_traj.py, unchanged)

FIXTURES = {"wooden_cabinet_1", "white_cabinet_1", "flat_stove_1", "microwave_1", "wine_rack_1", "desk_caddy_1"}
GAP_MERGE, MIN_HOLD, RELEASE_OPEN = 10, 5, 3


def goal_targets(goal):
    out = []
    for p in goal:
        for a in p[1:]:
            m = re.match(r"^(.*?_\d+)", a)
            if m and m.group(1) not in out and not a.startswith("main_table"):
                out.append(m.group(1))
    return out


def events_from_steps(st, nc, targets=()):
    g, cl, ct = st["grasped"], st["gripper_closing"].astype(bool), st["contacts"]
    n = len(g)
    held = []
    for t in range(n):
        objs = [o for o in str(g[t]).split("|") if o and o not in FIXTURES]
        held.append(objs[0] if objs and cl[t] else "")
    two = [set(str(x).split("|")) for x in g]
    runs = []
    for t, o in enumerate(held):
        if not o:
            continue
        if runs and runs[-1][2] == o and t - runs[-1][1] - 1 <= GAP_MERGE:
            runs[-1][1] = t
        elif runs and runs[-1][2] == o and t <= runs[-1][1]:
            continue
        else:
            runs.append([t, t, o])
        while runs[-1][1] + 1 < n and o in two[runs[-1][1] + 1]:
            runs[-1][1] += 1
    runs = [r for r in runs if r[1] - r[0] + 1 >= MIN_HOLD]
    grasps, releases = [], []
    for s, e, o in runs:
        grasps.append((s, o))
        r = e + 1
        if r < n:
            kind = "open" if not cl[r:r + RELEASE_OPEN].all() else "slip"
            releases.append((r, o, kind))
    contact = next(((t, f) for t, c in enumerate(ct) for f in str(c).split("|") if f in targets), (None, None))
    if contact[0] is None:
        contact = next(((t, str(c).split("|")[0]) for t, c in enumerate(ct) if c), (None, None))
    call = lambda t: None if t is None else min(t // 10, nc - 1)  # noqa: E731
    gcs = [call(s) for s, _ in grasps]
    return dict(grasp_t=[s for s, _ in grasps], gcs=gcs, release_t=[r for r, _, _ in releases],
                rcs=[call(r) for r, _, _ in releases], contact_t=contact[0], cc=call(contact[0]),
                gc=gcs[0] if gcs else None, bound=gcs[0] if gcs else call(contact[0]))


# ---------------------------------------------------------------- load


def resample(x, n):
    """(T, D) → (n, D) by linear interpolation over normalised time (T = 1 → repeated)."""
    x = np.asarray(x, np.float32)
    if len(x) == 1:
        return np.repeat(x, n, 0)
    t, tt = np.linspace(0, 1, len(x)), np.linspace(0, 1, n)
    return np.stack([np.interp(tt, t, x[:, j]) for j in range(x.shape[1])], 1).astype(np.float32)


def load_one(job):
    """one episode: per-call features, events, motion (eef + gripper and recorded action, bound-aligned), first frame."""
    rec, npz, steps, obs = job
    z = np.load(npz)
    d = {k: z[k] for k in NPZ_KEYS}
    nc = len(d["t"])
    with np.errstate(divide="ignore", invalid="ignore"):
        F = {f: fn(d).astype(np.float32) for f, fn in FEATS.items()}
    for f in F:
        F[f][~np.isfinite(F[f])] = np.nan
    st = np.load(steps)
    ev = events_from_steps(st, nc, rec["targets"])
    n = len(st["eef_pos"])
    bstep = ev["grasp_t"][0] if ev["grasp_t"] else (ev["contact_t"] if ev["contact_t"] is not None else n // 2)
    bstep = int(min(max(bstep, 1), n - 2))
    m = np.concatenate([st["eef_pos"], 10 * st["gripper_width"][:, None]], 1)
    a = st["action"]
    rec.update(ev, nc=nc, n_steps=n, bstep=bstep,
               bound_call=ev["bound"] if ev["bound"] is not None else nc // 2,
               motion=np.concatenate([resample(m[:bstep + 1], 10), resample(m[bstep:], 10)]),
               motion_act=np.concatenate([resample(a[:bstep + 1], 10), resample(a[bstep:], 10)]),
               feats=F)
    im = np.load(obs)["image"][0].astype(np.float32) / 255.0  # (360, 360, 3) call-0 agentview
    rec["pix"] = im.reshape(60, 6, 60, 6, 3).mean((1, 3)).astype(np.float32)
    return rec


def l90_meta():
    meta = json.loads(L9META.read_text())
    eps = {(r["task"], r["ep"]): r for r in csv.DictReader(open(L9EPS))}
    lscene = {}
    for t, m in meta.items():
        for L in m["scene_L"]:
            lscene[L] = m["scene"]
    return meta, eps, lscene


def cmd_load():
    CACHE.mkdir(parents=True, exist_ok=True)
    judge = {f"{r['cand']}_{r['ep']}": r for r in csv.DictReader(open(JUDGE))}
    meta, l9eps, lscene = l90_meta()
    for sname, groups in SRC.items():
        p = CACHE / f"eps_{sname}.pkl"
        if p.exists():
            print(f"[load] {sname}: cached", flush=True)
            continue
        jobs = []
        for grp, nd, jd, od, vf in groups:
            rows = [json.loads(x) for x in (jd / "episodes.jsonl").read_text().splitlines()]
            rows = [e for e in rows if vf is None or e["variant"].startswith(vf)]
            rows.sort(key=lambda e: (int(e["task_id"]), int(e["init_idx"])))
            for e in rows:
                v, i = e["variant"], int(e["init_idx"])
                name = f"{v}_i{i:02d}"
                if not (nd / f"{name}.npz").exists():
                    print(f"[load] missing npz {nd.name}/{name}", flush=True)
                    continue
                tid = int(e["task_id"])
                rec = dict(set=sname, group=grp, variant=v, init=i, name=name, instruction=e["instruction"],
                           success=bool(e["success"]), tid=tid, targets=goal_targets(e["variant_meta"]["goal"]))
                if sname == "train":
                    rec["task"] = f"{grp}{tid + 1}"
                    rec["scene"] = {"S": "SPATIAL", "O": "OBJECT", "G": "GOAL"}.get(grp) or lscene[rec["task"]]
                    rec["exec"] = rec["task"]
                elif sname == "cand":
                    j = judge[name]
                    rec["task"] = v
                    rec["base"] = f"G{tid + 1}"
                    rec["old"] = [f"G{t}" for t in json.loads(j["old"])]
                    rec["cat"] = j["cat"]
                    rec["success"] = j["cat"] in ("new", "new_byproduct")
                    rec["scene"] = "GOAL"
                    rec["exec_other"] = [t for t in rec["old"] if t != rec["base"]]
                    rec["exec"] = "+".join(rec["old"]) if j["cat"] in ("trained", "fail_trained") else v
                elif sname == "l90":
                    t = f"t{tid:02d}"
                    r = l9eps[(t, f"i{i:02d}")]
                    rec.update(task=t, scene=meta[t]["scene"], kind=meta[t]["kind"], outcome=r["outcome"],
                               reached=r["reached_instead"].split(), nearest_trained=meta[t]["nearest_trained"],
                               scene_L=meta[t]["scene_L"])
                    rec["exec_L"] = [u for u in rec["reached"] if u.startswith("L")]
                    rec["exec"] = rec["exec_L"][0] if rec["exec_L"] else t
                else:  # basket
                    rec.update(task=f"O{tid + 1}", scene="OBJECT", cond=grp)
                    rec["exec"] = f"O{tid + 1}:{grp}"
                jobs.append((rec, nd / f"{name}.npz", jd / e["steps_file"], od / f"{name}.npz"))
        with get_context("fork").Pool(NPROC) as pool:
            eps = pool.map(load_one, jobs, chunksize=4)
        pickle.dump(eps, open(p, "wb"), protocol=4)
        print(f"[load] {sname}: {len(eps)} episodes, calls {Counter(e['nc'] for e in eps)}", flush=True)


def load_set(name):
    return pickle.load(open(CACHE / f"eps_{name}.pkl", "rb"))


# ---------------------------------------------------------------- semantics


OBJ_NAME = {"akita_black_bowl": "black bowl", "white_bowl": "white bowl", "porcelain_mug": "white mug",
            "white_yellow_mug": "yellow-white mug", "red_coffee_mug": "red mug", "chefmate_8_frypan": "frying pan",
            "black_book": "book", "yellow_book": "book", "new_salad_dressing": "salad dressing"}


def obj_name(a):
    b = re.sub(r"_\d+$", "", a)
    return OBJ_NAME.get(b, b.replace("_", " "))


def dest_of(rel, a):
    """(relation, destination) token of a BDDL On/In second argument."""
    if a.endswith("_contain_region"):
        base = re.sub(r"_\d+_.*", "", a)
        part = re.search(r"desk_caddy_1_(\w+?)_contain", a)
        return f"in:{'caddy ' + part.group(1) if part else obj_name(base)}"
    m = re.match(r"(wooden|white)_cabinet_1_(top|middle|bottom)_region", a)
    if m:
        return f"in:drawer {m.group(2)}"
    if a.endswith("_top_side"):
        return f"on top of:{'shelf' if 'shelf' in a else 'cabinet'}"
    if a.startswith("wooden_two_layer_shelf_1_"):
        return "in:shelf" if "top_region" in a else "under:shelf"
    if a == "flat_stove_1_cook_region":
        return "on:stove"
    if a == "wine_rack_1_top_region":
        return "on:rack"
    if a == "microwave_1_heating_region":
        return "in:microwave"
    m = re.match(r"\w+?_table_(\w+?)_(front|left|right)_region", a)
    if m:
        ref = {"stove": "stove", "porcelain_mug": "white mug", "plate": "plate", "desk_caddy": "caddy"}[m.group(1)]
        return f"{m.group(2) + ' of' if m.group(2) != 'front' else 'front of'}:{ref}"
    return f"{'on' if rel == 'On' else 'in'}:{obj_name(a)}"


QUAL_PATTERNS = [r"(?<=pick up the black bowl )[\w ]+?(?= and (?:place|put))",  # Spatial source location, t60
                 r"(?<=pick up the book )[\w ]+?(?= and place)",  # t86–t89
                 r"at the (?:back|front)", r"\bmiddle\b(?! drawer)", r"\b(?:left|right) (?:plate|bowl|moka pot)"]


def semantics(instr, goal):
    s = instr.lower()
    verbs = set()
    for v, pat in [("open", r"\bopen\b"), ("close", r"\bclose\b"), ("turn_on", r"\bturn on\b"),
                   ("turn_off", r"\bturn off\b"), ("push", r"\bpush\b"), ("place", r"\b(put|place|stack)\b")]:
        if re.search(pat, s):
            verbs.add(v)
    if "pick up" in s and "place" not in verbs:
        verbs.add("lift")
    objs, dests, amb = [], [], []
    for p in goal:
        pr = p[0]
        if pr in ("On", "In"):
            objs.append(obj_name(p[1]))
            dests.append(dest_of(pr, p[2]))
        elif pr == "Up":
            objs.append(obj_name(p[1]))
        elif pr in ("Open", "Close"):
            objs.append("microwave" if p[1].startswith("microwave") else "drawer")
        elif pr in ("TurnOn", "TurnOff"):
            objs.append("stove")
    goal_verbs = {"TurnOn": "turn_on", "TurnOff": "turn_off", "Open": "open", "Close": "close"}
    for p in goal:
        gv = goal_verbs.get(p[0])
        if gv and gv not in verbs:
            amb.append(f"BDDL goal has {gv}, instruction does not (instruction used)")
            objs = [o for o in objs if not (gv.startswith("turn") and o == "stove")]
    for v, o in [("open", "drawer"), ("close", "drawer")]:  # "open the top drawer and put the bowl inside"
        if v in verbs and "drawer" in s and o not in objs:
            objs.append(o)
    if "microwave" in s and ("close" in verbs or "open" in verbs) and "microwave" not in objs:
        objs.append("microwave")
    if "in the bowl" in s and any(d == "on:black bowl" for d in dests):
        amb.append("BDDL On(x, bowl) but instruction says 'in the bowl' (BDDL relation used)")
    if "stack" in s:
        amb.append("stack = place on an object")
    if "push" in s:
        amb.append("push: BDDL goal is On(region) like put")
    found = sorted((m.start(), m.group(0)) for pat in QUAL_PATTERNS for m in re.finditer(pat, s))
    qual = list(dict.fromkeys(q for _, q in found))  # in sentence order: "left bowl on the right bowl" ≠ reverse
    if any(q.startswith("left") for q in qual) and any(q.startswith("right") for q in qual):
        amb.append("both left and right qualifiers")
    words = sorted(set(re.findall(r"[a-z]+", s)) - {"the", "a", "it", "and"})
    return dict(verb=sorted(verbs), object=sorted(set(objs)), dest=sorted(set(dests)), qual=qual, words=words,
                ambiguous="; ".join(dict.fromkeys(amb)))


def instructions():
    """every task / candidate: (label, set, instruction, goal) for U40, cand31, U90."""
    out = {}
    for sname in ("train", "cand", "l90"):
        for grp, nd, jd, od, vf in SRC[sname]:
            for x in (jd / "episodes.jsonl").read_text().splitlines():
                e = json.loads(x)
                tid = int(e["task_id"])
                lab = f"{grp}{tid + 1}" if sname == "train" else (e["variant"] if sname == "cand" else f"t{tid:02d}")
                out.setdefault(lab, (sname, e["instruction"], e["variant_meta"]["goal"]))
    return out


def cmd_sem():
    rows = []
    emb = {}
    if (CACHE / "emb.npz").exists():
        z = np.load(CACHE / "emb.npz", allow_pickle=True)
        emb = dict(zip(z["instr"].tolist(), z["vec"]))
    for lab, (sname, instr, goal) in instructions().items():
        s = semantics(instr, goal)
        rows.append(dict(task=lab, set=sname, instruction=instr, goal=json.dumps(goal),
                         verb="|".join(s["verb"]), object="|".join(s["object"]), destination="|".join(s["dest"]),
                         qualifier="|".join(s["qual"]), words="|".join(s["words"]), ambiguous=s["ambiguous"],
                         has_embedding=int(instr in emb)))
    with open(OUT / "semantics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[sem] {len(rows)} rows, ambiguous {sum(bool(r['ambiguous']) for r in rows)}", flush=True)


def cmd_emb():
    """lerobot venv only (safetensors + sentencepiece): mean pi05 Gemma input embedding of each instruction's tokens."""
    import sentencepiece as spm
    from safetensors import safe_open

    sp = spm.SentencePieceProcessor(model_file=str(SPM))
    instr = sorted({v[1] for v in instructions().values()})
    ids = {s: sp.encode(s.strip().lower()) for s in instr}
    need = sorted({i for v in ids.values() for i in v})
    with safe_open(str(CKPT), "pt") as f:  # lm_head is tied to embed_tokens in PaliGemma
        sl = f.get_slice("model.paligemma_with_expert.paligemma.lm_head.weight")
        E = {i: sl[i:i + 1].float().numpy()[0] for i in need}
    vec = np.stack([np.mean([E[i] for i in ids[s]], 0) for s in instr]).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "emb.npz", instr=np.array(instr), vec=vec)
    print(f"[emb] {len(instr)} instructions, {len(need)} distinct tokens, dim {vec.shape[1]}", flush=True)


def read_sem():
    S = {}
    for r in csv.DictReader(open(OUT / "semantics.csv")):
        S[r["task"]] = {k: set(filter(None, r[k].split("|"))) for k in ("verb", "object", "destination", "qualifier", "words")}
        S[r["task"]]["instruction"] = r["instruction"]
    z = np.load(CACHE / "emb.npz", allow_pickle=True)
    emb = dict(zip(z["instr"].tolist(), z["vec"]))
    for v in S.values():
        v["emb"] = emb[v["instruction"]]
    return S


def jacc(a, b):
    return 0.0 if not a and not b else 1 - len(a & b) / len(a | b)


SEM_MODELS = ["sem_verb", "sem_object", "sem_dest", "sem_qual", "sem_word", "sem_emb", "sem"]


def sem_rdms(labels, S):
    n = len(labels)
    R = {m: np.zeros((n, n)) for m in SEM_MODELS}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            A, B = S[a], S[b]
            R["sem_verb"][i, j] = jacc(A["verb"], B["verb"])
            R["sem_object"][i, j] = jacc(A["object"], B["object"])
            R["sem_dest"][i, j] = jacc(A["destination"], B["destination"])
            R["sem_qual"][i, j] = float(A["qualifier"] != B["qualifier"])
            R["sem_word"][i, j] = jacc(A["words"], B["words"])
            ea, eb = A["emb"], B["emb"]
            R["sem_emb"][i, j] = 1 - ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb))
    R["sem"] = (R["sem_verb"] + R["sem_object"] + R["sem_dest"] + R["sem_qual"]) / 4
    return R


# ---------------------------------------------------------------- shared feature handling

PHASES = ["pre", "post", "traj"]
NBIN = 5  # bins per phase


def zfit(eps, feat):
    """per-dimension z-score fitted on all calls of eps; NaN → 0 (= the fit mean); near-constant columns → 0."""
    flat = np.concatenate([e["feats"][feat] for e in eps])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sd = np.nanmean(flat, 0), np.nanstd(flat, 0)
    mu, sd = np.nan_to_num(mu), np.nan_to_num(sd)
    sd = np.where(sd > 1e-6, sd, np.inf)
    return lambda x: np.nan_to_num((x - mu) / sd).astype(np.float32)


def phase_vec(Xz, b, phase):
    pre, post = Xz[:b + 1], Xz[b + 1:] if b + 1 < len(Xz) else Xz[-1:]
    if phase == "pre":
        return resample(pre, NBIN).ravel()
    if phase == "post":
        return resample(post, NBIN).ravel()
    return np.concatenate([resample(pre, NBIN).ravel(), resample(post, NBIN).ravel()])


def ep_matrix(eps, feat, phase, group, z):
    cols = None if group == "all" else layer_cols(feat, int(group[1]))
    out = []
    for e in eps:
        X = z(e["feats"][feat])
        if cols is not None:
            X = X[:, cols]
        out.append(phase_vec(X, e["bound_call"], phase))
    return np.stack(out)


def motion_vec(e, phase, key="motion"):
    m = e[key]
    return {"pre": m[:10], "post": m[10:], "traj": m}[phase].ravel()


def groups_of(feat):
    return ["all"] if layer_cols(feat, 1) is None else LGROUPS


def eucl(A, B=None):
    B = A if B is None else B
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * np.einsum("id,jd->ij", A, B)
    return np.sqrt(np.maximum(d, 0))


# ---------------------------------------------------------------- statistics


def rankv(x):
    from scipy.stats import rankdata
    return rankdata(x)


def corr_rows(Y, x):
    Yc = Y - Y.mean(1, keepdims=True)
    xc = x - x.mean()
    return (Yc @ xc) / (np.sqrt((Yc ** 2).sum(1)) * np.sqrt((xc ** 2).sum()) + 1e-12)


def resid(C, Y):
    """residuals of the columns of Y (npairs, k) on C (npairs, m) incl. intercept."""
    C1 = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(C1, Y, rcond=None)
    return Y - C1 @ beta


def perm_stack(F, perms, mask=None):
    """values of F at the (permuted) pair positions: (len(perms), npairs). mask = pair mask (upper triangle)."""
    iu = np.nonzero(np.triu(np.ones_like(F, bool), 1) if mask is None else mask)
    return np.stack([F[p][:, p][iu] for p in perms]), iu


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# ---------------------------------------------------------------- A: task-level RSA

MOTION_MODELS = ["motion", "motion_act"]
SCENE_MODELS = ["scene_id", "scene_pix"]
MODELS = SEM_MODELS + MOTION_MODELS + SCENE_MODELS + ["len_words"]  # len_words: exploratory nuisance (not in PREREG)
PARTIALS = {"sem": ("motion", "scene_pix"), "motion": ("sem", "scene_pix"), "scene_pix": ("sem", "motion")}
# exploratory (added after seeing A; PREREG change log): motion as recorded action sequence, verb alone, embedding
# controlled for sentence length
PARTIALS_X = {"sem|motion_act": ("sem", ("motion_act", "scene_pix")),
              "motion_act|sem": ("motion_act", ("sem", "scene_pix")),
              "sem_verb|motion_act": ("sem_verb", ("motion_act", "scene_pix")),
              "motion_act|sem_verb": ("motion_act", ("sem_verb", "scene_pix")),
              "sem_emb|motion,scene,len": ("sem_emb", ("motion", "motion_act", "scene_pix", "len_words")),
              "sem_word|motion,scene,len": ("sem_word", ("motion", "motion_act", "scene_pix", "len_words"))}
NPERM = 1000


def unit_eps(unit):
    if unit == "U40":
        return load_set("train")
    return load_set("l90")


def task_models(eps, labels, S, phase):
    idx = {t: [i for i, e in enumerate(eps) if e["task"] == t] for t in labels}
    R = sem_rdms(labels, S)
    for key, name in (("motion", "motion"), ("motion_act", "motion_act")):
        M = np.stack([np.mean([motion_vec(eps[i], phase, key) for i in idx[t]], 0) for t in labels])
        R[name] = eucl(M)
    sc = [eps[idx[t][0]]["scene"] for t in labels]
    R["scene_id"] = np.array([[float(a != b) for b in sc] for a in sc])
    P = np.stack([np.mean([eps[i]["pix"].ravel() for i in idx[t]], 0) for t in labels])
    R["scene_pix"] = eucl(P)
    nw = np.array([len(S[t]["instruction"].split()) for t in labels], float)
    R["len_words"] = np.abs(nw[:, None] - nw[None])
    return R


def rsa_cell(job):
    unit, feat, phase, group = job
    eps, labels, models, z = _A[unit]["eps"], _A[unit]["labels"], _A[unit]["models"][phase], _A[unit]["z"][feat]
    V = ep_matrix(eps, feat, phase, group, z)
    D = V.shape[1]
    ti = np.array([labels.index(e["task"]) for e in eps])
    ev = np.array([e["init"] % 2 == 0 for e in eps])
    MA = np.stack([V[(ti == k) & ev].mean(0) for k in range(len(labels))])
    MB = np.stack([V[(ti == k) & ~ev].mean(0) for k in range(len(labels))])
    G = np.einsum("id,jd->ij", MA, MB)
    F = (np.diag(G)[:, None] + np.diag(G)[None] - G - G.T) / D  # cross-validated squared Euclidean
    n = len(labels)
    iu = np.triu_indices(n, 1)
    from scipy.stats import spearmanr
    rA, rB = eucl(MA)[iu], eucl(MB)[iu]
    r_sh = spearmanr(rA, rB)[0]
    ceiling = 2 * r_sh / (1 + r_sh) if r_sh > -1 else np.nan
    rng = np.random.default_rng(0)
    perms = [np.arange(n)] + [rng.permutation(n) for _ in range(NPERM)]
    Fr = np.zeros((n, n))
    Fr[iu] = rankv(F[iu])
    Fr = Fr + Fr.T
    Y, _ = perm_stack(Fr, perms)
    res = dict(unit=unit, feat=feat, phase=phase, group=group, dim=D, split_half=float(r_sh), ceiling=float(ceiling),
               rho={}, p={}, partial={}, p_partial={})
    mr = {m: rankv(models[m][iu]) for m in MODELS}
    for m in MODELS:
        r = corr_rows(Y, mr[m])
        res["rho"][m] = float(r[0])
        res["p"][m] = float(((np.abs(r[1:]) >= abs(r[0]) - 1e-12).sum() + 1) / (NPERM + 1))
    for m, cov in PARTIALS.items():
        C = np.column_stack([mr[c] for c in cov])
        x = resid(C, mr[m][:, None])[:, 0]
        Yr = resid(C, Y.T).T
        r = corr_rows(Yr, x)
        res["partial"][m] = float(r[0])
        res["p_partial"][m] = float(((np.abs(r[1:]) >= abs(r[0]) - 1e-12).sum() + 1) / (NPERM + 1))
    res["partial_x"], res["p_partial_x"] = {}, {}
    for key, (m, cov) in PARTIALS_X.items():
        C = np.column_stack([mr[c] for c in cov])
        x = resid(C, mr[m][:, None])[:, 0]
        r = corr_rows(resid(C, Y.T).T, x)
        res["partial_x"][key] = float(r[0])
        res["p_partial_x"][key] = float(((np.abs(r[1:]) >= abs(r[0]) - 1e-12).sum() + 1) / (NPERM + 1))
    return res


_A = {}


def cmd_A():
    S = read_sem()
    jobs = []
    for unit in ("U40", "U90"):
        eps = unit_eps(unit)
        labels = sorted({e["task"] for e in eps}, key=task_key)
        _A[unit] = dict(eps=eps, labels=labels, z={f: zfit(eps, f) for f in FEATS},
                        models={ph: task_models(eps, labels, S, ph) for ph in PHASES})
        np.savez(CACHE / f"models_{unit}.npz", labels=np.array(labels),
                 **{f"{ph}__{m}": _A[unit]["models"][ph][m] for ph in PHASES for m in MODELS})
        jobs += [(unit, f, ph, g) for f in FEAT_ORDER for ph in PHASES for g in groups_of(f)]
    with get_context("fork").Pool(NPROC) as pool:
        res = pool.map(rsa_cell, jobs, chunksize=1)
    # model-RDM inter-correlations (how separable semantics / motion / scene are)
    from scipy.stats import spearmanr
    inter = {}
    for unit in ("U40", "U90"):
        M = _A[unit]["models"]["traj"]
        iu = np.triu_indices(len(_A[unit]["labels"]), 1)
        inter[unit] = {a: {b: round(float(spearmanr(M[a][iu], M[b][iu])[0]), 3) for b in MODELS} for a in MODELS}
    json.dump(dict(cells=res, model_intercorr=inter), open(CACHE / "A.json", "w"))
    print(f"[A] {len(res)} cells", flush=True)


def task_key(t):
    m = re.match(r"([A-Za-z]+)(\d+)", t)
    return ("SOGLgt".index(m.group(1)[0]) if m.group(1)[0] in "SOGLgt" else 9, int(m.group(2)))


# ---------------------------------------------------------------- B: episode-pair regression

BSETS = ["basket_A100", "basket_4090", "L90_lscene", "cand31", "spatial"]
NPERM_B = 999


def bset(name):
    """episodes, block id (pairs only within a block), semantic label (row of semantics.csv) per episode."""
    tr = load_set("train")
    if name == "basket_A100":
        eps = [dict(e, cond="ctlA") for e in tr if e["group"] == "O"] + load_set("bskA")
        for e in eps:
            if e["cond"] == "ctlA":
                e["exec"] = f"{e['task']}:ctlA"
    elif name == "basket_4090":
        eps = load_set("bskV")
    elif name == "L90_lscene":
        l9 = [e for e in load_set("l90") if e["scene_L"]]
        scenes = {e["scene"] for e in l9}
        eps = [e for e in tr if e["group"] == "L" and e["scene"] in scenes] + l9
    elif name == "cand31":
        eps = [e for e in tr if e["group"] == "G"] + load_set("cand")
    else:
        eps = [e for e in tr if e["group"] == "S"]
    block = np.array([e["scene"] for e in eps])
    return eps, block


def b_design(eps, block, S, phase):
    labels = sorted({e["task"] for e in eps}, key=task_key)
    R = sem_rdms(labels, S)["sem"]
    li = np.array([labels.index(e["task"]) for e in eps])
    d_sem = R[li][:, li]
    d_mot = eucl(np.stack([motion_vec(e, phase) for e in eps]))
    d_sc = eucl(np.stack([e["pix"].ravel() for e in eps]))
    mask = np.triu(block[:, None] == block[None], 1)
    return d_sem, d_mot, d_sc, mask


def b_cell(job):
    name, feat, phase = job
    eps, block, S = _B[name]["eps"], _B[name]["block"], _B[name]["S"]
    d_sem, d_mot, d_sc, mask = _B[name]["design"][phase]
    z = zfit(eps, feat)
    V = ep_matrix(eps, feat, phase, "all", z)
    F = eucl(V) / np.sqrt(V.shape[1])
    iu = np.nonzero(mask)
    X = np.column_stack([d_sem[iu], d_mot[iu], d_sc[iu]])
    sdx = X.std(0)
    usable = sdx > 1e-9
    Xs = (X[:, usable] - X[:, usable].mean(0)) / sdx[usable]
    names = np.array(["sem", "motion", "scene"])[usable]
    y = F[iu]
    ys = (y - y.mean()) / y.std()
    X1 = np.column_stack([np.ones(len(ys)), Xs])
    beta = np.linalg.lstsq(X1, ys, rcond=None)[0][1:]

    def r2(cols):
        if not cols:
            return 0.0
        A = np.column_stack([np.ones(len(ys))] + [Xs[:, c] for c in cols])
        e = ys - A @ np.linalg.lstsq(A, ys, rcond=None)[0]
        return 1 - (e @ e) / (ys @ ys)

    k = Xs.shape[1]
    full = r2(list(range(k)))
    uniq = {names[c]: full - r2([x for x in range(k) if x != c]) for c in range(k)}
    # Mantel permutation: permute episodes within blocks
    rng = np.random.default_rng(1)
    blocks = [np.nonzero(block == b)[0] for b in np.unique(block)]
    pinv = np.linalg.pinv(X1)[1:]
    cnt = np.zeros(k)
    for s in range(0, NPERM_B, 50):
        Yp = []
        for _ in range(min(50, NPERM_B - s)):
            p = np.arange(len(eps))
            for bi in blocks:
                p[bi] = rng.permutation(bi)
            yp = F[p][:, p][iu]
            Yp.append((yp - yp.mean()) / yp.std())
        Bp = pinv @ np.stack(Yp, 1)
        cnt += (np.abs(Bp) >= np.abs(beta)[:, None] - 1e-12).sum(1)
    pval = (cnt + 1) / (NPERM_B + 1)
    # 2×2: same instruction × same executed behaviour
    lab = np.array([e["task"] for e in eps])
    ex = np.array([e["exec"] for e in eps])
    si, se = (lab[:, None] == lab[None])[iu], (ex[:, None] == ex[None])[iu]
    tab = {f"{'same' if a else 'diff'}_instr__{'same' if b else 'diff'}_motion":
           (float(y[(si == a) & (se == b)].mean() / y.mean()) if ((si == a) & (se == b)).any() else None,
            int(((si == a) & (se == b)).sum())) for a in (True, False) for b in (True, False)}
    return dict(set=name, feat=feat, phase=phase, n_eps=len(eps), n_pairs=int(len(ys)),
                beta={names[c]: float(beta[c]) for c in range(k)}, p={names[c]: float(pval[c]) for c in range(k)},
                r2=float(full), unique=uniq, shared=float(full - sum(uniq.values())), table2x2=tab)


_B = {}


def basket_graded():
    """distance of each moved trial to the same task·init ctl of the same machine ÷ ctl other-init same-task baseline."""
    out = {}
    rng = np.random.default_rng(2)
    for name, ctl, moved in (("basket_A100", "ctlA", ["near", "opp"]), ("basket_4090", "ctlV", ["far20", "far25"])):
        eps, _ = bset(name)
        key = {(e["cond"], e["task"], e["init"]): i for i, e in enumerate(eps)}
        tasks = sorted({e["task"] for e in eps}, key=task_key)
        for phase in PHASES:
            mats = {f: None for f in FEAT_ORDER}
            for f in FEAT_ORDER:
                z = zfit(eps, f)
                V = ep_matrix(eps, f, phase, "all", z)
                mats[f] = eucl(V) / np.sqrt(V.shape[1])
            mats["MOTION"] = eucl(np.stack([motion_vec(e, phase) for e in eps]))
            for f, F in mats.items():
                for c in moved:
                    num, den = [], []
                    for t in tasks:
                        for i in range(10):
                            a, b = key.get((c, t, i)), key.get((ctl, t, i))
                            if a is None or b is None:
                                continue
                            others = [key[(ctl, t, j)] for j in range(10) if j != i and (ctl, t, j) in key]
                            num.append(F[a, b])
                            den.append(F[b, others].mean())
                    num, den = np.array(num), np.array(den)
                    bs = [num[s].mean() / den[s].mean() for s in rng.integers(0, len(num), (1000, len(num)))]
                    out[f"{name}|{c}|{f}|{phase}"] = dict(ratio=float(num.mean() / den.mean()),
                                                         ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                         n=int(len(num)))
    return out


def cmd_B():
    S = read_sem()
    jobs = []
    for name in BSETS:
        eps, block = bset(name)
        _B[name] = dict(eps=eps, block=block, S=S, design={ph: b_design(eps, block, S, ph) for ph in PHASES})
        jobs += [(name, f, ph) for f in FEAT_ORDER for ph in PHASES]
    with get_context("fork").Pool(NPROC) as pool:
        res = pool.map(b_cell, jobs, chunksize=1)
    graded = basket_graded()
    json.dump(dict(cells=res, basket_graded=graded), open(CACHE / "B.json", "w"))
    print(f"[B] {len(res)} cells", flush=True)


# ---------------------------------------------------------------- C: decoding

NORM = 20
REL = np.arange(-10, 16)
MIN_REL = 150  # rel-call points with fewer training episodes are skipped
PHASE4 = ["reach", "grasp", "carry", "release"]
_C = {}


def clf(n_train, dim):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), PCA(n_components=min(50, n_train - 1, dim), random_state=0),
                         StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def cv_pred(X, y, groups, n_splits=None):
    """out-of-fold predictions; folds = leave-one-group-out (or GroupKFold(n_splits)); PCA fitted inside each fold."""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    cv = LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y, groups):
        if len(set(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        pred[te] = clf(len(tr), X.shape[1]).fit(X[tr], y[tr]).predict(X[te])
    return pred


def score(y, pred):
    from sklearn.metrics import balanced_accuracy_score
    return dict(acc=float(np.mean(pred == y)), bacc=float(balanced_accuracy_score(y, pred)), n=int(len(y)),
                k=int(len(set(y))), chance=float(1 / len(set(y))))


def primary(S, t, key):
    v = sorted(S[t][key])
    return v[0] if v else "none"


def prep_C():
    S = read_sem()
    tr = load_set("train")
    _C["S"] = S
    _C["tr"] = tr
    _C["z"] = {f: zfit(tr, f) for f in FEATS}
    _C["lab"] = dict(task=np.array([e["task"] for e in tr]), suite=np.array([e["task"][0] for e in tr]),
                     init=np.array([e["init"] for e in tr]))
    for key, nm in (("verb", "verb"), ("object", "object"), ("destination", "dest")):
        y = np.array([primary(S, e["task"], key) for e in tr])
        ntask = Counter(lab for lab in dict(zip(_C["lab"]["task"], y)).values())
        _C["lab"][nm] = np.where([ntask[v] >= 2 for v in y], y, None)
    sets = dict(tr=tr, cand=load_set("cand"), l90=load_set("l90"), bskA=load_set("bskA"), bskV=load_set("bskV"))
    _C["sets"] = sets
    _C["calls"] = {f: {s: [_C["z"][f](e["feats"][f]) for e in E] for s, E in sets.items()} for f in FEATS}
    _C["norm"], _C["rel"], _C["relmask"], _C["pp"] = {}, {}, {}, {}
    for f in FEATS:
        C = _C["calls"][f]["tr"]
        _C["norm"][f] = np.stack([resample(x, NORM) for x in C])
        R = np.zeros((len(C), len(REL), C[0].shape[1]), np.float32)
        M = np.zeros((len(C), len(REL)), bool)
        for i, (x, e) in enumerate(zip(C, tr)):
            for j, r in enumerate(REL):
                c = e["bound_call"] + r
                if 0 <= c < len(x):
                    R[i, j], M[i, j] = x[c], True
        _C["rel"][f], _C["relmask"][f] = R, M
        _C["pp"][f] = {s: np.stack([np.stack([x[:e["bound_call"] + 1].mean(0),
                                              (x[e["bound_call"] + 1:] if e["bound_call"] + 1 < len(x) else x[-1:]).mean(0)])
                                    for x, e in zip(_C["calls"][f][s], sets[s])]) for s in ("tr", "l90")}
    # per-call 4-phase labels (training episodes with a grasp and a later release)
    ph = []
    for e in tr:
        lab = [None] * e["nc"]
        if e["gc"] is not None:
            rc = next((r for r in e["rcs"] if r is not None and r > e["gc"]), None)
            for c in range(e["nc"]):
                if c < e["gc"] - 2:
                    lab[c] = "reach"
                elif c <= e["gc"] + 1:
                    lab[c] = "grasp"
                elif rc is not None and c < rc - 1:
                    lab[c] = "carry"
                elif rc is not None and c <= rc + 2:
                    lab[c] = "release"
                elif rc is None:
                    lab[c] = "carry"
        ph.append(lab)
    _C["phase4"] = ph


def dec_job(job):
    kind, f, axis, t = job
    L = _C["lab"]
    if axis == "norm":
        X, ok = _C["norm"][f][:, t], np.ones(len(L["task"]), bool)
    elif axis == "rel":
        X, ok = _C["rel"][f][:, t], _C["relmask"][f][:, t].copy()
    else:
        X, ok = _C["pp"][f]["tr"][:, t], np.ones(len(L["task"]), bool)
    out = dict(kind=kind, feat=f, axis=axis, t=int(t))
    if kind in ("task40", "suite"):
        if ok.sum() < MIN_REL:
            return None
        y = L["task" if kind == "task40" else "suite"][ok]
        pred = cv_pred(X[ok], y, L["init"][ok])
        out.update(score(y, pred))
        if axis == "pp" and kind == "task40":
            out["y"], out["pred"] = y.tolist(), pred.tolist()
    elif kind == "within":
        accs = {}
        for s in "SOGL":
            m = ok & (L["suite"] == s)
            if m.sum() < 40:
                continue
            accs[s] = score(L["task"][m], cv_pred(X[m], L["task"][m], L["init"][m]))["acc"]
        if not accs:
            return None
        out.update(acc=float(np.mean(list(accs.values()))), by_suite=accs, chance=0.1)
    else:  # verb / object / dest: generalise to held-out tasks
        m = ok & (L[kind] != None)  # noqa: E711
        if m.sum() < MIN_REL:
            return None
        y = L[kind][m].astype(str)
        out.update(score(y, cv_pred(X[m], y, L["task"][m])))
    return out


def phase_job(f):
    X, y, g = [], [], []
    for x, e, lab in zip(_C["calls"][f]["tr"], _C["tr"], _C["phase4"]):
        for c, l in enumerate(lab):
            if l is not None:
                X.append(x[c])
                y.append(l)
                g.append(e["init"])
    X, y, g = np.stack(X), np.array(y), np.array(g)
    pred = cv_pred(X, y, g)
    cm = [[int(((y == a) & (pred == b)).sum()) for b in PHASE4] for a in PHASE4]
    return dict(feat=f, **score(y, pred), confusion=cm)


def success_job(job):
    f, axis, t = job
    E = _C["sets"]["l90"]
    if axis == "pp":
        X = _C["pp"][f]["l90"][:, t]
    else:
        X = np.stack([resample(x, NORM)[t] for x in _C["calls"][f]["l90"]])
    y = np.array([str(e["success"]) for e in E])
    g = np.array([e["task"] for e in E])
    return dict(feat=f, axis=axis, t=int(t), **score(y, cv_pred(X, y, g, n_splits=5)))


def xgen_job(job):
    kind, f, c = job
    tr, L = _C["tr"], _C["lab"]
    Ctr = _C["calls"][f]["tr"]
    out = dict(kind=kind, feat=f, call=int(c))
    if kind == "cand":
        idx = [i for i, e in enumerate(tr) if e["group"] == "G"]
        X, y = np.stack([Ctr[i][c] for i in idx]), L["task"][idx]
        out["loio_acc"] = score(y, cv_pred(X, y, L["init"][idx]))["acc"]
        model = clf(len(y), X.shape[1]).fit(X, y)
        E = _C["sets"]["cand"]
        sel = [i for i, e in enumerate(E) if e["exec_other"]]
        pred = model.predict(np.stack([_C["calls"][f]["cand"][i][c] for i in sel]))
        out["executed"] = float(np.mean([p in E[i]["exec_other"] for p, i in zip(pred, sel)]))
        out["instructed"] = float(np.mean([p == E[i]["base"] for p, i in zip(pred, sel)]))
        out["n"] = len(sel)
        allp = model.predict(np.stack([x[c] for x in _C["calls"][f]["cand"]]))
        out["all_base_share"] = float(np.mean([p == e["base"] for p, e in zip(allp, E)]))
    elif kind == "l90":
        idx = [i for i, e in enumerate(tr) if e["nc"] > c]
        X, y = np.stack([Ctr[i][c] for i in idx]), L["task"][idx]
        model = clf(len(y), X.shape[1]).fit(X, y)
        E = _C["sets"]["l90"]
        sel = [i for i, e in enumerate(E) if e["kind"] == "lscene" and e["outcome"] == "trained" and e["exec_L"]]
        pred = model.predict(np.stack([_C["calls"][f]["l90"][i][c] for i in sel]))
        dis = [k for k, i in enumerate(sel) if E[i]["nearest_trained"] not in E[i]["exec_L"]]
        out["executed_all"] = float(np.mean([p in E[i]["exec_L"] for p, i in zip(pred, sel)]))
        out["executed"] = float(np.mean([pred[k] in E[sel[k]]["exec_L"] for k in dis]))
        out["instructed"] = float(np.mean([pred[k] == E[sel[k]]["nearest_trained"] for k in dis]))
        out["n"], out["n_all"] = len(dis), len(sel)
    elif kind == "basket":
        X, y = np.stack([x[c] for x in Ctr]), L["task"]
        model = clf(len(y), X.shape[1]).fit(X, y)
        o = [i for i, e in enumerate(tr) if e["group"] == "O"]
        pred_cv = cv_pred(X, y, L["init"])
        out["ctlA"] = float(np.mean(pred_cv[o] == y[o]))
        out["ctlA_suiteO"] = float(np.mean([str(p).startswith("O") for p in pred_cv[o]]))
        for s in ("bskA", "bskV"):
            E = _C["sets"][s]
            pred = model.predict(np.stack([x[c] for x in _C["calls"][f][s]]))
            for cond in sorted({e["cond"] for e in E}):
                ii = [i for i, e in enumerate(E) if e["cond"] == cond]
                out[cond] = float(np.mean([pred[i] == E[i]["task"] for i in ii]))
                out[cond + "_suiteO"] = float(np.mean([str(pred[i]).startswith("O") for i in ii]))
    else:  # basket condition, 3-way within machine, leave-one-init-out
        if kind == "bcond_A100":
            E = [e for e in tr if e["group"] == "O"] + _C["sets"]["bskA"]
            Xs = [Ctr[i] for i, e in enumerate(tr) if e["group"] == "O"] + _C["calls"][f]["bskA"]
            y = np.array(["ctl"] * 100 + [e["cond"] for e in _C["sets"]["bskA"]])
        else:
            E = _C["sets"]["bskV"]
            Xs = _C["calls"][f]["bskV"]
            y = np.array(["ctl" if e["cond"] == "ctlV" else e["cond"] for e in E])
        X = np.stack([x[c] for x in Xs])
        out.update(score(y, cv_pred(X, y, np.array([e["init"] for e in E]))))
    return out


def cmd_C():
    prep_C()
    jobs = [(k, f, "norm", t) for k in ("task40", "suite", "within", "verb", "object", "dest") for f in FEAT_ORDER
            for t in range(NORM)]
    jobs += [(k, f, "rel", t) for k in ("task40", "suite") for f in FEAT_ORDER for t in range(len(REL))]
    jobs += [(k, f, "pp", t) for k in ("task40", "suite", "within", "verb", "object", "dest") for f in FEAT_ORDER
             for t in range(2)]
    xjobs = [(k, f, c) for f in FEAT_ORDER for k, nc in (("cand", 30), ("l90", 28), ("basket", 28),
                                                         ("bcond_A100", 28), ("bcond_4090", 28)) for c in range(nc)]
    sjobs = [(f, "pp", t) for f in FEAT_ORDER for t in range(2)] + [(f, "norm", t) for f in FEAT_ORDER for t in range(NORM)]
    with get_context("fork").Pool(NPROC) as pool:
        dec = [r for r in pool.map(dec_job, jobs, chunksize=2) if r is not None]
        print(f"[C] decoding {len(dec)}", flush=True)
        ph = pool.map(phase_job, FEAT_ORDER, chunksize=1)
        suc = pool.map(success_job, sjobs, chunksize=2)
        print("[C] phase, success done", flush=True)
        xg = pool.map(xgen_job, xjobs, chunksize=2)
    # confusion vs model RDMs (task40, pre / post means)
    z = np.load(CACHE / "models_U40.npz")
    labels = [str(x) for x in z["labels"]]
    from scipy.stats import spearmanr
    conf = {}
    iu = np.triu_indices(40, 1)
    for r in dec:
        if r["axis"] == "pp" and r["kind"] == "task40":
            M = np.zeros((40, 40))
            for a, b in zip(r.pop("y"), r.pop("pred")):
                M[labels.index(a), labels.index(b)] += 1
            M /= M.sum(1, keepdims=True)
            Cs = (M + M.T) / 2
            ph_name = ["pre", "post"][r["t"]]
            conf[f"{r['feat']}|{ph_name}"] = dict(
                matrix=M.round(3).tolist(),
                rho={m: float(spearmanr(Cs[iu], z[f"{ph_name}__{m}"][iu])[0]) for m in MODELS},
                top_pairs=sorted(([labels[i], labels[j], round(float(Cs[i, j]), 3)] for i, j in zip(*iu) if Cs[i, j] > 0),
                                 key=lambda x: -x[2])[:12])
    json.dump(dict(decoding=dec, phase4=ph, success=suc, xgen=xg, confusion=conf, labels=labels),
              open(CACHE / "C.json", "w"))
    print(f"[C] done: {len(dec)} decoding, {len(xg)} cross-gen", flush=True)


# ---------------------------------------------------------------- D: figures, numbers, hypothesis verdicts

FHUE = {"gemma_task": "#2a78d6", "gemma": "#6250d6", "siglip": "#1baf7a", "patch_attention": "#008300",
        "attention": "#eda100", "expert": "#eb6834", "action": "#e34948"}  # reference palette slots 1,7,3,6,4,2,8
FLS = {"gemma_task": "-", "gemma": "--", "siglip": ":", "patch_attention": "-.", "attention": (0, (5, 1, 1, 1)),
       "expert": "-", "action": "--"}  # secondary encoding
FMK = {"gemma_task": "o", "gemma": "s", "siglip": "^", "patch_attention": "D", "attention": "v", "expert": "P",
       "action": "X"}
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e4e3df"
MODEL_LABEL = {"sem_verb": "(a) verb", "sem_object": "(b) object", "sem_dest": "(c) destination",
               "sem_qual": "(d) qualifier", "sem_word": "(e) words", "sem_emb": "(f) Gemma emb.", "sem": "sem = mean(a–d)",
               "motion": "motion (eef+grip)", "motion_act": "motion (actions)", "scene_id": "scene id",
               "scene_pix": "scene pixels", "len_words": "length (words)*"}


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})
    return plt


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bo", ["#2a78d6", "#f0efec", "#eb6834"])


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue", ["#f7f9fc", "#9ec5f4", "#2a78d6", "#0d366b"])


def fig_rsa(A, q, out):
    plt = plt_setup()
    for unit in ("U40", "U90"):
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
        for ax, ph in zip(axes, PHASES):
            M = np.array([[next(c for c in A if c["unit"] == unit and c["feat"] == f and c["phase"] == ph
                                and c["group"] == "all")["rho"][m] for m in MODELS] for f in FEAT_ORDER])
            ax.imshow(M, cmap=div_cmap(), vmin=-0.9, vmax=0.9, aspect="auto")
            for i, f in enumerate(FEAT_ORDER):
                for j, m in enumerate(MODELS):
                    sig = q[f"A|{unit}|{f}|{ph}|all|rho|{m}"] < 0.05
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                            color=INK if sig else MUTED, fontweight="bold" if sig else "normal")
            ax.set_xticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS], rotation=55, ha="right")
            ax.set_yticks(range(len(FEAT_ORDER)), FEAT_ORDER)
            ax.set_title(f"{unit}: phase = {ph}", loc="left", fontsize=9)
            ax.axvline(6.5, color=INK2, lw=0.8)
            ax.axvline(8.5, color=INK2, lw=0.8)
            ax.axvline(10.5, color=INK2, lw=0.8)
        fig.suptitle(f"Fig 1 — {unit}: Spearman ρ between each feature's task RDM and each model RDM "
                     "(bold = BH-FDR q < 0.05; * exploratory)", fontsize=10, x=0.01, ha="left")
        fig.subplots_adjust(left=0.07, right=0.995, bottom=0.3, top=0.86, wspace=0.04)
        fig.savefig(out / f"fig1_rsa_heatmap_{unit}.png", dpi=130)
        plt.close(fig)
    # partial correlations, pre-registered (traj, all layers) + layer groups
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    keys = [("sem", "semantics | motion, scene"), ("motion", "motion | semantics, scene"),
            ("scene_pix", "scene | semantics, motion")]
    hue = {"sem": "#2a78d6", "motion": "#eb6834", "scene_pix": "#1baf7a"}
    for r, unit in enumerate(("U40", "U90")):
        ax = axes[r, 0]
        x = np.arange(len(FEAT_ORDER))
        for k, (m, lab) in enumerate(keys):
            v = [next(c for c in A if c["unit"] == unit and c["feat"] == f and c["phase"] == "traj"
                      and c["group"] == "all")["partial"][m] for f in FEAT_ORDER]
            bars = ax.bar(x + (k - 1) * 0.26, v, 0.24, color=hue[m], label=lab)
            for b, f in zip(bars, FEAT_ORDER):
                if q[f"A|{unit}|{f}|traj|all|partial|{m}"] < 0.05:
                    ax.text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0) + 0.01, "•", ha="center",
                            fontsize=9, color=INK)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, FEAT_ORDER)
        ax.set_ylabel("partial Spearman ρ")
        ax.set_title(f"{unit}: pre-registered partial correlations (phase traj, all layers; • q < 0.05)",
                     loc="left", fontsize=9)
        clean(ax)
        ax.legend(frameon=False, fontsize=7)
        ax = axes[r, 1]
        xk = [("sem|motion_act", "semantics | action seq., scene", "#2a78d6"),
              ("motion_act|sem", "action seq. | semantics, scene", "#eb6834"),
              ("sem_verb|motion_act", "verb | action seq., scene", "#6250d6"),
              ("sem_emb|motion,scene,len", "Gemma emb. | motion, scene, length", "#1baf7a")]
        for k, (m, lab, h) in enumerate(xk):
            v = [next(c for c in A if c["unit"] == unit and c["feat"] == f and c["phase"] == "traj"
                      and c["group"] == "all").get("partial_x", {}).get(m, np.nan) for f in FEAT_ORDER]
            ax.bar(x + (k - 1.5) * 0.2, v, 0.18, color=h, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, FEAT_ORDER)
        ax.set_title(f"{unit}: exploratory partial correlations (added after seeing A)", loc="left", fontsize=9)
        clean(ax)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Fig 1b — what each feature tracks once the other model RDMs are partialled out", fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig1b_rsa_partial.png", dpi=130)
    plt.close(fig)
    # layer groups
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    for ax, unit in zip(axes, ("U40", "U90")):
        rows = [(f, g) for f in FEAT_ORDER for g in groups_of(f) if g != "all"]
        M = np.array([[next(c for c in A if c["unit"] == unit and c["feat"] == f and c["phase"] == "traj"
                            and c["group"] == g)["partial"][m] for f, g in rows] for m, _ in keys])
        ax.imshow(M, cmap=div_cmap(), vmin=-0.7, vmax=0.7, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(rows)), [f"{f} {g[1]}" for f, g in rows], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(3), [lab for _, lab in keys])
        ax.set_title(f"{unit}: partial ρ by layer group (1 = L0–5, 2 = L6–11, 3 = L12–17; siglip 0–8 / 9–17 / 18–26)",
                     loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1c_rsa_layers.png", dpi=130)
    plt.close(fig)


def fig_pairs(B, graded, q, out):
    plt = plt_setup()
    fig, axes = plt.subplots(1, len(BSETS), figsize=(19, 4.2), sharey=True)
    hue = {"sem": "#2a78d6", "motion": "#eb6834", "scene": "#1baf7a"}
    for ax, s in zip(axes, BSETS):
        x = np.arange(len(FEAT_ORDER))
        for k, m in enumerate(("sem", "motion", "scene")):
            v = [next(c for c in B if c["set"] == s and c["feat"] == f and c["phase"] == "traj")["beta"].get(m, np.nan)
                 for f in FEAT_ORDER]
            bars = ax.bar(x + (k - 1) * 0.26, v, 0.24, color=hue[m], label={"sem": "β semantics", "motion": "β motion",
                                                                             "scene": "β scene"}[m])
            for b, f in zip(bars, FEAT_ORDER):
                key = f"B|{s}|{f}|traj|{m}"
                if key in q and q[key] < 0.05:
                    ax.text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0) + 0.01, "•", ha="center", fontsize=8)
        n = next(c for c in B if c["set"] == s)
        ax.set_title(f"{s}  ({n['n_eps']} episodes, {n['n_pairs']} pairs)", loc="left", fontsize=8.5)
        ax.set_xticks(x, FEAT_ORDER, rotation=50, ha="right")
        ax.axhline(0, color=INK2, lw=0.8)
        clean(ax)
    axes[0].set_ylabel("standardised β")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Fig 2 — episode pairs: feature distance ~ semantic + motion + scene distance (phase traj; "
                 "• = Mantel permutation, BH-FDR q < 0.05)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig2_pair_beta.png", dpi=130)
    plt.close(fig)
    # 2×2: same instruction / different motion vs different instruction / same motion
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, s in zip(axes, ("L90_lscene", "cand31", "basket_A100")):
        for i, f in enumerate(FEAT_ORDER):
            t = next(c for c in B if c["set"] == s and c["feat"] == f and c["phase"] == "traj")["table2x2"]
            cells = ["same_instr__same_motion", "same_instr__diff_motion", "diff_instr__same_motion",
                     "diff_instr__diff_motion"]
            v = [t[c][0] if t[c][0] is not None else np.nan for c in cells]
            ax.plot(range(4), v, color=FHUE[f], ls=FLS[f], marker=FMK[f], ms=6, lw=1.6, label=f)
        ax.set_xticks(range(4), ["same instr.\nsame motion", "same instr.\ndiff. motion", "diff. instr.\nsame motion",
                                 "diff. instr.\ndiff. motion"])
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.set_title(s, loc="left", fontsize=9)
        clean(ax)
    axes[0].set_ylabel("mean feature distance ÷ mean over all pairs")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Fig 2b — the dissociation cells: lower = closer. Instruction-tracking features drop in column 2, "
                 "motion-tracking features in column 3", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig2b_pair_2x2.png", dpi=130)
    plt.close(fig)
    # Fig 6: basket graded
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    order = [("basket_A100", "near"), ("basket_4090", "far20"), ("basket_4090", "far25"), ("basket_A100", "opp")]
    for ax, ph in zip(axes, PHASES):
        for f in FEAT_ORDER:
            xs = [1.0] + [graded[f"{m}|{c}|MOTION|{ph}"]["ratio"] for m, c in order]
            ys = [1.0] + [graded[f"{m}|{c}|{f}|{ph}"]["ratio"] for m, c in order]
            lo = [0] + [graded[f"{m}|{c}|{f}|{ph}"]["ratio"] - graded[f"{m}|{c}|{f}|{ph}"]["ci"][0] for m, c in order]
            hi = [0] + [graded[f"{m}|{c}|{f}|{ph}"]["ci"][1] - graded[f"{m}|{c}|{f}|{ph}"]["ratio"] for m, c in order]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=FHUE[f], ls=FLS[f], marker=FMK[f], ms=5, lw=1.5, capsize=2,
                        label=f)
        for (m, c), x in zip(order, [graded[f"{m}|{c}|MOTION|{ph}"]["ratio"] for m, c in order]):
            ax.text(x, 0.02, c, ha="center", fontsize=7, color=INK2, transform=ax.get_xaxis_transform(), rotation=90,
                    va="bottom")
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("motion distance to same task·init ctl ÷ ctl other-init baseline")
        ax.set_title(f"phase = {ph}", loc="left", fontsize=9)
        clean(ax)
    axes[0].set_ylabel("feature distance ratio (same definition)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Fig 6 — basket: same instruction, the robot moves differently (near < far20 < far25 < opp). "
                 "ctl = (1, 1); 95% bootstrap CI over task·init", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig6_basket_graded.png", dpi=130)
    plt.close(fig)


def curves(C, kind, axis):
    d = defaultdict(dict)
    for r in C["decoding"]:
        if r["kind"] == kind and r["axis"] == axis:
            d[r["feat"]][r["t"]] = (r.get("bacc", r["acc"]), r["chance"])
    return d


def fig_decoding(C, out):
    plt = plt_setup()
    panels = [("task40", "norm", "task identity, all 40 (leave-one-init-out)"),
              ("within", "norm", "task within suite (10-way, mean of 4 suites)"),
              ("suite", "norm", "suite (4-way)"),
              ("task40", "rel", "task identity, all 40 — grasp-aligned"),
              ("verb", "norm", "verb — held-out tasks (balanced acc.)"),
              ("object", "norm", "object — held-out tasks (balanced acc.)"),
              ("dest", "norm", "destination — held-out tasks (balanced acc.)")]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, (kind, axis, title) in zip(axes.flat, panels):
        d = curves(C, kind, axis)
        for f in FEAT_ORDER:
            ts = sorted(d[f])
            xs = [REL[t] for t in ts] if axis == "rel" else [(t + 0.5) / NORM for t in ts]
            ax.plot(xs, [d[f][t][0] for t in ts], color=FHUE[f], ls=FLS[f], lw=1.6, label=f)
        ch = next(iter(d[FEAT_ORDER[0]].values()))[1]
        ax.axhline(ch, color=MUTED, lw=1, ls=":")
        ax.text(ax.get_xlim()[1], ch, " chance", va="bottom", ha="right", fontsize=7, color=MUTED)
        if axis == "rel":
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_xlabel("call relative to first grasp (or contact)")
        else:
            ax.set_xlabel("normalised time in episode")
        ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left", fontsize=9)
        clean(ax)
    axes[0, 0].legend(frameon=False, fontsize=7)
    ax = axes[1, 3]
    ph = {r["feat"]: r for r in C["phase4"]}
    suc = defaultdict(dict)
    for r in C["success"]:
        if r["axis"] == "norm":
            suc[r["feat"]][r["t"]] = r["bacc"]
    for f in FEAT_ORDER:
        ax.plot([(t + 0.5) / NORM for t in range(NORM)], [suc[f][t] for t in range(NORM)], color=FHUE[f], ls=FLS[f],
                lw=1.6, label=f"{f} (phase-4 {ph[f]['bacc']:.2f})")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("LIBERO-90 success vs failure (held-out tasks, bal. acc.)\nlegend: 4-phase decoding bal. acc. (chance .25)",
                 loc="left", fontsize=9)
    ax.set_xlabel("normalised time in episode")
    ax.legend(frameon=False, fontsize=6.5)
    clean(ax)
    fig.suptitle("Fig 3 — time-resolved linear decoding (PCA ≤ 50 + logistic regression, PCA fitted on the training "
                 "fold only)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig3_decoding_time.png", dpi=130)
    plt.close(fig)


def xg_table(C):
    x = defaultdict(dict)
    for r in C["xgen"]:
        x[(r["kind"], r["feat"])][r["call"]] = r
    return x


def fig_xgen(C, out):
    plt = plt_setup()
    x = xg_table(C)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for col, (kind, title) in enumerate([("cand", "cand31: 116 trials that did another training task — Goal 10-way decoder"),
                                         ("l90", "LIBERO-90 lscene: 78 'did the scene's L10 task' trials — 40-way decoder")]):
        for row, key in enumerate(("executed", "instructed")):
            ax = axes[row, col]
            for f in FEAT_ORDER:
                cs = sorted(x[(kind, f)])
                ax.plot(cs, [x[(kind, f)][c][key] for c in cs], color=FHUE[f], ls=FLS[f], lw=1.6, label=f)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("policy call (×10 env steps)")
            ax.set_ylabel(f"share decoded as the {key} task")
            ax.set_title(("" if row else title + "\n") + f"→ {key} task" +
                         (" (= base task of the candidate)" if kind == "cand" and key == "instructed" else
                          " (= training task with the most similar instruction)" if key == "instructed" else ""),
                         loc="left", fontsize=8.5)
            clean(ax)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for col, conds, title in ((2, ("near", "opp"), "basket A100: share decoded as its own O task (40-way)"),
                              (3, ("ctlV", "far20", "far25"), "basket 4090: same")):
        ax = axes[0, col]
        for f in FEAT_ORDER:
            cs = sorted(x[("basket", f)])
            for k, cond in enumerate(conds):
                ax.plot(cs, [x[("basket", f)][c][cond] for c in cs], color=FHUE[f], lw=1.4,
                        ls=["-", "--", ":"][k], label=f"{f}" if k == 0 else None)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title + "\n" + " / ".join(f"{c} {['—', '– –', '···'][k]}" for k, c in enumerate(conds)),
                     loc="left", fontsize=8.5)
        ax.set_xlabel("policy call")
        clean(ax)
    for col, kind, title in ((2, "bcond_A100", "basket condition ctl / near / opp (A100), 3-way, bal. acc."),
                             (3, "bcond_4090", "basket condition ctl / far20 / far25 (4090), 3-way, bal. acc.")):
        ax = axes[1, col]
        for f in FEAT_ORDER:
            cs = sorted(x[(kind, f)])
            ax.plot(cs, [x[(kind, f)][c]["bacc"] for c in cs], color=FHUE[f], ls=FLS[f], lw=1.6, label=f)
        ax.axhline(1 / 3, color=MUTED, lw=1, ls=":")
        ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("policy call")
        clean(ax)
    fig.suptitle("Fig 4 — cross-generalisation: decoders trained on the training episodes, applied to trials whose "
                 "instruction and executed behaviour differ", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig4_xgen.png", dpi=130)
    plt.close(fig)


def fig_confusion(C, out):
    plt = plt_setup()
    labels = C["labels"]
    feats = ["action", "expert", "siglip"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    for j, f in enumerate(feats):
        for i, ph in enumerate(("pre", "post")):
            ax = axes[i, j]
            M = np.array(C["confusion"][f"{f}|{ph}"]["matrix"])
            ax.imshow(M, cmap=seq_cmap(), vmin=0, vmax=1)
            for b in (9.5, 19.5, 29.5):
                ax.axhline(b, color=INK2, lw=0.6)
                ax.axvline(b, color=INK2, lw=0.6)
            ax.set_xticks(range(40), labels, rotation=90, fontsize=5.5)
            ax.set_yticks(range(40), labels, fontsize=5.5)
            rho = C["confusion"][f"{f}|{ph}"]["rho"]
            acc = np.trace(M) / 40
            ax.set_title(f"{f}, {ph}-grasp mean: acc {acc:.2f}; ρ(confusion, model distance): "
                         f"action seq. {rho['motion_act']:+.2f}, sem {rho['sem']:+.2f}, scene {rho['scene_pix']:+.2f}",
                         loc="left", fontsize=7.5)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
    fig.suptitle("Fig 5 — 40-task confusion matrices (leave-one-init-out; rows sum to 1). Negative ρ = confused pairs "
                 "are close in that model", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig5_confusion.png", dpi=120)
    plt.close(fig)


def cmd_D():
    A_all = json.load(open(CACHE / "A.json"))
    A = A_all["cells"]
    Bj = json.load(open(CACHE / "B.json"))
    B, graded = Bj["cells"], Bj["basket_graded"]
    C = json.load(open(CACHE / "C.json"))
    # FDR families: (1) A pre-registered ρ + partials, (2) A exploratory partials, (3) B β
    fam = {"A": {}, "Ax": {}, "B": {}}
    for c in A:
        base = f"A|{c['unit']}|{c['feat']}|{c['phase']}|{c['group']}"
        for m, p in c["p"].items():
            if m != "len_words":
                fam["A"][f"{base}|rho|{m}"] = p
            else:
                fam["Ax"][f"{base}|rho|{m}"] = p
        for m, p in c["p_partial"].items():
            fam["A"][f"{base}|partial|{m}"] = p
        for m, p in c.get("p_partial_x", {}).items():
            fam["Ax"][f"{base}|partial_x|{m}"] = p
    for c in B:
        for m, p in c["p"].items():
            fam["B"][f"B|{c['set']}|{c['feat']}|{c['phase']}|{m}"] = p
    q = {}
    for k, d in fam.items():
        keys = list(d)
        for kk, qq in zip(keys, bh([d[x] for x in keys])):
            q[kk] = float(qq)
    out = OUT
    fig_rsa(A, q, out)
    fig_pairs(B, graded, q, out)
    fig_decoding(C, out)
    fig_xgen(C, out)
    fig_confusion(C, out)
    # ---- hypothesis verdicts (pre-registered cells)
    cell = lambda u, f: next(c for c in A if c["unit"] == u and c["feat"] == f and c["phase"] == "traj"  # noqa: E731
                             and c["group"] == "all")
    H = {}
    for u in ("U40", "U90"):
        for f in ("gemma", "gemma_task"):
            c = cell(u, f)
            H[f"H-A1|{u}|{f}"] = dict(partial_sem=round(c["partial"]["sem"], 3), partial_motion=round(c["partial"]["motion"], 3),
                                      q_sem=round(q[f"A|{u}|{f}|traj|all|partial|sem"], 4),
                                      supported=bool(c["partial"]["sem"] > 0 and q[f"A|{u}|{f}|traj|all|partial|sem"] < 0.05
                                                     and c["partial"]["sem"] > c["partial"]["motion"]))
        for f in ("expert", "action"):
            c = cell(u, f)
            H[f"H-A2|{u}|{f}"] = dict(partial_motion=round(c["partial"]["motion"], 3), partial_sem=round(c["partial"]["sem"], 3),
                                      q_motion=round(q[f"A|{u}|{f}|traj|all|partial|motion"], 4),
                                      supported=bool(c["partial"]["motion"] > 0
                                                     and q[f"A|{u}|{f}|traj|all|partial|motion"] < 0.05
                                                     and c["partial"]["motion"] > c["partial"]["sem"]))
        for f in ("siglip", "patch_attention"):
            c = cell(u, f)
            H[f"H-A3|{u}|{f}"] = dict(**{m: round(c["partial"][m], 3) for m in PARTIALS},
                                      supported=bool(max(PARTIALS, key=lambda m: c["partial"][m]) == "scene_pix"))
    for s in BSETS:
        for f in ("gemma", "gemma_task", "expert", "action"):
            c = next(c for c in B if c["set"] == s and c["feat"] == f and c["phase"] == "traj")
            b = c["beta"]
            if "sem" not in b:
                continue
            hb = "H-B1" if f.startswith("gemma") else "H-B2"
            H[f"{hb}|{s}|{f}"] = dict(beta_sem=round(b["sem"], 3), beta_motion=round(b["motion"], 3),
                                      supported=bool(b["sem"] > b["motion"]) if hb == "H-B1" else bool(b["motion"] > b["sem"]))
    order = [("basket_A100", "near"), ("basket_4090", "far20"), ("basket_4090", "far25"), ("basket_A100", "opp")]
    for f in FEAT_ORDER + ["MOTION"]:
        H[f"H-B3|{f}"] = dict(ratios=[round(graded[f"{m}|{c}|{f}|traj"]["ratio"], 3) for m, c in order],
                              rise_near_to_opp=round(graded[f"basket_A100|opp|{f}|traj"]["ratio"]
                                                     - graded[f"basket_A100|near|{f}|traj"]["ratio"], 3))
    x = xg_table(C)
    xsum = {}
    for kind in ("cand", "l90"):
        for f in FEAT_ORDER:
            cs = sorted(x[(kind, f)])
            early = [c for c in cs if c <= 2]
            late = [c for c in cs if c >= 10]
            xsum[f"{kind}|{f}"] = {k: dict(early=round(float(np.mean([x[(kind, f)][c][k] for c in early])), 3),
                                           late=round(float(np.mean([x[(kind, f)][c][k] for c in late])), 3),
                                           all=round(float(np.mean([x[(kind, f)][c][k] for c in cs])), 3))
                                   for k in ("executed", "instructed")}
            xsum[f"{kind}|{f}"]["n"] = x[(kind, f)][0]["n"]
    for kind in ("cand", "l90"):
        H[f"H-C1|{kind}"] = dict(instructed_share={f: xsum[f"{kind}|{f}"]["instructed"]["all"] for f in FEAT_ORDER},
                                 executed_share={f: xsum[f"{kind}|{f}"]["executed"]["all"] for f in FEAT_ORDER},
                                 supported=bool(xsum[f"{kind}|gemma_task"]["instructed"]["all"]
                                                > max(xsum[f"{kind}|expert"]["instructed"]["all"],
                                                      xsum[f"{kind}|action"]["instructed"]["all"])
                                                and min(xsum[f"{kind}|expert"]["executed"]["all"],
                                                        xsum[f"{kind}|action"]["executed"]["all"])
                                                > xsum[f"{kind}|gemma_task"]["executed"]["all"]))
        H[f"H-C2|{kind}"] = dict(gemma_task_instructed_early=xsum[f"{kind}|gemma_task"]["instructed"]["early"],
                                 gemma_task_instructed_late=xsum[f"{kind}|gemma_task"]["instructed"]["late"],
                                 expert_executed_early=xsum[f"{kind}|expert"]["executed"]["early"],
                                 action_executed_early=xsum[f"{kind}|action"]["executed"]["early"],
                                 supported=bool(xsum[f"{kind}|gemma_task"]["instructed"]["early"]
                                                > xsum[f"{kind}|gemma_task"]["instructed"]["late"]))
    dec = {}
    for kind in ("task40", "within", "suite", "verb", "object", "dest"):
        d = curves(C, kind, "norm")
        pp = curves(C, kind, "pp")
        for f in FEAT_ORDER:
            v = [d[f][t][0] for t in range(NORM)]
            dec[f"{kind}|{f}"] = dict(mean=round(float(np.mean(v)), 3), first_bin=round(v[0], 3),
                                      max=round(max(v), 3), argmax_time=round((int(np.argmax(v)) + 0.5) / NORM, 3),
                                      last_bin=round(v[-1], 3), pre=round(pp[f][0][0], 3), post=round(pp[f][1][0], 3),
                                      chance=round(d[f][0][1], 3))
    best = max(FEAT_ORDER, key=lambda f: dec[f"task40|{f}"]["mean"])
    H["H-C3"] = dict(task40_mean={f: dec[f"task40|{f}"]["mean"] for f in FEAT_ORDER},
                     task40_pre={f: dec[f"task40|{f}"]["pre"] for f in FEAT_ORDER}, best=best,
                     supported=best in ("attention", "patch_attention"))
    numbers = dict(
        prereg="scripts/shortcut/analysis/rsa_decoding_PREREG.md",
        n_tests={k: len(v) for k, v in fam.items()},
        n_significant_q05={k: int(sum(q[x] < 0.05 for x in v)) for k, v in fam.items()},
        n_significant_p05={k: int(sum(p < 0.05 for p in v.values())) for k, v in fam.items()},
        hypotheses=H,
        A=[dict(c, q={m: q.get(f"A|{c['unit']}|{c['feat']}|{c['phase']}|{c['group']}|rho|{m}") for m in c["rho"]},
                q_partial={m: q[f"A|{c['unit']}|{c['feat']}|{c['phase']}|{c['group']}|partial|{m}"] for m in c["partial"]})
           for c in A],
        model_intercorr=A_all["model_intercorr"],
        B=[dict(c, q={m: q[f"B|{c['set']}|{c['feat']}|{c['phase']}|{m}"] for m in c["p"]}) for c in B],
        basket_graded=graded,
        C=dict(summary=dec, phase4={r["feat"]: dict(bacc=round(r["bacc"], 3), confusion=r["confusion"]) for r in C["phase4"]},
               success={f"{r['feat']}|{r['axis']}|{r['t']}": round(r["bacc"], 3) for r in C["success"]},
               xgen_summary=xsum,
               xgen=C["xgen"],
               confusion={k: dict(rho=v["rho"], top_pairs=v["top_pairs"]) for k, v in C["confusion"].items()},
               curves={f"{r['kind']}|{r['axis']}|{r['feat']}|{r['t']}": round(r.get("bacc", r["acc"]), 3)
                       for r in C["decoding"]}))
    json.dump(numbers, open(out / "numbers.json", "w"), indent=1, default=float)
    print(json.dumps(dict(n_tests=numbers["n_tests"], sig=numbers["n_significant_q05"],
                          H={k: v.get("supported") for k, v in H.items()}), indent=0), flush=True)


# ==== MAIN ====
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="load,sem")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for p in args.parts.split(","):
        {"load": cmd_load, "emb": cmd_emb, "sem": cmd_sem, "A": cmd_A, "B": cmd_B, "C": cmd_C, "D": cmd_D}[p]()
