"""PID demo analysis (CPU): semantic classification, PID numbers, figures, counts.json / pid.json.

uv run python scripts/pid_demo/analyze.py --out <dir>     # reads <dir>/raw from collect.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

COS_MIN = 0.7
D_MIN = 0.02  # m, commanded displacement over the executed segment; chosen from the |d| distribution
POS_SCALE = 0.05  # OSC controller output_max for position (m per unit action), read from the env
STOP = "stop/other"
N_SHUFFLE = 100
ALPHA = 0.5

# --- figure style (reference palette: first 4 categorical slots + neutral ink/grays)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
OTHER_OBJ = "#898781"
STOP_COL = "#c3c2b7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    }
)
DPI = 100

GOAL_SHORT = {
    "put the bowl on the plate": "bowl→plate",
    "put the wine bottle on the rack": "wine bottle→rack",
    "open the middle drawer of the cabinet": "open middle drawer",
    "turn on the stove": "turn on stove",
}


def pretty(name: str) -> str:
    return (
        name.replace("akita_black_", "")
        .replace("cabinet_", "")
        .replace("_drawer_handle", " drawer handle")
        .replace("_", " ")
    )


# ---------------------------------------------------------------- data
def load_grid(raw: Path, name: str, meta: dict) -> dict:
    z = dict(np.load(raw / f"grid_{name}.npz"))
    g = meta[name]
    g.update(z)
    g["name"] = name
    g["Y"] = sorted({c for s in g["scenes"] for c in s["cand_names"]}) + [STOP]
    return g


def displacement(g: dict, horizon: int) -> np.ndarray:
    """Cumulative commanded end-effector displacement (m) over the first `horizon` steps, base frame."""
    a = np.clip(g["actions"][..., :horizon, :3], -1.0, 1.0)  # env clips to the controller input range
    d_world = POS_SCALE * a.sum(-2)  # (V, L, K, 3)
    return np.einsum("vlki,vij->vlkj", d_world, g["base_rot"])  # world -> base: R^T d


def classify(g: dict, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels (V,L,K) into g['Y'], max-cos, |d|)."""
    d = displacement(g, horizon)
    V, L, K, _ = d.shape
    labels = np.full((V, L, K), -1)
    maxcos = np.full((V, L, K), np.nan)
    norm = np.linalg.norm(d, axis=-1)
    for v, s in enumerate(g["scenes"]):
        names = s["cand_names"]
        vec = g["cand_base"][v, : len(names)] - g["eef_base"][v]
        vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        for li in range(L):
            if not g["included"][v][li]:
                continue
            dd = d[v, li] / np.maximum(norm[v, li, :, None], 1e-9)
            cos = dd @ vec.T
            k = cos.argmax(1)
            mc = cos.max(1)
            maxcos[v, li] = mc
            for i in range(K):
                ok = mc[i] >= COS_MIN and norm[v, li, i] >= D_MIN
                labels[v, li, i] = g["Y"].index(names[k[i]]) if ok else g["Y"].index(STOP)
    return labels, maxcos, norm


def to_counts(labels: np.ndarray, included: np.ndarray, ny: int) -> np.ndarray:
    V, L, _ = labels.shape
    c = np.zeros((V, L, ny))
    for v in range(V):
        for li in range(L):
            if included[v][li]:
                c[v, li] = np.bincount(labels[v, li], minlength=ny)
    return c


def oracle_counts(g: dict, K: int) -> np.ndarray:
    V, L = np.shape(g["included"])
    c = np.zeros((V, L, len(g["Y"])))
    for v in range(V):
        for li, ins in enumerate(g["instructions"]):
            if g["included"][v][li]:
                c[v, li, g["Y"].index(ins["target"])] = K
    return c


# ---------------------------------------------------------------- information
def _kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float((p[m] * np.log2(p[m] / q[m])).sum())


def info_terms(counts: np.ndarray, included: np.ndarray, alpha: float = 0.0) -> dict:
    """Cell-uniform p(v,l) (excluded cells 0); p(y|v,l) = (n + alpha) / (K + alpha |Y|)."""
    inc = np.asarray(included, bool)
    pvl = inc / inc.sum()
    cond = counts + alpha
    cond = np.where(inc[..., None], cond / np.maximum(cond.sum(-1, keepdims=True), 1e-12), 0.0)
    joint = pvl[..., None] * cond  # p(v,l,y)
    pv, pl, py = pvl.sum(1), pvl.sum(0), joint.sum((0, 1))
    py_v = joint.sum(1) / np.maximum(pv[:, None], 1e-12)
    py_l = joint.sum(0) / np.maximum(pl[:, None], 1e-12)
    i_l_given_v = sum(pvl[v, li] * _kl_bits(cond[v, li], py_v[v]) for v, li in zip(*np.nonzero(inc), strict=False))
    i_v_given_l = sum(pvl[v, li] * _kl_bits(cond[v, li], py_l[li]) for v, li in zip(*np.nonzero(inc), strict=False))
    i_vl = sum(pvl[v, li] * _kl_bits(cond[v, li], py) for v, li in zip(*np.nonzero(inc), strict=False))
    i_v = sum(pv[v] * _kl_bits(py_v[v], py) for v in range(len(pv)) if pv[v] > 0)
    i_l = sum(pl[li] * _kl_bits(py_l[li], py) for li in range(len(pl)) if pl[li] > 0)
    return {
        "I(Y;L|V)=u_L+s": i_l_given_v,
        "I(Y;V|L)=u_V+s": i_v_given_l,
        "I(Y;V,L)": i_vl,
        "I(Y;V)": i_v,
        "I(Y;L)": i_l,
        "_joint": joint,
    }


def public(d: dict) -> dict:
    return {k: round(float(v), 4) for k, v in d.items() if not k.startswith("_")}


def shuffle_baseline(labels: np.ndarray, included: np.ndarray, ny: int, alpha: float, seed: int = 0) -> dict:
    """Pool all samples of the grid, permute them across cells (K per cell kept), recompute; mean of N_SHUFFLE."""
    rng = np.random.default_rng(seed)
    inc = np.asarray(included, bool)
    cells = list(zip(*np.nonzero(inc), strict=False))
    pool = np.concatenate([labels[v, li] for v, li in cells])
    K = labels.shape[-1]
    acc: dict[str, list[float]] = {}
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(pool)
        lab = labels.copy()
        for i, (v, li) in enumerate(cells):
            lab[v, li] = perm[i * K : (i + 1) * K]
        for k, val in public(info_terms(to_counts(lab, inc, ny), inc, alpha)).items():
            acc.setdefault(k, []).append(val)
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items()} | {
        f"{k} (sd)": round(float(np.std(v)), 4) for k, v in acc.items()
    }


def dit_pid(joint: np.ndarray) -> dict:
    try:
        import dit
        from dit.pid import PID_BROJA, PID_CCS
    except ImportError:
        return {"skipped": "dit not installed"}
    V, L, Y = joint.shape
    outs, ps = [], []
    for v in range(V):
        for li in range(L):
            for y in range(Y):
                if joint[v, li, y] > 1e-12:
                    outs.append((v, li, y))
                    ps.append(joint[v, li, y])
    ps = np.array(ps) / np.sum(ps)
    d = dit.Distribution(outs, ps.tolist())
    res = {}
    for name, P in (("BROJA", PID_BROJA), ("I_ccs", PID_CCS)):
        try:
            p = P(d, [[0], [1]], [2])  # inputs: V, L ; output: Y
            res[name] = {
                "r": round(float(p.get_pi(((0,), (1,)))), 4),
                "u_V": round(float(p.get_pi(((0,),))), 4),
                "u_L": round(float(p.get_pi(((1,),))), 4),
                "s": round(float(p.get_pi(((0, 1),))), 4),
            }
        except Exception as e:  # noqa: BLE001
            res[name] = {"error": repr(e)[:200]}
    return res


def grid_pid(g: dict, labels: np.ndarray) -> dict:
    inc = np.asarray(g["included"], bool)
    ny, K = len(g["Y"]), labels.shape[-1]
    counts = to_counts(labels, inc, ny)
    ocounts = oracle_counts(g, K)
    olabels = np.zeros_like(labels)
    for v, li in zip(*np.nonzero(inc), strict=False):
        olabels[v, li] = g["Y"].index(g["instructions"][li]["target"])
    out: dict = {}
    for tag, alpha in (("plugin", 0.0), ("smoothed_0.5", ALPHA)):
        m = info_terms(counts, inc, alpha)
        o = info_terms(ocounts, inc, alpha)
        ms = shuffle_baseline(labels, inc, ny, alpha)
        os_ = shuffle_baseline(olabels, inc, ny, alpha)
        key = "I(Y;L|V)=u_L+s"
        G = 1 - m[key] / o[key] if o[key] > 0 else None
        corr_m, corr_o = m[key] - ms[key], o[key] - os_[key]
        out[tag] = {
            "model": public(m),
            "oracle": public(o),
            "shuffle_model": ms,
            "shuffle_oracle": os_,
            "G": None if G is None else round(G, 4),
            "G_shuffle_corrected": round(1 - corr_m / corr_o, 4) if corr_o > 0 else None,
            "G_V(u_V+s)": round(1 - m["I(Y;V|L)=u_V+s"] / o["I(Y;V|L)=u_V+s"], 4)
            if o["I(Y;V|L)=u_V+s"] > 0
            else None,
            "dit_model": dit_pid(m["_joint"]),
            "dit_oracle": dit_pid(o["_joint"]),
        }
    out["log2_n_instructions"] = round(float(np.log2(inc.shape[1])), 4)
    out["log2_n_scenes"] = round(float(np.log2(inc.shape[0])), 4)
    return out


# ---------------------------------------------------------------- figures
def instr_short(g, ins) -> str:
    return (
        GOAL_SHORT.get(ins["text"], pretty(ins["target"])) if g["name"] == "goal" else pretty(ins["target"])
    )


def disp_cat(g: dict, yname: str) -> tuple[str, str]:
    targets = [ins["target"] for ins in g["instructions"]]
    if yname in targets:
        i = targets.index(yname)
        return pretty(yname), SLOTS[i]
    if yname == STOP:
        return STOP, STOP_COL
    return "other object", OTHER_OBJ


def project(cam: np.ndarray, pts: np.ndarray, size: int) -> np.ndarray:
    pts = np.atleast_2d(pts)
    h = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ cam.T
    return np.stack([size - 1 - h[:, 0] / h[:, 2], h[:, 1] / h[:, 2]], 1)


def disp(img):
    return img[::-1, ::-1]


def fig_rollout(raw: Path, out: Path):
    z = np.load(raw / "rollout.npz")
    meta = json.loads((raw / "rollout_meta.json").read_text())
    n = len(meta["moments"])
    fig, axes = plt.subplots(2, n, figsize=(min(16, 2.2 * n), 5.0), dpi=DPI)
    for i, mo in enumerate(meta["moments"]):
        axes[0, i].imshow(disp(z["agentview"][i]))
        axes[1, i].imshow(disp(z["wrist"][i]))
        axes[0, i].set_title(mo["label"].replace(" t=", "\nt="), fontsize=9, color=INK)
        for a in axes[:, i]:
            a.set_xticks([])
            a.set_yticks([])
    axes[0, 0].set_ylabel("3rd-person (agentview)")
    axes[1, 0].set_ylabel("wrist")
    fig.suptitle(
        f'π0.5 rollout, LIBERO-Goal "{meta["instruction"]}" (init_state 0): '
        f"{'success' if meta['success'] else 'failure'} in {meta['n_steps']} steps; "
        f"first grasp (gripper cmd → close) at t={meta['first_grasp_step']}",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out / "fig1_rollout_strip.png")
    plt.close(fig)


def fig_scenes(g: dict, out: Path, fname: str, title: str):
    V = len(g["scenes"])
    fig, axes = plt.subplots(1, V, figsize=(min(16, 3.6 * V), 4.2), dpi=DPI)
    size = g["agentview"].shape[1]
    for v, s in enumerate(g["scenes"]):
        ax = axes[v]
        ax.imshow(disp(g["agentview"][v]))
        names = s["cand_names"]
        xy = project(g["cam"][v], g["cand_world"][v, : len(names)], size)
        for nm, (x, y) in zip(names, xy, strict=False):
            _, col = disp_cat(g, nm)
            ax.plot(x, y, "o", ms=5, mfc=col if col != OTHER_OBJ else "white", mec=INK, mew=0.8)
            ax.text(
                x + 5,
                y - 4,
                pretty(nm),
                fontsize=7,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "lw": 0},
            )
        ax.set_title(s["label"], fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / fname)
    plt.close(fig)


def fig_samples(
    g: dict, labels: np.ndarray, cells: list[tuple[int, int]], horizon: int, out: Path, fname: str, title: str
):
    d_base = displacement(g, horizon)
    size = g["agentview"].shape[1]
    fig, axes = plt.subplots(1, len(cells), figsize=(min(16, 3.8 * len(cells)), 4.6), dpi=DPI)
    used = {}
    for ax, (v, li) in zip(axes, cells, strict=False):
        ax.imshow(disp(g["agentview"][v]))
        names = g["scenes"][v]["cand_names"]
        cam = g["cam"][v]
        p_w = g["eef_world"][v]
        p_xy = project(cam, p_w, size)[0]
        for k in range(labels.shape[-1]):
            d_w = g["base_rot"][v] @ d_base[v, li, k]
            e_xy = project(cam, p_w + d_w, size)[0]
            lab, col = disp_cat(g, g["Y"][labels[v, li, k]])
            used[lab] = col
            ax.annotate(
                "",
                xy=e_xy,
                xytext=p_xy,
                arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1.0, "alpha": 0.75, "mutation_scale": 8},
            )
        xy = project(cam, g["cand_world"][v, : len(names)], size)
        for nm, (x, y) in zip(names, xy, strict=False):
            ax.plot(x, y, "x", ms=6, color="white", mew=1.6)
            ax.text(
                x + 4,
                y + 10,
                pretty(nm),
                fontsize=6.5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "lw": 0},
            )
        ins = g["instructions"][li]
        n_t = int((labels[v, li] == g["Y"].index(ins["target"])).sum())
        diag = " (training pair)" if g["name"] == "object" and v == li else ""
        ax.set_title(
            f'{g["scenes"][v]["label"]}{diag}\ninstr: "{instr_short(g, ins)}" — {n_t}/32 → target',
            fontsize=8.5,
            color=INK,
        )
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(
        handles=[Patch(color=c, label=lab) for lab, c in used.items()],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=len(used),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(title, fontsize=10, color=INK)
    fig.text(
        0.5,
        0.005,
        "Arrow = sum of the first executed delta commands × 0.05 m (OSC scale), drawn from the gripper site; "
        "the realized sim motion is ≈1/6 of it, same direction (checked on Goal S0/L0 vs the rollout). × = candidate positions (BDDL objects, fixture handle/knob geoms).",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(out / fname)
    plt.close(fig)


def fig_grid(g: dict, counts: np.ndarray, out: Path, fname: str, title: str):
    V, L, _ = counts.shape
    cats: dict[str, str] = {}
    for y in g["Y"]:
        lab, col = disp_cat(g, y)
        cats.setdefault(lab, col)
    order = [pretty(ins["target"]) for ins in g["instructions"]] + ["other object", STOP]
    order = [o for o in order if o in cats]
    fig, axes = plt.subplots(V, L, figsize=(min(16, 2.9 * L + 1.5), 1.9 * V + 1.4), dpi=DPI, squeeze=False)
    for v in range(V):
        for li in range(L):
            ax = axes[v, li]
            if not g["included"][v][li]:
                ax.set_facecolor("#e1e0d9")
                ax.text(
                    0.5,
                    0.5,
                    "target absent\n(cell excluded)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=INK2,
                    fontsize=8,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                vals = dict.fromkeys(order, 0)
                for yi, y in enumerate(g["Y"]):
                    vals[disp_cat(g, y)[0]] += counts[v, li, yi]
                x = np.arange(len(order))
                ax.bar(
                    x,
                    [vals[o] for o in order],
                    color=[cats[o] for o in order],
                    width=0.8,
                    edgecolor="#fcfcfb",
                    linewidth=2,
                )
                for xi, o in zip(x, order, strict=False):
                    if vals[o] > 0:
                        ax.text(xi, vals[o] + 1, f"{int(vals[o])}", ha="center", fontsize=7, color=INK2)
                ax.set_ylim(0, 38)
                ax.set_xticks([])
                ax.set_yticks([0, 16, 32])
                ax.grid(axis="y", color=GRID, lw=0.6)
                ax.set_axisbelow(True)
                if g["name"] == "object" and v == li:
                    for sp in ax.spines.values():
                        sp.set_visible(True)
                        sp.set_color(INK)
                        sp.set_linewidth(1.2)
            if v == 0:
                ax.set_title(f'L{li}: "{instr_short(g, g["instructions"][li])}"', fontsize=9, color=INK)
            if li == 0:
                ax.set_ylabel(g["scenes"][v]["label"].replace(": ", ":\n"), fontsize=8, color=INK)
    fig.legend(
        handles=[Patch(color=cats[o], label=o) for o in order],
        loc="lower center",
        ncol=len(order),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / fname)
    plt.close(fig)


def fig_pid(pid: dict, out: Path):
    keys = [("I(Y;L|V)=u_L+s", "u_L + s = I(Y;L|V)"), ("I(Y;V|L)=u_V+s", "u_V + s = I(Y;V|L)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=DPI, sharey=True)
    for ax, gname, gtitle in zip(
        axes, ["object", "goal"], ["LIBERO-Object t=0 grid (4×4, 13 cells)", "LIBERO-Goal t=0 grid (3×4)"], strict=False
    ):
        r = pid[gname]["plugin"]
        x = np.arange(len(keys))
        w = 0.36
        for j, (who, col) in enumerate((("model", SLOTS[0]), ("oracle", SLOTS[1]))):
            vals = [r[who][k] for k, _ in keys]
            sh = [r[f"shuffle_{who}"][k] for k, _ in keys]
            xs = x + (j - 0.5) * w
            ax.bar(
                xs,
                vals,
                width=w,
                color=col,
                edgecolor="#fcfcfb",
                linewidth=2,
                label="π0.5" if who == "model" else "semantic oracle",
            )
            for xi, val, s in zip(xs, vals, sh, strict=False):
                ax.hlines(
                    s, xi - w / 2 + 0.03, xi + w / 2 - 0.03, colors=INK, linestyles=(0, (2, 1.5)), lw=1.4
                )
                ax.text(xi, val + 0.04, f"{val:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, [lab for _, lab in keys])
        ax.set_title(f"{gtitle}\nG = 1 − model/oracle (u_L+s) = {r['G']:.2f}", fontsize=9.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("bits (plug-in, no smoothing)")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK, ls=(0, (2, 1.5)), lw=1.4))
    lab.append("shuffle baseline (100 permutations)")
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig8_pid_bars.png")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    raw = out / "raw"
    meta = json.loads((raw / "grids_meta.json").read_text())
    run = json.loads((raw / "run_meta.json").read_text())
    n_exec = min(int(run["checkpoint_config"]["n_action_steps_used"]), 10)

    fig_rollout(raw, out)
    counts_json, pid_json = (
        {},
        {
            "settings": {
                "n_exec_steps": n_exec,
                "cos_min": COS_MIN,
                "d_min_m": D_MIN,
                "pos_scale_m_per_unit": POS_SCALE,
                "smoothing_alpha": ALPHA,
                "n_shuffle": N_SHUFFLE,
                "K": int(run["K"]),
            }
        },
    )
    dist = {}
    for name in ("object", "goal"):
        g = load_grid(raw, name, meta)
        labels, maxcos, norm = classify(g, n_exec)
        inc = np.asarray(g["included"], bool)
        counts = to_counts(labels, inc, len(g["Y"]))
        sel = inc[..., None].repeat(labels.shape[-1], -1)
        dist[name] = {
            "|d|_quantiles_m(0,5,25,50,75,95,100%)": np.quantile(
                norm[sel], [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
            )
            .round(4)
            .tolist(),
            "max_cos_quantiles(0,5,25,50,75,95,100%)": np.quantile(
                maxcos[sel], [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
            )
            .round(3)
            .tolist(),
            "frac_stop_by_cos": round(float((maxcos[sel] < COS_MIN).mean()), 4),
            "frac_stop_by_magnitude": round(float((norm[sel] < D_MIN).mean()), 4),
        }
        pid_json[name] = grid_pid(g, labels)
        # sensitivity: classify on the whole 50-step chunk instead of the executed 10 steps
        labels50, _, _ = classify(g, 50)
        sens = grid_pid(g, labels50)
        pid_json[name]["sensitivity_horizon50_plugin"] = {
            k: sens["plugin"][k] for k in ("model", "oracle", "G", "shuffle_model")
        }
        counts_json[name] = {
            "categories": g["Y"],
            "scenes": [s["label"] for s in g["scenes"]],
            "scene_candidates": [s["cand_names"] for s in g["scenes"]],
            "instructions": [
                {"text": i["text"], "target": i["target"], "task_id": i["task_id"]} for i in g["instructions"]
            ],
            "cells": {
                f"S{v},L{li}": (
                    {g["Y"][y]: int(counts[v, li, y]) for y in range(len(g["Y"])) if counts[v, li, y] > 0}
                    if inc[v, li]
                    else "excluded (target absent)"
                )
                for v in range(inc.shape[0])
                for li in range(inc.shape[1])
            },
            "cells_horizon50": {
                f"S{v},L{li}": {
                    g["Y"][y]: int(c)
                    for y, c in enumerate(np.bincount(labels50[v, li], minlength=len(g["Y"])))
                    if c > 0
                }
                for v in range(inc.shape[0])
                for li in range(inc.shape[1])
                if inc[v, li]
            },
        }
        if name == "object":
            fig_scenes(
                g,
                out,
                "fig2_object_scenes.png",
                "LIBERO-Object t=0 scenes: scene j = task j layout, init_state 0, after 10 no-op steps",
            )
            cells = [(0, 0), (3, 3), (3, 0), (0, 1)]
            fig_samples(
                g,
                labels,
                cells,
                n_exec,
                out,
                "fig4_object_samples.png",
                f"π0.5, 32 sampled chunks per cell: commanded EEF displacement over the first {n_exec} steps "
                "(arrows), coloured by semantic category",
            )
            fig_grid(
                g,
                counts,
                out,
                "fig6_object_grid.png",
                "LIBERO-Object t=0 grid: category counts out of K=32 per cell (bold frame = training pair)",
            )
        else:
            fig_scenes(
                g,
                out,
                "fig3_goal_scenes.png",
                "LIBERO-Goal t=0 scenes: original layout + two swapped layouts (free-joint qpos swap, 10 no-op settle)",
            )
            cells = [(0, 0), (1, 0), (2, 0), (0, 2)]
            fig_samples(
                g,
                labels,
                cells,
                n_exec,
                out,
                "fig5_goal_samples.png",
                f"π0.5, 32 sampled chunks per cell: commanded EEF displacement over the first {n_exec} steps "
                "(arrows), coloured by semantic category",
            )
            fig_grid(
                g,
                counts,
                out,
                "fig7_goal_grid.png",
                "LIBERO-Goal t=0 grid: category counts out of K=32 per cell",
            )
    pid_json["classifier_distributions"] = dist
    fig_pid(pid_json, out)
    (out / "counts.json").write_text(json.dumps(counts_json, indent=1, ensure_ascii=False))
    (out / "pid.json").write_text(json.dumps(pid_json, indent=1, ensure_ascii=False))
    for name in ("object", "goal"):
        r = pid_json[name]["plugin"]
        print(
            name,
            "model",
            r["model"],
            "\n  oracle",
            r["oracle"],
            "\n  shuffle",
            {k: v for k, v in r["shuffle_model"].items() if "sd" not in k},
            "\n  G",
            r["G"],
            "G_corr",
            r["G_shuffle_corrected"],
            "\n  dit",
            r["dit_model"],
            "\n  smoothed",
            pid_json[name]["smoothed_0.5"]["model"],
            "G",
            pid_json[name]["smoothed_0.5"]["G"],
            "\n  H50",
            pid_json[name]["sensitivity_horizon50_plugin"]["model"],
            "G",
            pid_json[name]["sensitivity_horizon50_plugin"]["G"],
        )
    print(json.dumps(dist, indent=1))


if __name__ == "__main__":
    main()
