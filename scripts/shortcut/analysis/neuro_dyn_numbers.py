import json

O = "/outputs/VLA-shortcut-exps/analysis/neuro_dyn/"
r = json.load(open(O + "a1/a1_results.json"))
sp = json.load(open(O + "a1/a1_exploratory_fallback_split.json"))
C = ["instr_para", "instr_goal", "instr_between", "scene_environment", "scene_position", "scene_object"]
num = {"stage": "A", "commit_prereg": "89b0981b", "A1": {"vlm_readin": {}, "expert_readout": {}}}
for key in ("L5_next", "L11_next", "L17_next"):
    e = r["vlm_readin"][key]
    num["A1"]["vlm_readin"][key] = {
        "read_layer": e["read_layer"], "structural_null": e["proxy_structural_null"], "chance_pf": e["chance_pf"],
        "segments": {
            seg: {"ACT_pf": s["act_pf_norm"], "ACT_gr": s["act_gr_norm"],
                  **{c: {"pf": s[c]["norm"]["pf_median"], "gr": s[c]["norm"]["gr_median"], "rel_norm": s[c]["norm"]["rel_norm_median"]} for c in C}}
            for seg, s in e["segments"].items()
        },
    }
for k in ("0", "9"):
    s = r["expert_readout"]["steps"][k]["L17"]
    num["A1"]["expert_readout"][f"step{k}_L17_pf_pooled"] = {"move": s["move_pf_pooled"], "ACT": s["act_pf"], **{c: s[c]["pf_pooled"] for c in C}}
num["A1"]["expert_readout"]["chance_pf"] = r["expert_readout"]["chance_pf"]
num["A1"]["decode_task_call0"] = r["decode"]
num["A1"]["validity_sig_h_call0"] = r["validity_sig_h_call0"]
num["A1"]["n_tests"] = r["n_tests"]
num["A1"]["n_sig_q05_bh"] = r["n_sig_q05"]
num["A1"]["primary_tests"] = [t for t in r["tests"] if (".task." in t["name"] and "_next" in t["name"] and "L17" not in t["name"]) or "exp.step0" in t["name"]]
num["A1"]["exploratory_fallback_split"] = sp
json.dump(num, open(O + "numbers.json", "w"), indent=1)
print("ok")
