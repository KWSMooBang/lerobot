import json

O = "/outputs/VLA-shortcut-exps/analysis/neuro_dyn/"
num = json.load(open(O + "numbers.json"))
B = O + "b/"
L = {k: json.load(open(B + k + ".json")) for k in ("b1", "a2", "a3", "a4", "a5", "a6", "a7")}
num["stage"] = "A+B"
num["B1"] = {k: L["b1"][k] for k in ("n_pairs", "n_fallback", "n_new", "fallback_vs_new", "lscene_secondary", "n_tests", "n_sig_q05")}
num["B1"]["exploratory_between_within"] = json.load(open(B + "b1_exploratory_between_within.json"))
num["A2"] = {"lang": {k: v for k, v in L["a2"]["a_lang"].items() if k != "dist_by_offset"}, "action": L["a2"]["b_action"],
             "n_tests": L["a2"]["n_tests"], "n_sig_q05": L["a2"]["n_sig_q05"]}
num["A3"] = {g: d for g, d in L["a3"].items() if isinstance(d, dict)}
num["A3"].update({"n_tests": L["a3"]["n_tests"], "n_sig_q05": L["a3"]["n_sig_q05"]})
num["A4"] = {"all": L["a4"]["all"], "by_condition": L["a4"]["by_condition"], "n_tests": L["a4"]["n_tests"], "n_sig_q05": L["a4"]["n_sig_q05"]}
num["A5"] = {k: L["a5"][k] for k in ("random_angle_median", "cca_place_pairs_median", "cca_mixed_pairs_median", "basket", "n_tests", "n_sig_q05")}
num["A6"] = {k: ({kk: vv for kk, vv in v.items() if kk != "hist"} if isinstance(v, dict) and "hist" in v else v) for k, v in L["a6"].items() if k != "tests"}
num["A6"]["action_space_robustness"] = json.load(open(B + "a6_action_space_robustness.json"))
num["A7"] = {k: v for k, v in L["a7"].items() if k not in ("per_trial", "tests")}
num["tests_all_families"] = {k: L[k]["tests"] for k in L}
json.dump(num, open(O + "numbers.json", "w"), indent=1, default=str)
print("ok")
