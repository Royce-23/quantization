#!/usr/bin/env python3
"""Sinh các bảng/macro LaTeX của báo cáo từ file kết quả thật trong repository.

Chạy từ thư mục gốc repo:  python report/scripts/gen_tables.py
Đầu ra: report/generated/*.tex (được commit để biên dịch báo cáo không cần Python).
Cần models/params_fp32.npz (tạo bằng `python main.py export-c`).
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
N_VAL, N_TEST = 1485, 2947


def vn(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def vn_signed(x, nd):
    s = vn(abs(x), nd)
    if float(s.replace(",", ".")) == 0.0:
        return s
    return ("$-$" if x < 0 else "$+$") + s


def thousands(n):
    return f"{int(n):,}".replace(",", ".")


def tt(s):
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- đọc dữ liệu
summary = list(csv.DictReader(open(ROOT / "results/summary.csv", encoding="utf-8")))
metrics = list(csv.DictReader(open(ROOT / "results/metrics.csv", encoding="utf-8")))
manifest = json.load(open(ROOT / "models/params_manifest.json", encoding="utf-8"))
meta = json.load(open(ROOT / "models/baseline_metadata.json", encoding="utf-8"))
qmeta = json.load(open(ROOT / "models/model_metadata.json", encoding="utf-8"))
split = json.load(open(ROOT / "data/processed/split.json", encoding="utf-8"))
hist = list(csv.DictReader(open(ROOT / "results/tv1_training_history.csv", encoding="utf-8")))
npz = np.load(ROOT / "models/params_fp32.npz")
params_md = (ROOT / "results/parameters_reference.md").read_text(encoding="utf-8")


def srow(split_name, kind, n=""):
    for r in summary:
        if r["split"] == split_name and r["model_kind"] == kind and r["calibration_size"] == str(n):
            return r
    raise KeyError((split_name, kind, n))


def cm(path):
    return np.loadtxt(ROOT / "results/confusion_matrices" / path, delimiter=",", dtype=int)


# ---------------------------------------------------------------- bảng chia dữ liệu
rows = [
    ("Train (huấn luyện)", len(split["train_subjects"]), split["sample_counts"]["train"]),
    ("Validation", len(split["validation_subjects"]), split["sample_counts"]["validation"]),
    ("Test chính thức", len(split["test_subjects"]), split["sample_counts"]["test"]),
]
t = ["\\begin{tabular}{lrr}", "\\toprule", "Tập & Số subject & Số cửa sổ \\\\", "\\midrule"]
for name, ns, nw in rows:
    t.append(f"{name} & {ns} & {thousands(nw)} \\\\")
tot_sub = sum(r[1] for r in rows)
tot_win = sum(r[2] for r in rows)
t += ["\\midrule", f"Tổng & {tot_sub} & {thousands(tot_win)} \\\\", "\\bottomrule", "\\end{tabular}", ""]
write("tab_split.tex", "\n".join(t))

# ---------------------------------------------------------------- bảng validation
t = ["\\begin{tabular}{lrrrr}", "\\toprule",
     "Mô hình & Số lần chạy & Accuracy (\\%) & Macro-F1 & Dung lượng (byte) \\\\", "\\midrule"]
fp = srow("validation", "fp32")
t.append(f"FP32 (TFLite) & 1 & {vn(float(fp['accuracy_mean_pct']), 4)} & {vn(float(fp['macro_f1_mean']), 4)} & "
         f"{thousands(float(fp['model_size_mean_bytes']))} \\\\")
for n in (50, 200, 500):
    r = srow("validation", "int8", n)
    best = r["selected"] == "yes"
    acc = f"{vn(float(r['accuracy_mean_pct']), 4)} $\\pm$ {vn(float(r['accuracy_std_pct']), 4)}"
    f1 = f"{vn(float(r['macro_f1_mean']), 4)} $\\pm$ {vn(float(r['macro_f1_std']), 4)}"
    if best:
        acc, f1 = f"\\textbf{{{acc}}}", f"\\textbf{{{f1}}}"
    t.append(f"INT8, $n={n}$ & 3 & {acc} & {f1} & {thousands(float(r['model_size_mean_bytes']))} \\\\")
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_val.tex", "\n".join(t))

# ---------------------------------------------------------------- bảng test
fpt, i8t = srow("test", "fp32"), srow("test", "int8", 500)
t = ["\\begin{tabular}{lrrrr}", "\\toprule",
     "Mô hình & Accuracy (\\%) & $\\Delta$ so với FP32 (điểm \\%) & Macro-F1 & Dung lượng (byte) \\\\", "\\midrule",
     f"FP32 (TFLite) & {vn(float(fpt['accuracy_mean_pct']), 4)} & 0,0000 & {vn(float(fpt['macro_f1_mean']), 4)} & "
     f"{thousands(float(fpt['model_size_mean_bytes']))} \\\\",
     f"INT8, $n=500$ (3 seed) & {vn(float(i8t['accuracy_mean_pct']), 4)} $\\pm$ {vn(float(i8t['accuracy_std_pct']), 4)} & "
     f"{vn_signed(float(i8t['delta_accuracy_pp_vs_fp32']), 4)} & "
     f"{vn(float(i8t['macro_f1_mean']), 4)} $\\pm$ {vn(float(i8t['macro_f1_std']), 4)} & "
     f"{thousands(float(i8t['model_size_mean_bytes']))} \\\\",
     "\\bottomrule", "\\end{tabular}", ""]
write("tab_test.tex", "\n".join(t))

# ---------------------------------------------------------------- bảng từng seed
t = ["\\begin{tabular}{llrrrrr}", "\\toprule",
     "Tập & Mô hình & Seed & Accuracy (\\%) & Macro-F1 & Số mẫu đúng & Số mẫu sai \\\\", "\\midrule"]
errs = {}
for split_name, N in (("validation", N_VAL), ("test", N_TEST)):
    for r in [m for m in metrics if m["split"] == split_name]:
        acc = float(r["accuracy_pct"])
        correct = round(acc / 100 * N)
        assert abs(correct / N * 100 - acc) < 1e-3, (r["experiment_id"], correct, acc)
        errs[r["experiment_id"]] = (correct, N - correct)
        if r["calibration_size"]:
            name, seed = f"INT8 $n={r['calibration_size']}$", r["calibration_seed"]
        else:
            name, seed = "FP32", "--"
        t.append(f"{split_name.capitalize()} & {name} & {seed} & {vn(acc, 4)} & {vn(float(r['macro_f1']), 4)} & "
                 f"{thousands(correct)} & {thousands(N - correct)} \\\\")
    if split_name == "validation":
        t.append("\\midrule")
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_seed.tex", "\n".join(t))

# ---------------------------------------------------------------- confusion matrices: per-class + lỗi chính
cm_fp = cm("test_baseline_fp32.csv")
cm_i8 = [cm(f"test_int8_n500_s{s}.csv") for s in (11, 22, 33)]
for c_, (mm, exp) in zip(["fp32"] + [11, 22, 33],
                         [(cm_fp, "test-baseline_fp32")] + [(c, f"test-int8_n500_s{s}") for c, s in zip(cm_i8, (11, 22, 33))]):
    assert int(np.trace(mm)) == errs[exp][0], (exp, int(np.trace(mm)), errs[exp])


def recall(m):
    return np.diag(m) / m.sum(axis=1)


def precision(m):
    return np.diag(m) / m.sum(axis=0)


rec_fp, pre_fp = recall(cm_fp), precision(cm_fp)
rec_i8 = np.mean([recall(m) for m in cm_i8], axis=0)
pre_i8 = np.mean([precision(m) for m in cm_i8], axis=0)
t = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
     "Lớp & Số mẫu & Recall FP32 & Recall INT8 & Precision FP32 & Precision INT8 & $\\Delta$ recall (điểm \\%) \\\\",
     "\\midrule"]
for i, lab in enumerate(LABELS):
    t.append(f"{tt(lab)} & {thousands(cm_fp[i].sum())} & {vn(100 * rec_fp[i], 2)} & {vn(100 * rec_i8[i], 2)} & "
             f"{vn(100 * pre_fp[i], 2)} & {vn(100 * pre_i8[i], 2)} & {vn_signed(100 * (rec_i8[i] - rec_fp[i]), 2)} \\\\")
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_perclass.tex", "\n".join(t))

t = ["\\begin{tabular}{lrrr}", "\\toprule",
     "Mô hình & \\texttt{SITTING}$\\to$\\texttt{STANDING} & \\texttt{STANDING}$\\to$\\texttt{SITTING} & Tổng số mẫu sai \\\\",
     "\\midrule"]
t.append(f"FP32 & {cm_fp[3, 4]} & {cm_fp[4, 3]} & {int(cm_fp.sum() - np.trace(cm_fp))} \\\\")
for s, m in zip((11, 22, 33), cm_i8):
    t.append(f"INT8 $n=500$, seed {s} & {m[3, 4]} & {m[4, 3]} & {int(m.sum() - np.trace(m))} \\\\")
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_confusion_pair.tex", "\n".join(t))
static_static_fp = int(cm_fp[3, 4] + cm_fp[4, 3])
static_total_fp = int(cm_fp.sum() - np.trace(cm_fp))

# ---------------------------------------------------------------- tham số lượng tử hóa vào/ra của 9 model INT8
t = ["\\begin{tabular}{lrrrrr}", "\\toprule",
     "Mô hình & Scale vào & Zero-point vào & Scale ra & Zero-point ra & Số tensor float \\\\", "\\midrule"]
for e in qmeta["models"]:
    if e["model_kind"] != "int8":
        continue
    nm = f"INT8 $n={e['calibration_size']}$, seed {e['calibration_seed']}"
    t.append(f"{nm} & {e['input']['scales'][0]:.6f} & {e['input']['zero_points'][0]} & "
             f"{e['output']['scales'][0]:.6f} & {e['output']['zero_points'][0]} & {e['float_tensor_count']} \\\\")
    assert e["full_integer"] and e["input"]["dtype"] == "int8" and e["output"]["dtype"] == "int8"
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_quant_io.tex", "\n".join(t))

# ---------------------------------------------------------------- tham số FP32: tổng quan
ROLE = {
    "block1_conv1": ("Khối 1", "A (chính)", "Conv1D, $K{=}3$, same"),
    "block1_conv2": ("Khối 1", "A (chính)", "Conv1D, $K{=}3$, same"),
    "block1_projection": ("Khối 1", "B (shortcut)", "Conv1D, $K{=}1$, valid"),
    "block2_conv1": ("Khối 2", "A (chính)", "Conv1D, $K{=}3$, same"),
    "block2_conv2": ("Khối 2", "A (chính)", "Conv1D, $K{=}3$, same"),
    "activity_logits": ("Đầu ra", "--", "Dense $32\\to6$"),
}
total = manifest["parameter_count"]
t = ["\\begin{tabular}{rllllrrr}", "\\toprule",
     "\\# & Layer & Khối & Nhánh & Phép toán & Shape kernel & Số tham số & \\% tổng \\\\", "\\midrule"]
acc_params = 0
for i, L in enumerate(manifest["layers"], 1):
    blk, br, op = ROLE[L["name"]]
    ks = "(" + ", ".join(map(str, L["kernel_shape"])) + ")"
    k_el = int(np.prod(npz[L["name"] + "_w"].shape))
    b_el = int(np.prod(npz[L["name"] + "_b"].shape))
    assert k_el + b_el == L["params"], L
    acc_params += L["params"]
    t.append(f"{i} & {tt(L['name'])} & {blk} & {br} & {op} & {ks} & {thousands(L['params'])} & "
             f"{vn(100 * L['params'] / total, 2)} \\\\")
assert acc_params == total == 10726
t += ["\\midrule", f"& & & & & \\textbf{{Tổng}} & \\textbf{{{thousands(total)}}} & \\textbf{{100,00}} \\\\",
      "\\bottomrule", "\\end{tabular}", ""]
write("tab_params_overview.tex", "\n".join(t))

# ---------------------------------------------------------------- tham số FP32: thống kê (đối chiếu với results/parameters_reference.md)
t = ["\\begin{tabular}{llrrrrr}", "\\toprule",
     "Layer & Tensor & Số phần tử & min & max & trung bình & độ lệch chuẩn \\\\", "\\midrule"]
warn = []
for L in manifest["layers"]:
    for suffix, label in (("_w", "kernel"), ("_b", "bias")):
        a = npz[L["name"] + suffix].astype(np.float64).ravel()
        mn, mx, me, sd = a.min(), a.max(), a.mean(), a.std()
        ref = f"min={mn:.6f}, max={mx:.6f}, mean={me:.6f}, std={sd:.6f}"
        if ref not in params_md:
            warn.append((L["name"], label, ref))
        first = tt(L["name"]) if suffix == "_w" else ""
        t.append(f"{first} & {label} & {thousands(a.size)} & {vn_signed(mn, 4)} & {vn_signed(mx, 4)} & "
                 f"{vn_signed(me, 4)} & {vn(sd, 4)} \\\\")
    if L is not manifest["layers"][-1]:
        t.append("\\addlinespace[2pt]")
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_params_stats.tex", "\n".join(t))
if warn:
    print("CẢNH BÁO: thống kê lệch với results/parameters_reference.md:", warn)
else:
    print("OK: 12/12 thống kê tensor khớp results/parameters_reference.md")

# ---------------------------------------------------------------- z-score
mean, std = npz["input_mean"].astype(np.float64), npz["input_std"].astype(np.float64)
chan = meta["channel_order"]
t = ["\\begin{tabular}{rlrr}", "\\toprule", "\\# & Kênh & $\\mu_c$ & $\\sigma_c$ \\\\", "\\midrule"]
for c, name in enumerate(chan):
    t.append(f"{c} & {tt(name)} & {vn_signed(mean[c], 6)} & {vn(std[c], 6)} \\\\")
    assert f"{mean[c]:.6f}" in params_md and f"{std[c]:.6f}" in params_md
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_zscore.tex", "\n".join(t))

# ---------------------------------------------------------------- Dense: toàn bộ kernel (32x6) + bias
W, b = npz["activity_logits_w"].astype(np.float64), npz["activity_logits_b"].astype(np.float64)
short = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY"]
t = ["\\begin{tabular}{r" + "r" * 6 + "}", "\\toprule",
     "$i$ & " + " & ".join(f"$W_{{i,{o}}}$ ({short[o]})" for o in range(6)) + " \\\\", "\\midrule"]
for i in range(32):
    t.append(f"{i} & " + " & ".join(vn_signed(W[i, o], 4) for o in range(6)) + " \\\\")
t += ["\\midrule", "$b_o$ & " + " & ".join(vn_signed(b[o], 4) for o in range(6)) + " \\\\", "\\bottomrule", "\\end{tabular}", ""]
write("tab_dense.tex", "\n".join(t))

# ---------------------------------------------------------------- projection kernel (1,9,32) + bias (phụ lục)
Wp, bp = npz["block1_projection_w"].astype(np.float64), npz["block1_projection_b"].astype(np.float64)
assert Wp.shape == (1, 9, 32)
t = ["\\begin{tabular}{r" + "r" * 10 + "}", "\\toprule",
     "$c_o$ & " + " & ".join(f"$c_i{{=}}{ci}$" for ci in range(9)) + " & $b$ \\\\", "\\midrule"]
for co in range(32):
    t.append(f"{co} & " + " & ".join(vn_signed(Wp[0, ci, co], 3) for ci in range(9)) + f" & {vn_signed(bp[co], 3)} \\\\")
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_projection.tex", "\n".join(t))

# ---------------------------------------------------------------- bias của 2 lớp Conv đầu (một phần, ví dụ trích)
c1 = npz["block1_conv1_w"].astype(np.float64)  # (3, 9, 32)
t = ["\\begin{tabular}{r" + "r" * 9 + "}", "\\toprule",
     "$k$ & " + " & ".join(f"$c_i{{=}}{ci}$" for ci in range(9)) + " \\\\", "\\midrule"]
for k in range(3):
    t.append(f"{k} & " + " & ".join(vn_signed(c1[k, ci, 0], 3) for ci in range(9)) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_conv1_filter0.tex", "\n".join(t))

# ---------------------------------------------------------------- file C sinh tự động
t = ["\\begin{tabular}{lrl}", "\\toprule", "Tệp sinh tự động & Byte & SHA-256 \\\\", "\\midrule"]
for g in manifest["generated_files"]:
    t.append(f"{tt(g['path'].replace('firmware/', ''))} & {thousands(g['bytes'])} & "
             f"\\texttt{{\\scriptsize {g['sha256']}}} \\\\")
t += ["\\bottomrule", "\\end{tabular}", ""]
write("tab_cfiles.tex", "\n".join(t))

# ---------------------------------------------------------------- macro số liệu dùng trong văn bản
best = int(meta["metrics"]["best_epoch"])
h = {int(r["epoch"]): r for r in hist}
n_correct_fp = errs["test-baseline_fp32"][0]
n_err_fp = errs["test-baseline_fp32"][1]
i8_errs = [errs[f"test-int8_n500_s{s}"][1] for s in (11, 22, 33)]
p_fp = n_correct_fp / N_TEST
se_pp = 100 * math.sqrt(p_fp * (1 - p_fp) / N_TEST)
param_bytes_fp = total * 4
macs_exact = 0
for L, ln, cin, cout, K in [(128, "b1c1", 9, 32, 3), (128, "b1c2", 32, 32, 3), (128, "b1p", 9, 32, 1),
                            (64, "b2c1", 32, 32, 3), (64, "b2c2", 32, 32, 3)]:
    p = K // 2
    macs_exact += (K * L - p * (p + 1)) * cin * cout
macs_exact += 32 * 6
macs_upper = sum(K * L * cin * cout for L, cin, cout, K in
                 [(128, 9, 32, 3), (128, 32, 32, 3), (128, 9, 32, 1), (64, 32, 32, 3), (64, 32, 32, 3)]) + 32 * 6
assert macs_exact == 927360 and macs_upper == 934080, (macs_exact, macs_upper)

M = {
    "ParamCount": thousands(total),
    "ParamBytesFP": thousands(param_bytes_fp),
    "ParamBytesInt": thousands(total),
    "KerasBytes": thousands(meta["metrics"]["keras_model_size_bytes"]),
    "TfliteFP": thousands(float(fpt["model_size_mean_bytes"])),
    "TfliteInt": thousands(float(i8t["model_size_mean_bytes"])),
    "SizeRed": vn(float(i8t["size_reduction_pct"]), 2),
    "SizeRatio": vn(float(i8t["compression_ratio"]), 2),
    "ValAccFP": vn(float(fp["accuracy_mean_pct"]), 2),
    "ValAccFPlong": vn(float(fp["accuracy_mean_pct"]), 4),
    "ValFoneFP": vn(float(fp["macro_f1_mean"]), 4),
    "TestAccFP": vn(float(fpt["accuracy_mean_pct"]), 2),
    "TestAccFPlong": vn(float(fpt["accuracy_mean_pct"]), 4),
    "TestFoneFP": vn(float(fpt["macro_f1_mean"]), 4),
    "TestAccInt": vn(float(i8t["accuracy_mean_pct"]), 2),
    "TestAccIntlong": vn(float(i8t["accuracy_mean_pct"]), 4),
    "TestAccIntSD": vn(float(i8t["accuracy_std_pct"]), 4),
    "TestFoneInt": vn(float(i8t["macro_f1_mean"]), 4),
    "TestFoneIntSD": vn(float(i8t["macro_f1_std"]), 4),
    "TestDelta": vn(abs(float(i8t["delta_accuracy_pp_vs_fp32"])), 4),
    "TestDeltaShort": vn(abs(float(i8t["delta_accuracy_pp_vs_fp32"])), 2),
    "TestN": thousands(N_TEST),
    "TestCorrectFP": thousands(n_correct_fp),
    "TestErrFP": thousands(n_err_fp),
    "TestErrIntMin": thousands(min(i8_errs)),
    "TestErrIntMax": thousands(max(i8_errs)),
    "TestErrDiffMin": thousands(min(i8_errs) - n_err_fp),
    "TestErrDiffMax": thousands(max(i8_errs) - n_err_fp),
    "TestSE": vn(se_pp, 2),
    "BestEpoch": str(best),
    "TrainAccBest": vn(100 * float(h[best]["accuracy"]), 2),
    "ValAccBest": vn(100 * float(h[best]["val_accuracy"]), 2),
    "TrainAccLast": vn(100 * float(h[300]["accuracy"]), 2),
    "ValAccLast": vn(100 * float(h[300]["val_accuracy"]), 2),
    "MACexact": thousands(macs_exact),
    "MACupper": thousands(macs_upper),
    "StaticSubjectErrFP": str(static_static_fp),
    "StaticShareFP": vn(100 * static_static_fp / static_total_fp, 1),
    "ValFPfromBest": vn(100 * float(h[best]["val_accuracy"]), 4),
}
lines = ["% Tự động sinh bởi report/scripts/gen_tables.py — KHÔNG SỬA TAY."]
for k, v in M.items():
    lines.append(f"\\newcommand{{\\{k}}}{{\\mbox{{{v}}}}}")
write("numbers.tex", "\n".join(lines) + "\n")
print("Đã ghi", len(list(OUT.glob("*.tex"))), "file vào", OUT)

# ---------------------------------------------------------------- trích đoạn code C (chỉ ASCII để listings in được)
import re

src_c = (ROOT / "firmware/src/har_resnet.c").read_text(encoding="utf-8").splitlines()


def extract_func(sig_prefix):
    start = next(i for i, l in enumerate(src_c) if l.startswith(sig_prefix))
    end = next(i for i in range(start, len(src_c)) if src_c[i] == "}")
    return "\n".join(src_c[start:end + 1]) + "\n"


def drop_nonascii_comments(text):
    def repl(m):
        return "" if not m.group(0).isascii() else m.group(0)
    text = re.sub(r"^[ \t]*/\*.*?\*/[ \t]*\n", lambda m: repl(m), text, flags=re.S | re.M)
    text = re.sub(r"[ \t]*/\*.*?\*/", lambda m: repl(m), text, flags=re.S)
    text = re.sub(r"[ \t]+\n", "\n", text)
    assert text.isascii(), "còn ký tự không phải ASCII trong trích đoạn C"
    return text


write("c_conv1d.c", drop_nonascii_comments(extract_func("void har_conv1d_same")))
head = "".join(drop_nonascii_comments(extract_func(s)) + "\n" for s in
               ("void har_block1_forward", "void har_block2_forward", "void har_head_forward", "void har_resnet_predict"))
write("c_blocks.c", head.rstrip() + "\n")

params_c = (ROOT / "firmware/src/har_resnet_params.c").read_text(encoding="utf-8").splitlines()
i0 = next(i for i, l in enumerate(params_c) if l.startswith("/* shape (3, 9, 32)"))
exc = "\n".join(params_c[i0:i0 + 4]) + "\n"
assert exc.isascii()
write("c_params_excerpt.c", exc)
first_vals = [float(x.rstrip("f")) for x in params_c[i0 + 2].strip().rstrip(",").split(",")]
assert np.allclose(first_vals, npz["block1_conv1_w"].ravel()[:8], rtol=0, atol=1e-8), "trích đoạn C khác npz"
print("OK: trích đoạn C khớp params_fp32.npz; ghi c_*.c")
