"""Regenerate report figures/tables from archived experiment artifacts only."""
import json
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
GREEN, BLUE, ORANGE = "#147864", "#416b9c", "#bd623d"

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)

with zipfile.ZipFile(ROOT / "results/task1_submission_artifacts.zip") as z:
    hist = json.loads(z.read("outputs/qwen2.5-0.5b-lora/train_metrics.json"))
train = [r for r in hist if "loss" in r]
val = [r for r in hist if "eval_loss" in r]
fig, ax = plt.subplots(figsize=(3.45, 2.35))
ax.plot([r["step"] for r in train], [r["loss"] for r in train], color=GREEN, lw=1.4, label="Training (logged batches)")
ax.plot([r["step"] for r in val], [r["eval_loss"] for r in val], "o-", color=ORANGE, lw=1.4, label="Validation (epoch end)")
ax.set(xlabel="Optimizer update", ylabel="Target-token cross-entropy")
ax.grid(alpha=.18); ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout(); save(fig, "training_curve")

d = json.loads((ROOT / "results/task2_error_analysis.json").read_text())
fine = d["models"]["finetuned"]
abbrev = {"included":"INC", "not included":"NINC", "not excluded":"NEX",
          "excluded":"EX", "not applicable":"NA", "not enough information":"NEI"}
fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), gridspec_kw={"width_ratios":[5,4,3]})
for ax, group, title in zip(axes, [fine["overall"], fine["by_criterion_type"]["inclusion"], fine["by_criterion_type"]["exclusion"]], ["(a) Overall, n=206", "(b) Inclusion, n=75", "(c) Exclusion, n=131"]):
    m = np.asarray(group["classification"]["confusion_matrix"])
    labels = [abbrev[x] for x in group["classification"]["labels"]]
    ax.imshow(m, cmap="Greens", vmin=0, vmax=91)
    for (i,j), v in np.ndenumerate(m):
        ax.text(j,i,str(v),ha="center",va="center",fontsize=7,color="white" if v>50 else "#173d32")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right",fontsize=7)
    ax.set_yticks(range(len(labels)), labels,fontsize=7)
    ax.set(xlabel="Predicted category", title=title)
axes[0].set_ylabel("Expert category")
fig.tight_layout(w_pad=1.2); save(fig,"confusion_matrices")

cache = json.loads((ROOT/"results/cache_benchmark_final.json").read_text())
spec = json.loads((ROOT/"results/speculative_benchmark_t4.json").read_text())
fig, axes = plt.subplots(1,2,figsize=(7.15,2.5))
rows = [r for r in cache["records"] if r["requested_output_length"]==32]
for method, label, color in [("cache","KV cache",GREEN),("no_cache_reference","No-cache reference",BLUE)]:
    axes[0].plot([r["input_length"] for r in rows],[r[method]["tokens_per_second"] for r in rows],"o-",label=label,color=color)
axes[0].set(xlabel="Prompt tokens (output cap=32)",ylabel="Decode tokens/s",title="(a) Cache reuse")
axes[0].set_xticks([64,256,512]); axes[0].legend(fontsize=7,frameon=False)
x=np.arange(len(spec["records"]))
for off,method,label,color in [(-.18,"ordinary","Ordinary greedy",GREEN),(.18,"speculative","Speculative",ORANGE)]:
    axes[1].bar(x+off,[r[method]["overall_tokens_per_second"] for r in spec["records"]],width=.36,label=label,color=color)
axes[1].set_xticks(x,["128", "512"])
axes[1].set(xlabel="Prompt tokens (64 output tokens)",ylabel="Total tokens/s",title="(b) Draft verification")
axes[1].legend(fontsize=7,frameon=False)
for ax in axes: ax.grid(axis="y",alpha=.18); ax.set_axisbelow(True)
fig.tight_layout();save(fig,"decoding_comparison")

attention=json.loads((ROOT/"results/attention_benchmark_t4.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(7.15,2.45))
for method,label,color in [("reference","Reference",BLUE),("tiled_online","Tiled online",ORANGE),("pytorch_sdpa","PyTorch SDPA",GREEN)]:
    rows=[r for r in attention["records"] if r["method"]==method]
    xs=[r["length"] for r in rows]
    axes[0].plot(xs,[r["median_seconds"]*1000 for r in rows],"o-",label=label,color=color)
    axes[1].plot(xs,[r["peak_cuda_allocated_bytes"]/1e6 for r in rows],"o-",color=color)
axes[0].set(yscale="log",xlabel="Sequence length",ylabel="Median runtime (ms, log scale)",title="(a) Attention latency")
axes[1].set(xlabel="Sequence length",ylabel="Peak CUDA allocation (MB)",title="(b) Attention memory")
axes[0].legend(fontsize=7,frameon=False)
for ax in axes: ax.set_xticks([64,256,512]); ax.grid(alpha=.18)
fig.tight_layout();save(fig,"attention_comparison")

fig, ax=plt.subplots(figsize=(7.15,3.0)); ax.set_xlim(0,10);ax.set_ylim(0,4.2);ax.axis("off")
def box(x,y,w,h,text,fill="#edf5f0"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05",edgecolor="#48675e",facecolor=fill,lw=.8))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8)
def arrow(a,b):ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","lw":.9,"color":"#48675e"})
box(.05,2.5,1.9,1.1,"Patient note\n+ criterion type\n+ criterion text")
box(2.3,2.5,1.7,1.1,"Prompt + native\ntokenizer\n(2,048-token budget)")
box(4.4,2.3,2.2,1.5,"Qwen2.5 decoder x24\nRMSNorm / RoPE / GQA\nSwiGLU / residuals\nLoRA on Q,K,V,O")
box(7.0,2.5,2.9,1.1,"Greedy JSON generation\nlabel + sentence IDs\n(cache or draft verification)")
for a,b in [((1.95,3.05),(2.3,3.05)),((4,3.05),(4.4,3.05)),((6.6,3.05),(7,3.05))]:arrow(a,b)
box(7.0,.5,2.9,1.1,"ID lookup in original note\nExact evidence + audit text\nBrowser / HTML export")
arrow((8.45,2.5),(8.45,1.6))
box(.05,.5,3.95,1.1,"Training only: expert JSON targets\nPrompt tokens masked from loss\nFrozen base weights + low-rank updates",fill="#fff3e8")
arrow((4,1.05),(5.5,2.3));
ax.text(5.5,.75,"Benchmarks: cache reuse,\nonline softmax, speculative decoding",ha="center",fontsize=7,color="#4c6059")
fig.tight_layout(pad=.2);save(fig,"system_architecture")

# Per-class table is generated directly from the recorded group metrics.
lines=[r"\begin{tabular}{llrrrr}",r"\toprule",r"Group & Class & P & R & F1 & $n$\\",r"\midrule"]
for title,g in [("Overall",fine["overall"]),("Inclusion",fine["by_criterion_type"]["inclusion"]),("Exclusion",fine["by_criterion_type"]["exclusion"])]:
    for label in g["classification"]["labels"]:
        p=g["classification"]["per_class"][label]
        lines.append(f"{title} & {abbrev[label]} & {p['precision']:.3f} & {p['recall']:.3f} & {p['f1-score']:.3f} & {int(p['support'])}"+r"\\")
    lines.append(r"\midrule")
lines[-1]=r"\bottomrule";lines.append(r"\end{tabular}")
(HERE/"per_class_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("Regenerated five figure pairs and per-class table from archived logs.")
