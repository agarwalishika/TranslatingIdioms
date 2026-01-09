import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from glob import glob
import pandas as pd
import os
import json

def run_everything(EVAL_TYPE="chinese_eval"):

    files = glob(f'results/{EVAL_TYPE}/*.csv')

    codes = list(set([f[:f.rfind("_")]for f in files]))

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    CODES_CANONICAL = [
        "Llama GRPO DA",
        "Llama GRPO QE-C",
        "Llama GRPO QE-N",
        "Llama GRPO QE-P",
        "Llama SFT",
        "Llama TF",
        "Llama LIA",
        "Llama Base",
        "Qwen GRPO DA",
        "Qwen GRPO QE-C",
        "Qwen GRPO QE-N",
        "Qwen GRPO QE-P",
        "Qwen SFT",
        "Qwen TF",
        "Qwen LIA",
        "Qwen Base",
        "Command-R 32B",
        "Command-R 7B",
        "NLLB-3.3B",
        "NLLB-1.3B"
    ]
    CODES_CANONICAL = reversed(CODES_CANONICAL)

    # 2) Build a stable mapping: item -> RGBA color
    cmap = plt.get_cmap("tab20b")
    plot_colors = {code: cmap(i % cmap.N) for i, code in enumerate(CODES_CANONICAL)}

    # Optional: fail fast if you accidentally try to plot something unmapped
    def colors_for(codes_list):
        missing = [c for c in codes_list if c not in plot_colors]
        if missing:
            raise KeyError(f"Missing colors for: {missing}")
        return [plot_colors[c] for c in codes_list]

    def print_rows(k=1):
        all_results = {}
        for code in codes:
            results = {
                "qe": 0,
                "da": 0,
                "rouge": 0,
                "embed_distance": 0,
                "laj": 0
            }

            is_evaluate = True
            for i in range(k):
                if not os.path.exists(f'{code}_{i}.csv'):
                    is_evaluate = False
                    continue
                df = pd.read_csv(f'{code}_{i}.csv', sep="|")

                # df = df[df['predicted'].notna()]
                # df = df[df['predicted'].apply(lambda x: len(x) > 1)]

                results['qe'] += df['qe'].mean()
                results['da'] += df['da'].mean()
                results['rouge'] += df['rouge'].mean()
                results['embed_distance'] += df['embed_distance'].mean()
                results['laj'] += df['laj'].mean()
            
            if not is_evaluate: continue
            results['qe'] /= k
            results['da'] /= k
            results['rouge'] /= k
            results['embed_distance'] /= k
            results['laj'] /= k

            print(f'\n\n\n PASS @ {k} RESULTS for {code}')
            print("".join(["#"] * 150))

            da = round(results['da'] * 100, 2)
            qe = round(results['qe'] * 100, 2)
            rouge = round(results['rouge'] * 100, 2)
            embed_dist = round(results['embed_distance'] * 100, 2)
            laj = round(results['laj'], 2)

            print(f'& {da} & {qe} & {rouge} & {embed_dist} & {laj} \\\\')

            print("".join(["#"] * 150))
            print(f'PASS @ {k} RESULTS\n\n\n')

            all_results[code] = results

        return all_results


    CANONICAL_ORDER = [
        # NLLB
        "nllb 1.3B",
        "nllb 3.3B",

        # Command R
        "command r 7b",
        "command r 32b",

        # Qwen
        "qwen base",
        "qwen lia",
        "qwen trainingfree",
        "qwen sft",
        "qwen qe pos",
        "qwen qe neg",
        "qwen qe cons",
        "qwen da",

        # LLaMA
        "llama base",
        "llama lia",
        "llama trainingfree",
        "llama sft",
        "llama qe pos",
        "llama qe neg",
        "llama qe cons",
        "llama da",
    ]

    def normalize(code: str) -> str:
        c = code.lower()

        if "nllb" and "1.3b" in c: return "nllb 1.3B"
        if "nllb" and "3.3b" in c: return "nllb 3.3B"

        # remove path + suffix noise
        c = c.replace("results/", "")
        c = c.replace("-chinese-results", "")
        c = c.replace("_chinese_", " ")
        c = c.replace("-chinese-", " ")
        c = c.replace("-hindi-results", "")
        c = c.replace("_hindi_", " ")
        c = c.replace("-hindi-", " ")

        # normalize separators
        for ch in ["_", "-", "="]:
            c = c.replace(ch, " ")

        # collapse known aliases
        c = c.replace("distilled", "")
        c = c.replace("grpo", "")
        c = c.replace("qe positive", "qe pos")
        c = c.replace("qe negative", "qe neg")

        c = c.replace('llama8b', 'llama')
        c = c.replace('llama 8b', 'llama')
        c = c.replace('qwen3b', 'qwen')
        c = c.replace('qwen 3b', 'qwen')

        # normalize spacing
        c = " ".join(c.split())

        return c

    ORDER_INDEX = {k: i for i, k in enumerate(CANONICAL_ORDER)}

    def sort_code(code: str):
        norm = normalize(code)

        for key, idx in ORDER_INDEX.items():
            if key in norm:
                return idx

        # unmatched stuff goes last (but visible)
        return len(CANONICAL_ORDER)



    def plot_per_metric(k):
        output_dir = f"plots/{EVAL_TYPE}"
        all_results = print_rows(k=k)
        # os.makedirs(output_dir, exist_ok=True)

        # Keep a fixed order for columns
        metric_order = ["da", "qe", "rouge", "embed_distance", "laj"]
        metrics = {
            "da": ("DA (%)", 100),
            "qe": ("QE (%)", 100),
            "rouge": ("ROUGE (%)", 100),
            "embed_distance": ("Embedding Distance (%)", 100),
            "laj": ("LAJ (x/5)", 1),
        }

        codes_list = sorted(list(all_results.keys()), key=sort_code)

        # --- label mapping (persisted) ---
        if os.path.exists("plot_labels.json"):
            plot_labels = json.load(open("plot_labels.json", "r"))
        else:
            plot_labels = {}

        for code in codes_list:
            if code not in plot_labels:
                plot_labels[code] = input(f"plot label for {code}: ")

        json.dump(plot_labels, open("plot_labels.json", "w"), indent=2)

        labels = [plot_labels[c] for c in codes_list]
        n = len(codes_list)
        y = np.arange(n)

        # --- consistent colors per CODE (not per label) ---
        # cmap = plt.get_cmap("tab20b")
        # code_to_color = {c: cmap(i % cmap.N) for i, c in enumerate(codes_list)}
        bar_colors = colors_for(labels)

        # --- layout: 1 legend column + N metric columns ---
        # Tweak these if you want wider/narrower panels
        legend_col_w = 2.0
        metric_col_w = 3.2
        fig_w = legend_col_w + metric_col_w * len(metric_order)
        fig_h = max(6.0, 0.35 * n + 1.0)

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            nrows=1,
            ncols=1 + len(metric_order),
            width_ratios=[legend_col_w] + [metric_col_w] * len(metric_order),
            wspace=0.08,
        )

        # --- left "legend" axis (aligned with rows) ---
        ax_leg = fig.add_subplot(gs[0, 0])
        ax_leg.set_xlim(0, 1)
        ax_leg.set_ylim(-0.5, n - 0.5)
        ax_leg.invert_yaxis()
        ax_leg.axis("off")

        # Draw colored squares + text, aligned with each bar row
        x0 = 0.02
        swatch_w = 0.06
        for yi, (lab, col) in enumerate(zip(labels, bar_colors)):
            ax_leg.add_patch(Rectangle((x0, yi - 0.30), swatch_w, 0.60, color=col, ec="none"))
            ax_leg.text(x0 + swatch_w + 0.03, yi, lab, va="center", ha="left", fontsize=10)

        # --- helper for midrules/break bands ---
        def add_breaks(ax):
            pad = 0.25
            lw = 1.5
            breaks = [4,12] if "transfer" not in EVAL_TYPE else [5]
            for b in breaks:
                ax.axhspan(b - 0.5 - pad, b - 0.5 + pad, color="white", zorder=2)
                ax.axhline(y=b - 0.5, color="black", linewidth=lw, alpha=0.7, zorder=3)

        # --- metric axes ---
        metric_axes = []
        for j, metric in enumerate(metric_order):
            ax = fig.add_subplot(gs[0, j + 1], sharey=ax_leg)
            metric_axes.append(ax)

            title, scale = metrics[metric]
            vals = np.array([all_results[c][metric] * scale for c in codes_list], dtype=float)

            bars = ax.barh(y, vals, color=bar_colors, height=0.80, zorder=1)

            # x-limits similar to your per-plot logic
            vmin, vmax = float(vals.min()), float(vals.max())
            ax.set_xlim(vmin * 0.9, vmax * 1.1) if vmin != 0 else ax.set_xlim(0, vmax * 1.1)

            # value labels
            dx = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
            for yi, v in zip(y, vals):
                ax.text(v + dx, yi, f"{v:.2f}", va="center", ha="left", fontsize=9)

            add_breaks(ax)

            ax.set_title(title)
            ax.tick_params(axis="y", left=False, labelleft=False)  # no y labels anywhere (legend handles it)
            ax.grid(False)

        # Optional: if you want x tick labels only on bottom, keep as-is.
        # If you want fewer x ticks, add: ax.xaxis.set_major_locator(...)

        fig.suptitle("")  # keep empty; you already have per-panel titles
        fig.tight_layout()

        save_path = os.path.join('plots', f"{EVAL_TYPE}.pdf")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print("Saved:", save_path)

    plot_per_metric(1)

run_everything("chinese_eval")
run_everything("hindi_eval")
run_everything("opus_chinese_eval")
run_everything("opus_hindi_eval")
run_everything("opus_transfer_hi2zh")
run_everything("opus_transfer_zh2hi")
run_everything("transfer_eval_hi2zh")
run_everything("transfer_eval_zh2hi")
