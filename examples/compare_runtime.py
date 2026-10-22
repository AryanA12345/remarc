import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from examples.mdp_runtime import time_mdp
from examples.rl_runtime import time_rl

SEQ_LENGTHS = [1, 2, 3]  
DRUG_COUNTS = [2, 5, 10]

out_dir = Path(__file__).resolve().parent.parent / "log" / "runtime"


def make_plot(df):
    genotypes = sorted(df["genotypes"].unique())
    positions = {g: i for i, g in enumerate(genotypes)}  
    drugs = sorted(df["drugs"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, col, title in [(axes[0], "mdp_s", "MDP"), (axes[1], "rl_s", "RL ")]:
        for D in drugs:
            g = df[df["drugs"] == D].dropna(subset=[col]).sort_values("genotypes")
            ax.plot([positions[x] for x in g["genotypes"]], g[col],
                    marker="o", label=f"{D} drugs")
        ax.set_xticks(range(len(genotypes)))
        ax.set_xticklabels(genotypes)
        ax.set_xlabel("Number of genotypes")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Runtime (seconds)")

    not_run = df[df["mdp_s"].isna()]["genotypes"]
    if len(not_run):
        axes[0].text(0.97, 0.5, f"MDP too large to run\nat {not_run.min()}+ genotypes",
                     transform=axes[0].transAxes, ha="right", va="center", color="gray")

    plt.tight_layout()
    plt.savefig(out_dir / "runtime.png", dpi=200)
    print(f"Saved graph to {out_dir / 'runtime.png'}")


if __name__ == "__main__":
    rows = []
    for N in SEQ_LENGTHS:
        for D in DRUG_COUNTS:
            mdp_s = time_mdp(N, D)
            rl_s = time_rl(N, D)
            rows.append({"genotypes": 2**N, "drugs": D, "mdp_s": mdp_s, "rl_s": rl_s})
            print(f"\n>>> DONE genotypes={2**N} drugs={D} mdp={mdp_s} rl={rl_s}\n")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    make_plot(df)
    plt.show()