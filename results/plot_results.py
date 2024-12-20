# %%
import pickle
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pathlib import Path
from tap import Tap


class Args(Tap):
    bench_name: str


# plot options
ALPHA_ = 1.0
FIGSIZE_ = (7, 4)
SAVE_FORMATS = (".pdf", ".png", ".svg")
plt.rcParams["figure.dpi"] = 200
args = Args().parse_args()
BENCH_NAME = args.bench_name


def save_fig(fig: plt.Figure, name: str):
    for ext in SAVE_FORMATS:
        fig.savefig(f"{name}{ext}")


def load_files(bench_name):
    basepath = Path(bench_name)
    all_csv = list(basepath.glob("*.csv"))
    all_pkl = list(basepath.glob("*.pkl"))
    nr = len(all_csv)

    dfs = []
    suppl_data = {}
    for i in range(nr):
        dfs.append(pl.read_csv(all_csv[i]))
        with open(all_pkl[i], "rb") as f:
            suppl_data.update(pickle.load(f))
    df = pl.concat(dfs)
    return df, suppl_data, len(dfs)


# %%
df_, suppl_data_, num_instances = load_files(BENCH_NAME)
df_ = df_.with_columns(avg_iter_time=pl.col("solve_time_s") / pl.col("niter"))
print(f"Num. instances: {num_instances}")

# Create aliases for the unique configurations

# %%
solver_configs_ = [
    {**entry["solver"], "run_id": run_id} for run_id, entry in suppl_data_.items()
]

df_sc = pl.DataFrame(solver_configs_)
df_sc = df_sc.sort("config_name")
print(df_sc)

# %%
df_sc.drop("run_id").unique(maintain_order=True)

# %%
config_names = df_sc["config_name"].unique().sort()
print(config_names)

palette = sns.color_palette("tab10", len(config_names))

# %%
df_fused = df_.join(df_sc, ["run_id", "name"])
df_fused

# %%
expr_ok = pl.col("status") == "CONVERGED"
expr_ok = expr_ok & pl.col("config_name").is_in(
    [
        "ALTRO:0",
        "Ipopt:0",
        "ProxDDP:1",
        "ProxDDP:3",
    ]
)
df_ok = df_fused.filter(expr_ok)

g = df_ok.group_by("config_name")
df_niter = g.agg(pl.col("niter").sort()).sort("config_name")
df_niter

# %%
config_names_ok = df_ok["config_name"].unique()
print("Config names OK:", config_names_ok)

subpalette = [
    palette[i] for i in range(len(config_names)) if config_names[i] in config_names_ok
]
# subpalette = sns.color_palette(subpalette)
subpalette = sns.color_palette(palette)

# %%
sns.set_palette(subpalette)

# %%
max_iters = df_ok["max_iters"].max()
iterr = np.arange(1, max_iters)

# %%
plt.style.use("seaborn-v0_8-talk")
fig = plt.figure(figsize=FIGSIZE_)

for row in df_niter.iter_rows():
    lab = row[0]
    mask = iterr[:, None] >= np.asarray(row[1])
    counts = mask.sum(axis=1) / num_instances
    plt.step(iterr, counts, label=lab, alpha=ALPHA_)

plt.xlabel("Max number of iterations")
plt.ylabel("Proportion of problems solved")
plt.ylim((-0.05, 1.05))
plt.legend(fontsize=8)
plt.grid()
plt.tight_layout()

save_fig(fig, f"{BENCH_NAME}_iterations")

# %% [markdown]
# Now we look at final tolerances

# %%
solve_time_q90 = df_ok["solve_time_s"].quantile(0.99)
print("solve_time_s quantile:", solve_time_q90)

df_min_time = g.agg(pl.col("solve_time_s")).sort("config_name")
df_min_time

# %%
fig2 = plt.figure(figsize=FIGSIZE_)

for row in df_min_time.iter_rows():
    lab = row[0]
    _times = np.linspace(0.0, solve_time_q90, 101)
    mask = _times[:, None] >= np.asarray(row[1])
    counts = mask.sum(axis=1) / num_instances
    plt.step(_times, counts, label=lab, alpha=ALPHA_)

plt.xlabel("Total wall time (s)")
plt.ylabel("Proportion of problems solved")
plt.xlim((-0.05 * solve_time_q90, solve_time_q90))
plt.ylim((-0.05, 1.05))
plt.legend(fontsize=8)
plt.grid()
plt.tight_layout()

save_fig(fig2, f"{BENCH_NAME}_solve_times")

# %% [markdown]
# # Average times

# %%

fig = plt.figure()
ax = sns.stripplot(
    data=df_fused,
    x="config_name",
    y="avg_iter_time",
    linewidth=1,
    alpha=0.3,
)
ax.xaxis.set_tick_params(rotation=20)
ax.set_axisbelow(True)
ax.grid(which="minor", axis="y")
ax.set_title("Average iteration time (s)")
fig.set_size_inches(FIGSIZE_)
ax.grid(axis="both")
ax.set_xlabel("Solver (and configuration)")
ax.set_ylabel("")
fig.tight_layout()
save_fig(fig, f"{BENCH_NAME}_avg_time_violin")

# %% [markdown]
# # Performance profile

# %%
g2 = df_ok.group_by("instance")
best_times = g2.agg(best_time_s=pl.col("solve_time_s").min())

df_best_times = df_ok.join(best_times, on="instance")
df_best_times = df_best_times.with_columns(
    perf_ratio_s=pl.col("solve_time_s") / pl.col("best_time_s")
)
df_best_times

# %%
g3 = df_best_times.group_by("config_name")

df_perf_ratios = g3.agg(pl.col("perf_ratio_s")).sort("config_name")

worst_perf_ratio = df_best_times["perf_ratio_s"].max()
print("Word perf. ratio:", worst_perf_ratio)

df_perf_ratios

# %%
fig4 = plt.figure(figsize=FIGSIZE_)

xvals_ = np.power(10, np.linspace(0.0, np.log10(worst_perf_ratio), 201))

for cname, perfratios in df_perf_ratios.iter_rows():
    mask = xvals_[:, None] >= np.asarray(perfratios)
    counts = mask.sum(axis=1) / num_instances
    plt.step(xvals_, counts, label=cname, alpha=ALPHA_)

plt.xlabel("Performance ratio $\\tau$")
plt.ylabel("Proportion of problems solved")
plt.ylim((-0.05, 1.05))
plt.xscale("log", base=2)
plt.legend(fontsize=8)
plt.grid()
plt.grid(which="minor")
plt.tight_layout()

save_fig(fig4, f"{BENCH_NAME}_perfprofile_time")
