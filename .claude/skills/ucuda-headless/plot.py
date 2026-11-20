"""Быстрый PNG по CSV ucuda_run: тип графика — по суффиксу имени (_bif1d, _lle2d, ...).
    python plot.py <file.csv> [column]  ->  <file>.png рядом с CSV
column — что рисовать на 2D-карте / в metrics (по умолчанию: lle, period, LE1, basin, Max)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
col = sys.argv[2] if len(sys.argv) > 2 else None
task = path.rsplit("_", 1)[-1].removesuffix(".csv")
data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
names = data.dtype.names
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

if task == "bif1d":
    m = np.isfinite(data["value"])
    ax.plot(data[names[0]][m], data["value"][m], ",k", alpha=0.6)
    ax.set_ylabel("peaks")
elif task in ("lle1d", "ls1d"):
    for c in names[1:-1]:
        ax.plot(data[names[0]], data[c], lw=0.8, label=c)
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend()
elif task in ("bif2d", "lle2d", "ls2d", "basins") or (task == "metrics" and names[1] != "regime"):
    default = {"bif2d": "period", "lle2d": "lle", "ls2d": "LE1", "basins": "basin", "metrics": "Max"}[task]
    c = col or default
    x, y = np.unique(data[names[0]]), np.unique(data[names[1]])
    z = data[c].reshape(len(y), len(x))
    im = ax.pcolormesh(x, y, np.ma.masked_invalid(z), shading="nearest",
                       cmap="tab20" if task in ("basins", "bif2d") else "turbo")
    fig.colorbar(im, ax=ax, label=c)
    ax.set_ylabel(names[1])
elif task == "metrics":
    c = col or "Max"
    ax.plot(data[names[0]], data[c], lw=0.8)
    ax.set_ylabel(c)
elif task == "dft1d":
    x, f = np.unique(data[names[0]]), np.unique(data["freq"])
    z = data["amplitude"].reshape(len(x), len(f)).T
    im = ax.pcolormesh(x, f, np.log10(z + 1e-300), shading="nearest", cmap="magma")
    fig.colorbar(im, ax=ax, label="log10 amplitude")
    ax.set_ylabel("frequency")
else:
    sys.exit(f"unknown task suffix '{task}'")

ax.set_xlabel(names[0])
ax.set_title(path.replace("\\", "/").rsplit("/", 1)[-1])
fig.tight_layout()
out = path.removesuffix(".csv") + ".png"
fig.savefig(out)
print(out)
