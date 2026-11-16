"""Figura: la galaxia ejemplo en g, r, z y la suma g+r+z (caja de cada una, franja ±10 %, perfiles de ε y resumen),
y la tabla de las 448 en r, g, z, suma y JPG (de data_barras/grz_448.csv)."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from matplotlib.patches import Circle

import _estilo as e
from src import measure

warnings.filterwarnings("ignore")
AS = r"$^{\prime\prime}$"
PS, Z, REF, ERR = 0.262, 0.039407, e.A_REF_EXAMPLE, 0.10
B = e.DATA / "bandas"

# --- tabla de las 448 ---
t = pd.read_csv(e.DATA / "grz_448.csv")
a = t.pivot(index="TARGETID", columns="metodo", values="a")
ref = pd.read_parquet(e.DATA / "reference_all.parquet").set_index("TARGETID").a_ref
M = ["r", "g", "z", "suma", "jpg"]
q = a[M].div(ref, axis=0)
q = q[q.notna().all(axis=1)]
ok = (q - 1).abs() < 0.2
print(f"N = {len(q)} con los cinco medidos")
print(pd.DataFrame({m: [ok[m].mean(), ((q[m] - 1).abs() < 0.1).mean(), q[m].median(),
                        (q[m].quantile(.84) - q[m].quantile(.16)) / 2, (ok[m] & ~ok.r).sum(), (~ok[m] & ok.r).sum()]
                    for m in M}, index=["<20%", "<10%", "mediana", "semiancho", "arregla", "rompe"]).round(3).to_string())
f = q[~ok.r]
print(f"fallan en r: {len(f)} ({(f.r < .8).sum()} cortas, {(f.r > 1.2).sum()} se pasan); r y suma a <10 %: "
      f"{((f.r / f.suma - 1).abs() < .1).sum()}")

# --- ejemplo ---
ims = {b: fits.getdata(B / f"ej_desi_{b}.fits").astype(float) for b in "grz"}
ims["suma"] = ims["g"] + ims["r"] + ims["z"]
COL = {"g": e.ISO, "r": e.COMB, "z": e.FOU, "suma": e.DEC}
LAB = {"g": "$g$", "r": "$r$", "z": "$z$", "suma": "suma $g+r+z$"}
res = {}
for b, im in ims.items():
    iso = B / f"ej_{b}_iso.ecsv"
    m = measure.measure_image(im, PS, Z, iso_path=str(iso))
    res[b] = (m, *e.bar.profiles(Table.read(iso), PS)[:2])
    print(f"ejemplo {b:5s} a = {m['a_bar']:.2f} ± {ERR * m['a_bar']:.2f}″  W = {m['W_bar']:.2f}″  a/Valeria = {m['a_bar'] / REF:.2f}")

e.cm.style(e.W2)
fig = plt.figure(figsize=(e.W2, 0.62 * e.W2), layout="constrained")
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.85])
h = 1.45 * 16.5 / PS
for k, (b, (m, _, _)) in enumerate(res.items()):
    s = fig.add_subplot(gs[0, k]); aa, c = m["a_bar"], COL[b]
    rot, cx, cy = e.on_x(ims[b], m["x0"], m["y0"], (m["pa_bar"] + 90) % 180)
    s.imshow(rot, origin="lower", cmap="gray_r",
             norm=ImageNormalize(rot, interval=PercentileInterval(99.8), stretch=AsinhStretch(0.03)))
    for sg in (-1, 1):
        s.axvspan(cx + sg * (1 - ERR) * aa / PS, cx + sg * (1 + ERR) * aa / PS, color=c, alpha=0.25, lw=0)
    s.plot(*e.boxy(aa / PS, m["W_bar"] / 2 / PS, 0, cx, cy), color=c, lw=1.3)
    s.add_patch(Circle((cx, cy), REF / PS, fill=False, color=e.REF, ls="--", lw=0.8))
    s.set_xlim(cx - h, cx + h), s.set_ylim(cy - h, cy + h), s.set_xticks([]), s.set_yticks([])
    s.set_title(f"({'abcd'[k]}) {LAB[b]}\n$a$ = {aa:.1f} $\\pm$ {ERR * aa:.1f}{AS}", fontsize=6, color=c)
p = fig.add_subplot(gs[1, :3])
for b, (m, x, eps) in res.items():
    p.plot(x, eps, ".-" if b != "suma" else ":", color=COL[b], ms=1.8, lw=0.8, label=LAB[b])
    p.axvline(m["a_bar"], color=COL[b], lw=0.8)
p.axvline(REF, color=e.REF, ls="--", lw=0.8, label="Valeria")
p.set_xlim(0, 25), p.set_ylim(0, 0.8), p.set_xlabel(r"$a$ [" + AS + "]"), p.set_ylabel(r"$\varepsilon$")
p.legend(frameon=False, fontsize=5.5, loc="upper left"), p.set_title("(e)", fontsize=6, loc="left")
s = fig.add_subplot(gs[1, 3])
for y, (b, (m, _, _)) in enumerate(res.items()):
    s.errorbar(m["a_bar"], y, xerr=ERR * m["a_bar"], fmt="o", color=COL[b], ms=3, lw=1, capsize=2)
s.axvline(REF, color=e.REF, ls="--", lw=0.8)
s.axvspan(0.8 * REF, 1.2 * REF, color="0.5", alpha=0.1, lw=0)
s.set_yticks(range(len(res))), s.set_yticklabels(["$g$", "$r$", "$z$", "suma"]), s.invert_yaxis()
s.set_xlabel(r"$a$ [" + AS + "]"), s.set_xlim(10, 19), s.set_title("(f)", fontsize=6, loc="left")
e.save(fig, "bandas")
