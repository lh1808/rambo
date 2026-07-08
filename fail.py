import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
 
 
def plot_kumulierter_cate(df, cate_col, sort_col, ascending=False, step=50_000,
                          xlim=None, ylim=None, ax=None):
    """Sortiert nach sort_col, summiert cate_col kumuliert auf und zeichnet die Linie.
 
    Alle `step` Samples wird ein Stützpunkt gesetzt: an der Linie steht der
    kumulierte Wert, zwischen zwei Stützpunkten der Zuwachs zum letzten Punkt.
    """
    def de(v):  # deutsche Tausenderpunkte
        return f"{v:,.0f}".replace(",", ".")
 
    cum = (df.sort_values(sort_col, ascending=ascending)[cate_col]
             .to_numpy(dtype=float).cumsum())
    n = len(cum)
    x = np.arange(1, n + 1)
 
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.2))
 
    ax.plot(x, cum, color="#9B111E", lw=2)
 
    # Stützpunkte strikt auf den Vielfachen von step (<= n).
    # Ein eventueller Rest dahinter läuft nur als Linie weiter – ohne Punkt,
    # ohne Zuwachs.
    marks = list(range(step, n + 1, step))
 
    prev_x, prev_c = 0, 0.0
    for i, m in enumerate(marks):
        c = cum[m - 1]
 
        # Punkt + kumulierter Wert oberhalb
        ax.plot(m, c, "o", color="#9B111E", ms=5)
        last = i == len(marks) - 1
        ax.annotate(de(c), (m, c), textcoords="offset points",
                    xytext=(-2 if last else 0, 10),
                    ha="right" if last else "center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#9B111E")
 
        # Zuwachs seit letztem Punkt, unter der Sehnenmitte
        ax.annotate(f"+{de(c - prev_c)}",
                    ((prev_x + m) / 2, (prev_c + c) / 2),
                    textcoords="offset points", xytext=(0, -13),
                    ha="center", va="top", fontsize=8, color="#7A7A7A")
 
        prev_x, prev_c = m, c
 
    # Achsen
    ax.set_xlim(*(xlim if xlim is not None else (0, n)))
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, cum[-1] * 1.12)          # Luft für Labels oben
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: de(v)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: de(v)))
    ax.set_xlabel("Anzahl (nach Rang)")
    ax.set_ylabel("kumulierter CATE")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.figure.tight_layout()
    return ax
