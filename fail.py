import matplotlib.pyplot as plt
import numpy as np


def plot_kumulierter_cate(df, cate_col, sort_col, ascending=False,
                          xlim=None, ylim=None, ax=None):
    """Sortiert nach sort_col, summiert cate_col kumuliert auf, zeichnet die Linie."""
    cum = (
        df.sort_values(sort_col, ascending=ascending)[cate_col]
        .to_numpy()
        .cumsum()
    )
    x = np.arange(1, len(cum) + 1)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, cum)
    if xlim is not None:
        ax.set_xlim(*xlim)      # für den Vergleich bei beiden Aktionen gleich setzen
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Anzahl (nach Rang)")
    ax.set_ylabel("kumulierter CATE")
    return ax
