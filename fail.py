import pandas as pd

def treatment_effect_check(df, score="BEA_SCORE_WERT_HIST", treatment="Treatment",
                           outcome="TA_HUS_RS_ABSCHLUSS_CNT", sample=False, seed=42):
    """Naiver vs. auf gleiche Treatment-Rate je Score balancierter Effekt (Treatment 0/1)."""
    diff = lambda d: (d.loc[d[treatment] == 1, outcome].mean()
                      - d.loc[d[treatment] == 0, outcome].mean())

    # Tabelle je Score
    tab = df.pivot_table(index=score, columns=treatment, values=outcome, aggfunc="mean")
    tab = tab.rename(columns={0: "y_control", 1: "y_treated"})
    tab["effekt"] = tab["y_treated"] - tab["y_control"]
    tab["n"] = df.groupby(score).size()
    tab["treatment_rate"] = df.groupby(score)[treatment].mean()

    res = {
        "naiv": diff(df),
        "balanciert": (tab["effekt"] * tab["n"]).sum() / tab["n"].sum(),
    }

    # Optional: echtes Downsampling der Treatment-Gruppe auf die niedrigste Rate
    if sample:
        p = tab["treatment_rate"].min()
        odds = p / (1 - p)
        parts = []
        for _, g in df.groupby(score):
            c, t = g[g[treatment] == 0], g[g[treatment] == 1]
            parts.append(c)
            parts.append(t.sample(n=min(len(t), round(odds * len(c))), random_state=seed))
        res["gesampelt"] = diff(pd.concat(parts))

    res = pd.Series(res, name="effekt")
    res["verzerrung_naiv"] = res["naiv"] - res["balanciert"]
    return tab, res


tab, res = treatment_effect_check(df, sample=True)
print(tab)
print(res)
