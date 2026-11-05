tab = pd.crosstab(df["produkt"], df["binaer"], normalize="columns")
tab["diff"] = tab[1] - tab[0]
tab.sort_values("diff")
