def check_functional_dependency(df: pd.DataFrame, key_col: str, value_col: str) -> pd.DataFrame:
    """
    Prüft, ob jeder Wert in `key_col` immer genau einen Wert in `value_col` hat.

    Gibt einen DataFrame mit allen Keys zurück, die mehrfach vorkommen,
    inkl. Anzahl Zeilen, Anzahl unterschiedlicher Values und einem Flag
    `consistent` (True = immer gleicher Value).
    """
    grp = (
        df.groupby(key_col, dropna=False)[value_col]
          .agg(n_rows="size", n_unique="nunique", values=lambda s: sorted(s.dropna().unique()))
          .reset_index()
    )
    grp = grp[grp["n_rows"] > 1].copy()          # nur mehrfach vorkommende Keys
    grp["consistent"] = grp["n_unique"] <= 1

    n_incons = (~grp["consistent"]).sum()
    print(f"{len(grp)} Keys kommen mehrfach vor, davon {n_incons} mit variierendem '{value_col}'.")

    return grp.sort_values("consistent")
