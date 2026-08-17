"""
kundenwert_analyse.py
=====================

Segmentierung von Kunden (vn_partner_id) in "wertvoll" (positiv) vs. "nicht wertvoll"
(negativ) und anschließender Vergleich der beiden Gruppen.

Klassifikationsregel (je vn_partner_id, über alle Verträge aggregiert):
    negativ, wenn  sum(ve_profit) < 0
                   UND |sum(ve_profit)| >= 20 % von sum(ve_bestandsjahresnettobeitrag)
    sonst positiv.

Erwartete Spalten:
    vn_partner_id, ve_id, ve_produkt, ve_sparte, ve_gesellschaft,
    ve_bestandsjahresnettobeitrag, ve_expected_claim_amount,
    ve_total_cost, ve_cr, ve_profit

Schnellstart:
    import pandas as pd
    from kundenwert_analyse import analysiere_kundenwert, drucke_report, exportiere_excel

    erg = analysiere_kundenwert(df)              # schwelle=0.20
    drucke_report(erg)
    exportiere_excel(erg, "kundenwert_analyse.xlsx")

    erg["kunden"]          # Kundenebene inkl. wert_flag  -> für eigene Auswertungen
    erg["kennzahlen"]      # Kennzahlenvergleich negativ vs. positiv
    erg["sparten_mix"]     # Sparten-Zusammensetzung im Mittel
    erg["produkt_mix"]     # Produkt-Zusammensetzung im Mittel
    ...
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Spaltennamen zentral (hier anpassen, falls sie im Datensatz anders heißen)
# --------------------------------------------------------------------------------------
ID = "vn_partner_id"
VERTRAG = "ve_id"
PRODUKT = "ve_produkt"
SPARTE = "ve_sparte"
GESELLSCHAFT = "ve_gesellschaft"
BEITRAG = "ve_bestandsjahresnettobeitrag"
SCHADEN = "ve_expected_claim_amount"
KOSTEN = "ve_total_cost"
CR = "ve_cr"
PROFIT = "ve_profit"

NUM_SPALTEN = [BEITRAG, SCHADEN, KOSTEN, CR, PROFIT]
STR_SPALTEN = [PRODUKT, SPARTE, GESELLSCHAFT]


# ======================================================================================
# 1) Vorbereitung
# ======================================================================================
def bereinige(df: pd.DataFrame, kopie: bool = True) -> pd.DataFrame:
    """Typen sichern, fehlende Strings auffüllen, Dubletten auf ve_id melden."""
    d = df.copy() if kopie else df

    for c in NUM_SPALTEN:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in STR_SPALTEN:
        if c in d.columns:
            d[c] = d[c].astype("string").fillna("(unbekannt)").str.strip()

    if VERTRAG in d.columns and d[VERTRAG].duplicated().any():
        n_dub = int(d[VERTRAG].duplicated().sum())
        print(f"[Hinweis] {n_dub:,} doppelte ve_id im Datensatz - Zeilen = Verträge wird "
              f"dadurch ungenau. Ggf. vorher aggregieren/deduplizieren.")

    fehlend = d[BEITRAG].isna().sum() + d[PROFIT].isna().sum()
    if fehlend:
        print(f"[Hinweis] {int(fehlend):,} fehlende Werte in Beitrag/Profit -> als 0 gewertet.")
        d[[BEITRAG, PROFIT]] = d[[BEITRAG, PROFIT]].fillna(0)

    return d


# ======================================================================================
# 2) Kundenaggregat + Klassifikation
# ======================================================================================
def kunden_aggregat(df: pd.DataFrame, schwelle: float = 0.20) -> pd.DataFrame:
    """
    Aggregiert auf vn_partner_id und klassifiziert in 'negativ' / 'positiv'.

    Regel (einzige):
        negativ  <=>  sum(ve_profit) < 0
                      UND  |sum(ve_profit)| >= schwelle * sum(ve_bestandsjahresnettobeitrag)

    Äquivalent: marge = sum(profit)/sum(beitrag) <= -schwelle (bei Beitrag > 0).
    Bei Beitrag <= 0 greift die Bedingung über den Absolutbetrag; die Spalte marge
    bleibt dort NaN, weil sie ökonomisch nicht interpretierbar ist.
    """
    d = df
    agg = d.groupby(ID, observed=True).agg(
        n_vertraege=(VERTRAG, "size"),
        beitrag_sum=(BEITRAG, "sum"),
        profit_sum=(PROFIT, "sum"),
        schaden_sum=(SCHADEN, "sum") if SCHADEN in d.columns else (BEITRAG, "size"),
        kosten_sum=(KOSTEN, "sum") if KOSTEN in d.columns else (BEITRAG, "size"),
        n_sparten=(SPARTE, "nunique"),
        n_produkte=(PRODUKT, "nunique"),
        n_gesellschaften=(GESELLSCHAFT, "nunique"),
    )

    agg["beitrag_je_vertrag"] = agg["beitrag_sum"] / agg["n_vertraege"]
    agg["profit_je_vertrag"] = agg["profit_sum"] / agg["n_vertraege"]
    agg["marge"] = np.where(agg["beitrag_sum"] > 0,
                            agg["profit_sum"] / agg["beitrag_sum"], np.nan)
    agg["cr_gesamt"] = np.where(agg["beitrag_sum"] > 0,
                                (agg["schaden_sum"] + agg["kosten_sum"]) / agg["beitrag_sum"],
                                np.nan)

    # Anteil verlustbringender Verträge je Kunde (Zusatzinfo, nicht Teil der Regel)
    neg_vertrag = d.assign(_neg=(d[PROFIT] < 0)).groupby(ID, observed=True)["_neg"].mean()
    agg["anteil_verlustvertraege"] = neg_vertrag

    # Klassifikation: Profit negativ UND betragsmäßig mind. `schwelle` vom Beitrag
    ist_negativ = (agg["profit_sum"] < 0) & \
                  (agg["profit_sum"].abs() >= schwelle * agg["beitrag_sum"])

    agg["wert_flag"] = np.where(ist_negativ.fillna(False), "negativ", "positiv")
    agg["regel"] = f"profit_sum < 0 und |profit_sum| >= {schwelle:.0%} * beitrag_sum"
    return agg


# ======================================================================================
# 3) Kennzahlenvergleich negativ vs. positiv
# ======================================================================================
def _quantil(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else np.nan


def kennzahlen_vergleich(kunden: pd.DataFrame) -> pd.DataFrame:
    """Zentrale Vergleichstabelle: eine Zeile je Kennzahl, Spalten negativ/positiv/Delta."""
    zeilen: Dict[str, Dict[str, float]] = {}
    gruppen = {g: kunden[kunden["wert_flag"] == g] for g in ("negativ", "positiv")}

    ges_kunden = len(kunden)
    ges_vertraege = kunden["n_vertraege"].sum()
    ges_beitrag = kunden["beitrag_sum"].sum()
    ges_profit = kunden["profit_sum"].sum()

    for g, k in gruppen.items():
        n = len(k)
        v = k["n_vertraege"]
        b = k["beitrag_sum"]
        p = k["profit_sum"]
        werte = {
            "Kunden (Anzahl)": n,
            "Kunden (Anteil)": n / ges_kunden if ges_kunden else np.nan,
            "Verträge (Anzahl gesamt)": v.sum(),
            "Verträge (Anteil am Bestand)": v.sum() / ges_vertraege if ges_vertraege else np.nan,
            "Verträge je Kunde (Mittel)": v.mean(),
            "Verträge je Kunde (Median)": v.median(),
            "Verträge je Kunde (p25)": _quantil(v, 0.25),
            "Verträge je Kunde (p75)": _quantil(v, 0.75),
            "Verträge je Kunde (max)": v.max(),
            "Anteil Kunden mit nur 1 Vertrag": (v == 1).mean(),
            "Anteil Kunden mit >=3 Verträgen": (v >= 3).mean(),
            "Sparten je Kunde (Mittel)": k["n_sparten"].mean(),
            "Produkte je Kunde (Mittel)": k["n_produkte"].mean(),
            "Gesellschaften je Kunde (Mittel)": k["n_gesellschaften"].mean(),
            "Anteil Kunden mit >1 Sparte": (k["n_sparten"] > 1).mean(),
            "Beitrag gesamt": b.sum(),
            "Beitrag (Anteil am Gesamtbeitrag)": b.sum() / ges_beitrag if ges_beitrag else np.nan,
            "Beitrag je Kunde (Mittel)": b.mean(),
            "Beitrag je Kunde (Median)": b.median(),
            "Beitrag je Kunde (p25)": _quantil(b, 0.25),
            "Beitrag je Kunde (p75)": _quantil(b, 0.75),
            "Beitrag je Vertrag (gepoolt)": b.sum() / v.sum() if v.sum() else np.nan,
            "Beitrag je Vertrag je Kunde (Mittel)": k["beitrag_je_vertrag"].mean(),
            "Beitrag je Vertrag je Kunde (Median)": k["beitrag_je_vertrag"].median(),
            "Profit gesamt": p.sum(),
            "Profit (Anteil am Gesamtprofit)": p.sum() / ges_profit if ges_profit else np.nan,
            "Profit je Kunde (Mittel)": p.mean(),
            "Profit je Kunde (Median)": p.median(),
            "Profit je Vertrag (gepoolt)": p.sum() / v.sum() if v.sum() else np.nan,
            "Marge gepoolt (Profit/Beitrag)": p.sum() / b.sum() if b.sum() else np.nan,
            "Marge je Kunde (Median)": k["marge"].median(),
            "CR gepoolt": (k["schaden_sum"].sum() + k["kosten_sum"].sum()) / b.sum() if b.sum() else np.nan,
            "Anteil Verlustverträge je Kunde (Mittel)": k["anteil_verlustvertraege"].mean(),
        }
        zeilen[g] = {kk: float(vv) for kk, vv in werte.items()}

    out = pd.DataFrame(zeilen)
    out["differenz_neg_minus_pos"] = out["negativ"] - out["positiv"]
    out["verhaeltnis_neg_zu_pos"] = np.where(out["positiv"] != 0,
                                             out["negativ"] / out["positiv"], np.nan)
    out.index.name = "kennzahl"
    return out


# ======================================================================================
# 4) Mix-Analysen (Sparte / Produkt / Gesellschaft / Sparte x Produkt)
# ======================================================================================
def mix_analyse(
    df: pd.DataFrame,
    kunden: pd.DataFrame,
    spalten: Union[str, Sequence[str]] = SPARTE,
    sortiere_nach: str = "anteil_vertraege_negativ",
    min_vertraege: int = 0,
) -> pd.DataFrame:
    """
    Zusammensetzung der Gruppen nach Kategorie (z. B. Sparte oder Produkt).

    Geliefert werden je Gruppe (negativ/positiv):
      - anteil_vertraege_*  : mittlerer kundenindividueller Anteil dieser Kategorie
                              an allen Verträgen des Kunden ("im Mittel zusammengesetzt aus")
      - anteil_beitrag_*    : dito, aber beitragsgewichtet
      - penetration_*       : Anteil der Kunden der Gruppe mit mind. 1 Vertrag der Kategorie
      - vertraege_*, beitrag_*, profit_*, marge_*
      - delta_*             : negativ minus positiv (in Prozentpunkten)
    """
    keys: List[str] = [spalten] if isinstance(spalten, str) else list(spalten)

    d = df[[ID, *keys, BEITRAG, PROFIT]].copy()
    d["wert_flag"] = d[ID].map(kunden["wert_flag"])
    d = d[d["wert_flag"].notna()]

    # Kunde x Kategorie
    kk = d.groupby([ID, *keys], observed=True).agg(
        n=(BEITRAG, "size"), beitrag=(BEITRAG, "sum"), profit=(PROFIT, "sum")
    ).reset_index()

    ges = d.groupby(ID, observed=True).agg(n_ges=(BEITRAG, "size"),
                                           beitrag_ges=(BEITRAG, "sum")).reset_index()
    kk = kk.merge(ges, on=ID, how="left")
    kk["anteil_n"] = kk["n"] / kk["n_ges"]
    kk["anteil_b"] = np.where(kk["beitrag_ges"] > 0, kk["beitrag"] / kk["beitrag_ges"], np.nan)
    kk["wert_flag"] = kk[ID].map(kunden["wert_flag"])

    n_kunden = kunden["wert_flag"].value_counts()

    g = kk.groupby(["wert_flag", *keys], observed=True).agg(
        kunden_mit=(ID, "nunique"),
        vertraege=("n", "sum"),
        beitrag=("beitrag", "sum"),
        profit=("profit", "sum"),
        summe_anteil_n=("anteil_n", "sum"),
        summe_anteil_b=("anteil_b", "sum"),
    ).reset_index()

    g["n_kunden_gruppe"] = g["wert_flag"].map(n_kunden)
    # Kunden ohne diese Kategorie haben Anteil 0 -> Summe / alle Kunden der Gruppe
    g["anteil_vertraege"] = g["summe_anteil_n"] / g["n_kunden_gruppe"]
    g["anteil_beitrag"] = g["summe_anteil_b"] / g["n_kunden_gruppe"]
    g["penetration"] = g["kunden_mit"] / g["n_kunden_gruppe"]
    g["beitrag_je_vertrag"] = g["beitrag"] / g["vertraege"]
    g["marge"] = np.where(g["beitrag"] > 0, g["profit"] / g["beitrag"], np.nan)

    metriken = ["anteil_vertraege", "anteil_beitrag", "penetration",
                "kunden_mit", "vertraege", "beitrag", "profit",
                "beitrag_je_vertrag", "marge"]
    wide = g.pivot_table(index=keys, columns="wert_flag", values=metriken, observed=True)
    wide.columns = [f"{m}_{f}" for m, f in wide.columns]
    wide = wide.reindex(columns=[f"{m}_{f}" for m in metriken for f in ("negativ", "positiv")])
    wide = wide.fillna({c: 0 for c in wide.columns if not c.startswith("marge")})

    for m in ["anteil_vertraege", "anteil_beitrag", "penetration",
              "beitrag_je_vertrag", "marge"]:
        wide[f"delta_{m}"] = wide.get(f"{m}_negativ") - wide.get(f"{m}_positiv")

    wide["vertraege_gesamt"] = wide["vertraege_negativ"] + wide["vertraege_positiv"]
    if min_vertraege:
        wide = wide[wide["vertraege_gesamt"] >= min_vertraege]

    if sortiere_nach in wide.columns:
        wide = wide.sort_values(sortiere_nach, ascending=False)
    return wide


def produkt_mix_je_sparte(df: pd.DataFrame, kunden: pd.DataFrame,
                          min_vertraege: int = 0) -> pd.DataFrame:
    """Produktmix, hierarchisch nach Sparte gegliedert."""
    out = mix_analyse(df, kunden, [SPARTE, PRODUKT],
                      sortiere_nach="vertraege_gesamt", min_vertraege=min_vertraege)
    return out.sort_index()


# ======================================================================================
# 5) Vertragsebene
# ======================================================================================
def vertragsebene(df: pd.DataFrame, kunden: pd.DataFrame,
                  je_sparte: bool = False) -> pd.DataFrame:
    """Durchschnittliche Vertragskennzahlen je Gruppe (optional zusätzlich je Sparte)."""
    d = df.copy()
    d["wert_flag"] = d[ID].map(kunden["wert_flag"])
    d = d[d["wert_flag"].notna()]

    keys = ["wert_flag", SPARTE] if je_sparte else ["wert_flag"]
    out = d.groupby(keys, observed=True).agg(
        vertraege=(BEITRAG, "size"),
        beitrag_mittel=(BEITRAG, "mean"),
        beitrag_median=(BEITRAG, "median"),
        beitrag_summe=(BEITRAG, "sum"),
        profit_mittel=(PROFIT, "mean"),
        profit_median=(PROFIT, "median"),
        profit_summe=(PROFIT, "sum"),
        anteil_verlustvertraege=(PROFIT, lambda s: float((s < 0).mean())),
    )
    out["marge_gewichtet"] = np.where(out["beitrag_summe"] > 0,
                                      out["profit_summe"] / out["beitrag_summe"], np.nan)
    if CR in d.columns:
        out["cr_mittel_ungewichtet"] = d.groupby(keys, observed=True)[CR].mean()
        out["cr_median"] = d.groupby(keys, observed=True)[CR].median()
    return out


# ======================================================================================
# 6) Zusatzanalysen
# ======================================================================================
def verteilung_vertragsanzahl(kunden: pd.DataFrame) -> pd.DataFrame:
    """Anteil der Kunden je Vertragsanzahl-Klasse."""
    bins = [0, 1, 2, 3, 5, 10, np.inf]
    labels = ["1", "2", "3", "4-5", "6-10", ">10"]
    k = kunden.assign(klasse=pd.cut(kunden["n_vertraege"], bins=bins, labels=labels))
    tab = pd.crosstab(k["klasse"], k["wert_flag"], normalize="columns")
    tab_abs = pd.crosstab(k["klasse"], k["wert_flag"])
    tab.columns = [f"anteil_{c}" for c in tab.columns]
    tab_abs.columns = [f"kunden_{c}" for c in tab_abs.columns]
    out = tab_abs.join(tab)
    if {"anteil_negativ", "anteil_positiv"} <= set(out.columns):
        out["delta"] = out["anteil_negativ"] - out["anteil_positiv"]
    return out


def margen_verteilung(kunden: pd.DataFrame) -> pd.DataFrame:
    """Dezile der Kundenmarge - zeigt, wie scharf die 20-%-Grenze wirklich trennt."""
    q = np.arange(0, 1.01, 0.1)
    out = pd.DataFrame({
        "quantil": q,
        "marge_alle": kunden["marge"].quantile(q).values,
        "marge_negativ": kunden.loc[kunden["wert_flag"] == "negativ", "marge"].quantile(q).values,
        "marge_positiv": kunden.loc[kunden["wert_flag"] == "positiv", "marge"].quantile(q).values,
    })
    return out.set_index("quantil")


def verlusttreiber(df: pd.DataFrame, kunden: pd.DataFrame, top_n: int = 15,
                   ebene: str = PRODUKT) -> pd.DataFrame:
    """Welche Produkte/Sparten erzeugen den Verlust innerhalb der negativen Kunden?"""
    d = df.copy()
    d["wert_flag"] = d[ID].map(kunden["wert_flag"])
    neg = d[d["wert_flag"] == "negativ"]
    if neg.empty:
        return pd.DataFrame()
    out = neg.groupby(ebene, observed=True).agg(
        vertraege=(BEITRAG, "size"),
        beitrag=(BEITRAG, "sum"),
        profit=(PROFIT, "sum"),
    )
    out["marge"] = np.where(out["beitrag"] > 0, out["profit"] / out["beitrag"], np.nan)
    verlust = out.loc[out["profit"] < 0, "profit"].sum()
    out["anteil_am_gesamtverlust"] = np.where(out["profit"] < 0,
                                              out["profit"] / verlust if verlust else np.nan, 0.0)
    return out.sort_values("profit").head(top_n)


def konzentration(kunden: pd.DataFrame) -> pd.DataFrame:
    """Wie stark konzentriert sich der Verlust auf wenige negative Kunden?"""
    neg = kunden[kunden["wert_flag"] == "negativ"].sort_values("profit_sum")
    if neg.empty:
        return pd.DataFrame()
    gesamt = neg["profit_sum"].sum()
    zeilen = []
    for anteil in (0.01, 0.05, 0.10, 0.25, 0.50):
        n = max(1, int(round(len(neg) * anteil)))
        zeilen.append({
            "top_anteil_kunden": anteil,
            "kunden": n,
            "profit_summe": neg["profit_sum"].iloc[:n].sum(),
            "anteil_am_negativen_profit": neg["profit_sum"].iloc[:n].sum() / gesamt if gesamt else np.nan,
            "beitrag_summe": neg["beitrag_sum"].iloc[:n].sum(),
        })
    return pd.DataFrame(zeilen).set_index("top_anteil_kunden")


def sensitivitaet(df: pd.DataFrame,
                  schwellen: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50),
                  ) -> pd.DataFrame:
    """Wie viele Kunden/Beiträge/Profit fallen je nach Schwellenwert ins Negativ-Segment?"""
    zeilen = []
    for s in schwellen:
        k = kunden_aggregat(df, schwelle=s)
        neg = k[k["wert_flag"] == "negativ"]
        zeilen.append({
            "schwelle": s,
            "kunden_negativ": len(neg),
            "anteil_kunden_negativ": len(neg) / len(k) if len(k) else np.nan,
            "anteil_beitrag_negativ": neg["beitrag_sum"].sum() / k["beitrag_sum"].sum()
            if k["beitrag_sum"].sum() else np.nan,
            "profit_negativ": neg["profit_sum"].sum(),
            "profit_positiv": k.loc[k["wert_flag"] == "positiv", "profit_sum"].sum(),
        })
    return pd.DataFrame(zeilen).set_index("schwelle")


def signifikanz(kunden: pd.DataFrame,
                spalten: Sequence[str] = ("n_vertraege", "beitrag_sum",
                                          "beitrag_je_vertrag", "n_sparten")) -> pd.DataFrame:
    """Mann-Whitney-U + Cliff's Delta als Effektstärke (robust, ohne Normalitätsannahme)."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return pd.DataFrame({"hinweis": ["scipy nicht verfügbar"]})

    a = kunden[kunden["wert_flag"] == "negativ"]
    b = kunden[kunden["wert_flag"] == "positiv"]
    zeilen = []
    for c in spalten:
        x, y = a[c].dropna(), b[c].dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        u, p = mannwhitneyu(x, y, alternative="two-sided")
        cliffs = 2 * u / (len(x) * len(y)) - 1  # -1..1
        zeilen.append({"merkmal": c, "median_negativ": x.median(), "median_positiv": y.median(),
                       "p_wert": p, "cliffs_delta": cliffs})
    return pd.DataFrame(zeilen).set_index("merkmal")


# ======================================================================================
# 7) Orchestrierung
# ======================================================================================
def analysiere_kundenwert(
    df: pd.DataFrame,
    schwelle: float = 0.20,
    top_n_produkte: int = 25,
    min_vertraege_produkt: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Führt die komplette Analyse aus und gibt ein Dict von DataFrames zurück."""
    d = bereinige(df)
    kunden = kunden_aggregat(d, schwelle=schwelle)

    erg: Dict[str, pd.DataFrame] = {
        "kunden": kunden,
        "kennzahlen": kennzahlen_vergleich(kunden),
        "sparten_mix": mix_analyse(d, kunden, SPARTE),
        "produkt_mix": mix_analyse(d, kunden, PRODUKT,
                                   min_vertraege=min_vertraege_produkt).head(top_n_produkte),
        "produkt_mix_je_sparte": produkt_mix_je_sparte(d, kunden,
                                                       min_vertraege=min_vertraege_produkt),
        "gesellschaft_mix": mix_analyse(d, kunden, GESELLSCHAFT),
        "vertragsebene": vertragsebene(d, kunden),
        "vertragsebene_je_sparte": vertragsebene(d, kunden, je_sparte=True),
        "verteilung_vertragsanzahl": verteilung_vertragsanzahl(kunden),
        "margen_verteilung": margen_verteilung(kunden),
        "verlusttreiber_produkt": verlusttreiber(d, kunden, ebene=PRODUKT),
        "verlusttreiber_sparte": verlusttreiber(d, kunden, ebene=SPARTE, top_n=50),
        "konzentration": konzentration(kunden),
        "sensitivitaet": sensitivitaet(d),
        "signifikanz": signifikanz(kunden),
    }
    return erg


def drucke_report(erg: Dict[str, pd.DataFrame], top: int = 12) -> None:
    """Kompakte Konsolenausgabe der wichtigsten Ergebnisse."""
    pd.set_option("display.width", 200, "display.max_columns", 60,
                  "display.float_format", lambda x: f"{x:,.3f}")

    k = erg["kunden"]
    print("=" * 100)
    print(f"KUNDENWERT-SEGMENTIERUNG  |  Regel: {k['regel'].iloc[0]}  |  Kunden: {len(k):,}")
    print("=" * 100)

    print("\n--- 1) Kennzahlenvergleich ---")
    print(erg["kennzahlen"])

    print("\n--- 2) Spartenmix (mittlere Zusammensetzung je Kunde) ---")
    sp = erg["sparten_mix"][["anteil_vertraege_negativ", "anteil_vertraege_positiv",
                             "delta_anteil_vertraege", "penetration_negativ",
                             "penetration_positiv", "marge_negativ", "marge_positiv"]]
    print(sp.head(top))

    print("\n--- 3) Produktmix (Top nach Anteil bei negativen Kunden) ---")
    pr = erg["produkt_mix"][["anteil_vertraege_negativ", "anteil_vertraege_positiv",
                             "delta_anteil_vertraege", "beitrag_je_vertrag_negativ",
                             "beitrag_je_vertrag_positiv", "marge_negativ"]]
    print(pr.head(top))

    print("\n--- 4) Vertragsebene ---")
    print(erg["vertragsebene"])

    print("\n--- 5) Verteilung Vertragsanzahl ---")
    print(erg["verteilung_vertragsanzahl"])

    print("\n--- 6) Größte Verlusttreiber (Produkte, negative Kunden) ---")
    print(erg["verlusttreiber_produkt"].head(top))

    print("\n--- 7) Verlustkonzentration ---")
    print(erg["konzentration"])

    print("\n--- 8) Sensitivität der Schwelle ---")
    print(erg["sensitivitaet"])


def exportiere_excel(erg: Dict[str, pd.DataFrame], pfad: str = "kundenwert_analyse.xlsx",
                     mit_kundenliste: bool = False) -> str:
    """Schreibt alle Ergebnistabellen in eine Excel-Datei (Kundenliste optional)."""
    with pd.ExcelWriter(pfad, engine="openpyxl") as writer:
        for name, tab in erg.items():
            if name == "kunden" and not mit_kundenliste:
                continue
            if tab is None or len(tab) == 0:
                continue
            tab.to_excel(writer, sheet_name=name[:31])
    return pfad


# ======================================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_kunden, sparten = 4000, ["KFZ", "Hausrat", "Haftpflicht", "Rechtsschutz", "Wohngebäude"]
    zeilen = []
    for kid in range(n_kunden):
        for _ in range(rng.integers(1, 6)):
            sp = rng.choice(sparten, p=[0.4, 0.2, 0.2, 0.1, 0.1])
            beitrag = float(rng.gamma(3, 120))
            schaden = beitrag * float(rng.normal(0.75, 0.35))
            kosten = beitrag * 0.25
            zeilen.append({
                ID: f"K{kid:05d}", VERTRAG: f"V{len(zeilen):07d}",
                PRODUKT: f"{sp}_{rng.integers(1, 4)}", SPARTE: sp,
                GESELLSCHAFT: rng.choice(["A", "B"]), BEITRAG: beitrag,
                SCHADEN: schaden, KOSTEN: kosten,
                CR: (schaden + kosten) / beitrag, PROFIT: beitrag - schaden - kosten,
            })
    demo = pd.DataFrame(zeilen)
    drucke_report(analysiere_kundenwert(demo))
