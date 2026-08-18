"""
pkw_sichtvergleich.py
=====================

Vergleicht für PKW-Kunden drei Aggregationsperspektiven:

    PKW-Sicht    : nur die PKW-Verträge des Kunden
    KFZ-Sicht    : alle KFZ-Verträge (PKW und KFZ_Rest wie Kraftrad, LKW, Anhänger)
    Gesamtsicht  : der komplette Kompositbestand (KFZ und HUS)

Untersucht wird, wie groß das Delta zwischen den Sichten ist und wo es entsteht.
Hält ein Kunde nur PKW-Verträge, sind alle drei Sichten identisch - das Delta stammt
ausschließlich von Kunden mit zusätzlichem KFZ_Rest- oder HUS-Geschäft. Die Zwischenstufe
KFZ-Sicht zeigt, welcher Teil des Deltas schon durch übrige KFZ-Produkte entsteht.

Verwendung:
    import pandas as pd
    from pkw_sichtvergleich import analysiere_sichten, erstelle_pkw_report, drucke_sichtreport

    erg = analysiere_sichten(df)                 # schwelle=0.20, pkw_produkte=("PKW",)
    drucke_sichtreport(erg)
    erstelle_pkw_report(erg, "PKW_Sichtvergleich.pdf")

Ergebnis-Dict:
    kunden            Kundenebene mit beiden Sichten und Wechselstatus
    eckwerte          Kernzahlen der Grundgesamtheit
    kreuztabelle      Einstufung PKW-Sicht x Gesamtsicht (Kunden, Beitrag, Ergebnis)
    wechselgruppen    Profil der vier Gruppen (Verträge, Beitrag, Marge, HUS-Anteil)
    delta_verteilung  Quantile des Margenunterschieds
    treiber_husanteil Wechselquote nach HUS-Beitragsanteil
    treiber_zweig     HUS-Zweige in den Wechselgruppen
    rangvergleich     Überlappung der schlechtesten Kunden je Sicht
    sensitivitaet     Wechselquote je Schwellenwert
"""

from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle

import matplotlib.pyplot as plt

# Dieses Modul baut auf kundenwert_analyse und kundenwert_report auf.
# Liegen beide als .py-Datei daneben, greift der normale Import. Wird der Code
# stattdessen in ein Notebook eingefügt, stehen die Namen bereits im Namensraum -
# dann wird der Importfehler übersprungen.
_BENOETIGT = ("ID", "VERTRAG", "BEITRAG", "PROFIT", "BEREICH", "BEREICH_GROB", "ZWEIG",
              "ZWEIG_VOLL", "KFZ_REST_PRODUKTE", "bereinige", "ergaenze_ebenen",
              "vertragsaggregat", "C_LINIE", "C_PRIMAER", "MPL_AKZENT", "MPL_GRAU",
              "MPL_NEGATIV", "MPL_POSITIV", "MPL_PRIMAER", "_Doc", "_bild", "_eckwerte",
              "_eur", "_eur_kurz", "_fmt", "_hinweis", "_pct", "_styles", "_tabelle",
              "_zwei_bilder")
try:
    from kundenwert_analyse import (ID, VERTRAG, BEITRAG, PROFIT, BEREICH, BEREICH_GROB,
                                    ZWEIG, ZWEIG_VOLL, KFZ_REST_PRODUKTE, bereinige,
                                    ergaenze_ebenen, vertragsaggregat)
    from kundenwert_report import (C_LINIE, C_PRIMAER, MPL_AKZENT, MPL_GRAU, MPL_NEGATIV,
                                   MPL_POSITIV, MPL_PRIMAER, _Doc, _bild, _eckwerte, _eur,
                                   _eur_kurz, _fmt, _hinweis, _pct, _styles, _tabelle,
                                   _zwei_bilder)
except ModuleNotFoundError as _fehler:
    _fehlend = [n for n in _BENOETIGT if n not in globals()]
    if _fehlend:
        raise ModuleNotFoundError(
            "pkw_sichtvergleich benötigt kundenwert_analyse und kundenwert_report. "
            "Entweder beide als .py-Dateien in dasselbe Verzeichnis legen (bzw. den Pfad "
            "über sys.path.append ergänzen), oder im Notebook zuerst die Zellen mit "
            "beiden Modulen ausführen. Noch nicht definiert: "
            + ", ".join(_fehlend)) from _fehler

# Reihenfolge der Wechselgruppen - überall gleich verwendet
GRUPPEN = ["in beiden Sichten negativ", "nur in PKW-Sicht negativ",
           "nur in Gesamtsicht negativ", "in beiden Sichten positiv"]
WECHSLER = ["nur in PKW-Sicht negativ", "nur in Gesamtsicht negativ"]
KURZ = {"in beiden Sichten negativ": "beide negativ",
        "nur in PKW-Sicht negativ": "nur PKW negativ",
        "nur in Gesamtsicht negativ": "nur Gesamt negativ",
        "in beiden Sichten positiv": "beide positiv"}


def _skala(werte: Sequence[float]):
    """Einheitliche Skalierung für Geldspalten: gibt (Teiler, Einheitstext) zurück."""
    m = max((abs(float(w)) for w in werte if w is not None and np.isfinite(w)), default=0.0)
    if m >= 1e10:
        return 1e9, "Mrd. EUR"
    if m >= 1e7:
        return 1e6, "Mio. EUR"
    return 1e3, "Tsd. EUR"


# ======================================================================================
# Analyse
# ======================================================================================
def _negativ(profit: pd.Series, beitrag: pd.Series, schwelle: float) -> pd.Series:
    """Segmentregel: Ergebnis negativ UND betragsmäßig mind. `schwelle` vom Beitrag."""
    return ((profit < 0) & (profit.abs() >= schwelle * beitrag)).fillna(False)


def kunden_drei_sichten(df: pd.DataFrame, schwelle: float = 0.20,
                        pkw_produkte: Sequence[str] = ("PKW",),
                        kfz_rest_produkte: Sequence[str] = KFZ_REST_PRODUKTE
                        ) -> pd.DataFrame:
    """
    Kundenebene für alle Kunden mit mindestens einem PKW-Vertrag, jeweils mit
    PKW-Sicht, KFZ-Sicht und Gesamtsicht sowie dem daraus folgenden Wechselstatus.
    """
    d = ergaenze_ebenen(bereinige(df), pkw_produkte=pkw_produkte,
                        kfz_rest_produkte=kfz_rest_produkte)
    v = vertragsaggregat(d)

    gesamt = v.groupby(ID, observed=True).agg(
        n_vertraege_gesamt=(VERTRAG, "nunique"),
        beitrag_gesamt=("beitrag", "sum"),
        profit_gesamt=("profit", "sum"))
    je_bereich = v.pivot_table(index=ID, columns=BEREICH,
                               values=["beitrag", "profit"], aggfunc="sum",
                               observed=True).fillna(0.0)
    anzahl = v.pivot_table(index=ID, columns=BEREICH, values=VERTRAG,
                           aggfunc="nunique", observed=True).fillna(0)

    k = gesamt.copy()
    for b, sp in (("PKW", "pkw"), ("KFZ_Rest", "kfz_rest"), ("HUS", "hus")):
        k[f"beitrag_{sp}"] = (je_bereich[("beitrag", b)]
                              if ("beitrag", b) in je_bereich.columns else 0.0)
        k[f"profit_{sp}"] = (je_bereich[("profit", b)]
                             if ("profit", b) in je_bereich.columns else 0.0)
        k[f"n_vertraege_{sp}"] = anzahl[b] if b in anzahl.columns else 0

    # KFZ-Sicht = PKW + KFZ_Rest
    k["beitrag_kfz"] = k["beitrag_pkw"] + k["beitrag_kfz_rest"]
    k["profit_kfz"] = k["profit_pkw"] + k["profit_kfz_rest"]
    k["n_vertraege_kfz"] = k["n_vertraege_pkw"] + k["n_vertraege_kfz_rest"]

    # Grundgesamtheit: Kunden mit mindestens einem PKW-Vertrag
    k = k[k["n_vertraege_pkw"] > 0].copy()
    k["hat_kfz_rest"] = k["n_vertraege_kfz_rest"] > 0
    k["hat_hus"] = k["n_vertraege_hus"] > 0
    k["hat_zusatz"] = k["hat_kfz_rest"] | k["hat_hus"]
    k["bestandsprofil"] = np.select(
        [k["hat_kfz_rest"] & k["hat_hus"], k["hat_kfz_rest"], k["hat_hus"]],
        ["PKW + KFZ_Rest + HUS", "PKW + KFZ_Rest", "PKW + HUS"], default="nur PKW")

    k["kfz_rest_beitragsanteil"] = np.where(k["beitrag_gesamt"] > 0,
                                            k["beitrag_kfz_rest"] / k["beitrag_gesamt"], np.nan)
    k["hus_beitragsanteil"] = np.where(k["beitrag_gesamt"] > 0,
                                       k["beitrag_hus"] / k["beitrag_gesamt"], np.nan)
    k["zusatz_beitragsanteil"] = np.where(k["beitrag_gesamt"] > 0,
                                          1 - k["beitrag_pkw"] / k["beitrag_gesamt"], np.nan)

    for sicht in ("pkw", "kfz", "gesamt"):
        k[f"marge_{sicht}"] = np.where(k[f"beitrag_{sicht}"] > 0,
                                       k[f"profit_{sicht}"] / k[f"beitrag_{sicht}"], np.nan)
        k[f"flag_{sicht}"] = np.where(
            _negativ(k[f"profit_{sicht}"], k[f"beitrag_{sicht}"], schwelle),
            "negativ", "positiv")

    k["delta_marge_pkw_kfz"] = k["marge_pkw"] - k["marge_kfz"]
    k["delta_marge_kfz_gesamt"] = k["marge_kfz"] - k["marge_gesamt"]
    k["delta_marge"] = k["marge_pkw"] - k["marge_gesamt"]

    neg_pkw = k["flag_pkw"] == "negativ"
    neg_ges = k["flag_gesamt"] == "negativ"
    k["gruppe"] = np.select(
        [neg_pkw & neg_ges, neg_pkw & ~neg_ges, ~neg_pkw & neg_ges],
        GRUPPEN[:3], default=GRUPPEN[3])
    k["wechsel"] = k["gruppe"].isin(WECHSLER)
    k["wechsel_pkw_kfz"] = k["flag_pkw"] != k["flag_kfz"]
    k["wechsel_kfz_gesamt"] = k["flag_kfz"] != k["flag_gesamt"]
    return k


# Rückwärtskompatibler Name
kunden_beide_sichten = kunden_drei_sichten


def eckwerte(k: pd.DataFrame) -> pd.DataFrame:
    """Kernzahlen der Grundgesamtheit und der Deltas zwischen den drei Sichten."""
    zusatz = k[k["hat_zusatz"]]
    zeilen = {
        "PKW-Kunden gesamt": len(k),
        "davon nur PKW": int((~k["hat_zusatz"]).sum()),
        "davon mit KFZ_Rest": int(k["hat_kfz_rest"].sum()),
        "davon mit HUS": int(k["hat_hus"].sum()),
        "Anteil mit Zusatzgeschäft": k["hat_zusatz"].mean(),
        "Kunden mit abweichender Einstufung (PKW vs. gesamt)": int(k["wechsel"].sum()),
        "Anteil an allen PKW-Kunden": k["wechsel"].mean(),
        "Anteil an Kunden mit Zusatzgeschäft": (zusatz["wechsel"].mean()
                                                if len(zusatz) else np.nan),
        "davon bereits durch KFZ_Rest (PKW vs. KFZ)": k["wechsel_pkw_kfz"].mean(),
        "davon erst durch HUS (KFZ vs. gesamt)": k["wechsel_kfz_gesamt"].mean(),
        "Negativ-Quote in PKW-Sicht": (k["flag_pkw"] == "negativ").mean(),
        "Negativ-Quote in KFZ-Sicht": (k["flag_kfz"] == "negativ").mean(),
        "Negativ-Quote in Gesamtsicht": (k["flag_gesamt"] == "negativ").mean(),
        "Beitrag PKW": k["beitrag_pkw"].sum(),
        "Beitrag KFZ": k["beitrag_kfz"].sum(),
        "Beitrag gesamt": k["beitrag_gesamt"].sum(),
        "Ergebnis PKW": k["profit_pkw"].sum(),
        "Ergebnis KFZ": k["profit_kfz"].sum(),
        "Ergebnis gesamt": k["profit_gesamt"].sum(),
    }
    return pd.DataFrame({"wert": zeilen})


def sichtstufen(k: pd.DataFrame) -> pd.DataFrame:
    """Die drei Sichten nebeneinander - Volumen, Ergebnis, Negativ-Quote."""
    zeilen = []
    for sicht, name, basis in (("pkw", "PKW-Sicht", "nur PKW-Verträge"),
                               ("kfz", "KFZ-Sicht", "PKW und KFZ_Rest"),
                               ("gesamt", "Gesamtsicht", "alle Verträge")):
        beitrag, profit = k[f"beitrag_{sicht}"].sum(), k[f"profit_{sicht}"].sum()
        zeilen.append({
            "sicht": name,
            "basis": basis,
            "vertraege": k[f"n_vertraege_{sicht}"].sum(),
            "beitrag": beitrag,
            "profit": profit,
            "marge": profit / beitrag if beitrag else np.nan,
            "negativ_quote": (k[f"flag_{sicht}"] == "negativ").mean(),
        })
    return pd.DataFrame(zeilen).set_index("sicht")


def stufenwechsel(k: pd.DataFrame) -> pd.DataFrame:
    """Wo entsteht der Wechsel: schon beim Schritt zu KFZ oder erst durch HUS?"""
    zeilen = {
        "PKW-Sicht zu KFZ-Sicht": {
            "kunden": int(k["wechsel_pkw_kfz"].sum()),
            "anteil": k["wechsel_pkw_kfz"].mean(),
            "betroffene_kunden": int(k["hat_kfz_rest"].sum()),
            "quote_in_betroffenen": (k.loc[k["hat_kfz_rest"], "wechsel_pkw_kfz"].mean()
                                     if k["hat_kfz_rest"].any() else np.nan)},
        "KFZ-Sicht zu Gesamtsicht": {
            "kunden": int(k["wechsel_kfz_gesamt"].sum()),
            "anteil": k["wechsel_kfz_gesamt"].mean(),
            "betroffene_kunden": int(k["hat_hus"].sum()),
            "quote_in_betroffenen": (k.loc[k["hat_hus"], "wechsel_kfz_gesamt"].mean()
                                     if k["hat_hus"].any() else np.nan)},
        "PKW-Sicht zu Gesamtsicht": {
            "kunden": int(k["wechsel"].sum()),
            "anteil": k["wechsel"].mean(),
            "betroffene_kunden": int(k["hat_zusatz"].sum()),
            "quote_in_betroffenen": (k.loc[k["hat_zusatz"], "wechsel"].mean()
                                     if k["hat_zusatz"].any() else np.nan)},
    }
    return pd.DataFrame(zeilen).T


def kreuztabelle(k: pd.DataFrame) -> pd.DataFrame:
    """Einstufung PKW-Sicht gegen Gesamtsicht, mit Volumen je Feld."""
    g = k.groupby(["flag_pkw", "flag_gesamt"], observed=True).agg(
        kunden=("beitrag_pkw", "size"),
        beitrag_pkw=("beitrag_pkw", "sum"),
        beitrag_gesamt=("beitrag_gesamt", "sum"),
        profit_pkw=("profit_pkw", "sum"),
        profit_gesamt=("profit_gesamt", "sum"))
    g["anteil_kunden"] = g["kunden"] / len(k)
    return g


def wechselgruppen(k: pd.DataFrame) -> pd.DataFrame:
    """Profil der vier Gruppen - wodurch unterscheiden sich die Wechsler?"""
    g = k.groupby("gruppe", observed=True).agg(
        kunden=("beitrag_pkw", "size"),
        anteil_mit_zusatz=("hat_zusatz", "mean"),
        vertraege_pkw=("n_vertraege_pkw", "mean"),
        vertraege_kfz_rest=("n_vertraege_kfz_rest", "mean"),
        vertraege_hus=("n_vertraege_hus", "mean"),
        beitrag_pkw=("beitrag_pkw", "mean"),
        zusatz_beitragsanteil=("zusatz_beitragsanteil", "mean"),
        profit_pkw_summe=("profit_pkw", "sum"),
        profit_kfz_rest_summe=("profit_kfz_rest", "sum"),
        profit_hus_summe=("profit_hus", "sum"),
        marge_pkw=("marge_pkw", "median"),
        marge_kfz=("marge_kfz", "median"),
        marge_gesamt=("marge_gesamt", "median"),
    ).reindex(GRUPPEN)
    g["anteil_kunden"] = g["kunden"] / len(k)
    g["profit_kfz_rest_je_kunde"] = g["profit_kfz_rest_summe"] / g["kunden"]
    g["profit_hus_je_kunde"] = g["profit_hus_summe"] / g["kunden"]
    g["profit_zusatz_je_kunde"] = ((g["profit_kfz_rest_summe"] + g["profit_hus_summe"])
                                   / g["kunden"])
    return g


def delta_verteilung(k: pd.DataFrame) -> pd.DataFrame:
    """Quantile des Margenunterschieds - insgesamt und nur für Verbundkunden."""
    q = np.arange(0, 1.01, 0.1)
    verbund = k[k["hat_hus"]]
    return pd.DataFrame({
        "quantil": q,
        "delta_marge_alle": k["delta_marge"].quantile(q).values,
        "delta_marge_verbund": (verbund["delta_marge"].quantile(q).values
                                if len(verbund) else np.nan),
    }).set_index("quantil")


def treiber_husanteil(k: pd.DataFrame) -> pd.DataFrame:
    """Wechselquote nach Gewicht des HUS-Geschäfts am Gesamtbeitrag."""
    verbund = k[k["hat_zusatz"]].copy()
    if verbund.empty:
        return pd.DataFrame()
    kanten = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    labels = ["bis 10 %", "10-25 %", "25-50 %", "50-75 %", "über 75 %"]
    verbund["klasse"] = pd.cut(verbund["zusatz_beitragsanteil"], bins=kanten,
                               labels=labels, include_lowest=True)
    g = verbund.groupby("klasse", observed=True).agg(
        kunden=("wechsel", "size"),
        wechselquote=("wechsel", "mean"),
        delta_marge_median=("delta_marge", "median"),
        beitrag_gesamt=("beitrag_gesamt", "sum"))
    return g[g["kunden"] > 0]


def treiber_zweig(df: pd.DataFrame, k: pd.DataFrame,
                  pkw_produkte: Sequence[str] = ("PKW",),
                  kfz_rest_produkte: Sequence[str] = KFZ_REST_PRODUKTE) -> pd.DataFrame:
    """
    Welche Zweige außerhalb des PKW halten die Wechsler? Verglichen wird die
    Durchdringung in den Wechselgruppen mit der aller Kunden mit Zusatzgeschäft.
    """
    d = ergaenze_ebenen(bereinige(df), pkw_produkte=pkw_produkte,
                        kfz_rest_produkte=kfz_rest_produkte, melde_zuordnung=False)
    hus = d[(d[BEREICH] != "PKW") & (d[ID].isin(k.index[k["hat_zusatz"]]))]
    if hus.empty:
        return pd.DataFrame()
    hus = hus.assign(gruppe=hus[ID].map(k["gruppe"]))

    n_gruppe = k[k["hat_zusatz"]].groupby("gruppe", observed=True).size()
    g = hus.groupby([ZWEIG_VOLL, "gruppe"], observed=True).agg(
        kunden=(ID, "nunique"), beitrag=(BEITRAG, "sum"), profit=(PROFIT, "sum")
    ).reset_index()
    g["durchdringung"] = g["kunden"] / g["gruppe"].map(n_gruppe)

    wide = g.pivot_table(index=ZWEIG_VOLL, columns="gruppe",
                         values=["durchdringung", "kunden", "profit"], observed=True)
    wide.columns = [f"{m}__{f}" for m, f in wide.columns]

    basis = hus.groupby(ZWEIG_VOLL, observed=True)[ID].nunique() / len(k[k["hat_zusatz"]])
    wide["durchdringung_alle_zusatz"] = basis
    for gruppe in WECHSLER:
        sp = f"durchdringung__{gruppe}"
        if sp in wide.columns:
            wide[f"delta__{gruppe}"] = wide[sp] - wide["durchdringung_alle_zusatz"]
    return wide.sort_values("durchdringung_alle_zusatz", ascending=False)


def wertvergleich(k: pd.DataFrame) -> pd.DataFrame:
    """
    Der Kundenwert selbst - unabhängig von jeder Schwelle. Je Sicht das aggregierte
    Ergebnis, das Ergebnis je Kunde und die Marge.
    """
    zeilen = []
    for sicht, name in (("pkw", "PKW-Sicht"), ("kfz", "KFZ-Sicht"),
                        ("gesamt", "Gesamtsicht")):
        beitrag, profit = k[f"beitrag_{sicht}"].sum(), k[f"profit_{sicht}"].sum()
        zeilen.append({
            "sicht": name,
            "beitrag": beitrag,
            "profit": profit,
            "marge": profit / beitrag if beitrag else np.nan,
            "profit_je_kunde_mittel": k[f"profit_{sicht}"].mean(),
            "profit_je_kunde_median": k[f"profit_{sicht}"].median(),
            "marge_je_kunde_median": k[f"marge_{sicht}"].median(),
            "kunden_mit_negativem_wert": float((k[f"profit_{sicht}"] < 0).mean()),
        })
    return pd.DataFrame(zeilen).set_index("sicht")


def wertdelta(k: pd.DataFrame) -> pd.DataFrame:
    """
    Verteilung der Wertunterschiede je Kunde - in Euro und in Margenpunkten.
    Positiv heißt: In der PKW-Sicht steht der Kunde besser da als in der Gesamtsicht.
    """
    q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    paare = {
        "PKW minus KFZ": (k["profit_pkw"] - k["profit_kfz"], k["delta_marge_pkw_kfz"]),
        "KFZ minus gesamt": (k["profit_kfz"] - k["profit_gesamt"],
                             k["delta_marge_kfz_gesamt"]),
        "PKW minus gesamt": (k["profit_pkw"] - k["profit_gesamt"], k["delta_marge"]),
    }
    zeilen = {}
    for name, (eur, marge) in paare.items():
        werte = {"mittel_eur": eur.mean(), "median_eur": eur.median(),
                 "median_marge": marge.median(),
                 "anteil_ueber_100_eur": float((eur.abs() > 100).mean()),
                 "anteil_ueber_10_pp": float((marge.abs() > 0.10).mean())}
        for qq in q:
            werte[f"eur_p{int(qq * 100)}"] = eur.quantile(qq)
        zeilen[name] = werte
    return pd.DataFrame(zeilen).T


def dezil_matrix(k: pd.DataFrame, sicht_a: str = "pkw", sicht_b: str = "gesamt",
                 groesse: str = "marge", n: int = 10) -> pd.DataFrame:
    """
    Wanderung zwischen den Dezilen: Wo landet ein Kunde, der in Sicht A im Dezil X
    liegt, wenn man nach Sicht B sortiert? Dezil 1 = schlechteste Kunden.
    Werte sind Zeilenanteile.
    """
    d = k[[f"{groesse}_{sicht_a}", f"{groesse}_{sicht_b}"]].dropna()
    if d.empty:
        return pd.DataFrame()
    labels = list(range(1, n + 1))
    da = pd.qcut(d[f"{groesse}_{sicht_a}"].rank(method="first"), n, labels=labels)
    db = pd.qcut(d[f"{groesse}_{sicht_b}"].rank(method="first"), n, labels=labels)
    m = pd.crosstab(da, db, normalize="index")
    m.index.name = f"Dezil {sicht_a}"
    m.columns.name = f"Dezil {sicht_b}"
    return m


def rangkennzahlen(k: pd.DataFrame, groesse: str = "marge", n: int = 10) -> pd.DataFrame:
    """
    Wie stabil ist die Reihenfolge zwischen den Sichten? Rangkorrelation,
    durchschnittliche Perzentilverschiebung und Dezilwanderung je Sichtpaar.
    """
    zeilen = {}
    for a, b, name in (("pkw", "kfz", "PKW gegen KFZ"),
                       ("kfz", "gesamt", "KFZ gegen gesamt"),
                       ("pkw", "gesamt", "PKW gegen gesamt")):
        d = k[[f"{groesse}_{a}", f"{groesse}_{b}"]].dropna()
        if d.empty:
            continue
        ra = d[f"{groesse}_{a}"].rank(pct=True, method="first")
        rb = d[f"{groesse}_{b}"].rank(pct=True, method="first")
        da = pd.qcut(ra, n, labels=range(1, n + 1)).astype(int)
        db = pd.qcut(rb, n, labels=range(1, n + 1)).astype(int)
        sprung = (da - db).abs()
        werte = {
            "kunden": len(d),
            "spearman": float(ra.corr(rb, method="spearman")),
            "perzentilverschiebung_mittel": float((ra - rb).abs().mean()),
            "perzentilverschiebung_p90": float((ra - rb).abs().quantile(0.90)),
            "gleiches_dezil": float((sprung == 0).mean()),
            "sprung_1_dezil": float((sprung == 1).mean()),
            "sprung_2_plus": float((sprung >= 2).mean()),
            "sprung_3_plus": float((sprung >= 3).mean()),
        }
        try:
            from scipy.stats import kendalltau
            werte["kendall"] = float(kendalltau(ra, rb).statistic)
        except ImportError:
            pass
        zeilen[name] = werte
    return pd.DataFrame(zeilen).T


def auswahlueberlappung(k: pd.DataFrame, groesse: str = "marge",
                        anteile: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
                        ) -> pd.DataFrame:
    """
    Überlappung der Auswahllisten: Wie viele Kunden stehen auf beiden Listen, wenn
    man die schlechtesten X Prozent nach der jeweiligen Sicht auswählt?
    """
    d = k[[f"{groesse}_pkw", f"{groesse}_kfz", f"{groesse}_gesamt",
           "profit_gesamt", "beitrag_gesamt"]].dropna(
        subset=[f"{groesse}_pkw", f"{groesse}_kfz", f"{groesse}_gesamt"])
    if d.empty:
        return pd.DataFrame()
    zeilen = []
    for a in anteile:
        n = max(1, int(round(len(d) * a)))
        listen = {s: set(d[f"{groesse}_{s}"].nsmallest(n).index)
                  for s in ("pkw", "kfz", "gesamt")}
        nur_pkw = listen["pkw"] - listen["gesamt"]
        zeilen.append({
            "auswahl": a,
            "kunden_je_liste": n,
            "pkw_vs_kfz": len(listen["pkw"] & listen["kfz"]) / n,
            "kfz_vs_gesamt": len(listen["kfz"] & listen["gesamt"]) / n,
            "pkw_vs_gesamt": len(listen["pkw"] & listen["gesamt"]) / n,
            "nur_in_pkw_liste": len(nur_pkw),
            "ergebnis_nur_pkw_liste": d.loc[list(nur_pkw), "profit_gesamt"].sum(),
            "beitrag_nur_pkw_liste": d.loc[list(nur_pkw), "beitrag_gesamt"].sum(),
        })
    return pd.DataFrame(zeilen).set_index("auswahl")


def rangvergleich(k: pd.DataFrame, anteile: Sequence[float] = (0.05, 0.10, 0.20)) -> pd.DataFrame:
    """
    Für Priorisierungen: Wählt man die schlechtesten X % nach PKW-Sicht oder nach
    Gesamtsicht aus - wie stark überlappen die beiden Listen?
    """
    d = k[k["marge_pkw"].notna() & k["marge_gesamt"].notna() & k["marge_kfz"].notna()]
    if d.empty:
        return pd.DataFrame()
    zeilen = []
    for a in anteile:
        n = max(1, int(round(len(d) * a)))
        liste_pkw = set(d["marge_pkw"].nsmallest(n).index)
        liste_kfz = set(d["marge_kfz"].nsmallest(n).index)
        liste_ges = set(d["marge_gesamt"].nsmallest(n).index)
        zeilen.append({
            "auswahl": a,
            "kunden_je_liste": n,
            "pkw_vs_kfz": len(liste_pkw & liste_kfz) / n,
            "kfz_vs_gesamt": len(liste_kfz & liste_ges) / n,
            "pkw_vs_gesamt": len(liste_pkw & liste_ges) / n,
            "nur_in_pkw_liste": n - len(liste_pkw & liste_ges),
        })
    out = pd.DataFrame(zeilen).set_index("auswahl")
    out.attrs["spearman"] = float(d["marge_pkw"].corr(d["marge_gesamt"], method="spearman"))
    out.attrs["spearman_pkw_kfz"] = float(d["marge_pkw"].corr(d["marge_kfz"],
                                                              method="spearman"))
    return out


def sensitivitaet_sichten(df: pd.DataFrame,
                          schwellen: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50),
                          pkw_produkte: Sequence[str] = ("PKW",),
                          kfz_rest_produkte: Sequence[str] = KFZ_REST_PRODUKTE
                          ) -> pd.DataFrame:
    """Wechselquote und Segmentgrößen je Schwellenwert."""
    zeilen = []
    for s in schwellen:
        k = kunden_drei_sichten(df, schwelle=s, pkw_produkte=pkw_produkte,
                                kfz_rest_produkte=kfz_rest_produkte)
        zeilen.append({
            "schwelle": s,
            "negativ_pkw_sicht": (k["flag_pkw"] == "negativ").mean(),
            "negativ_kfz_sicht": (k["flag_kfz"] == "negativ").mean(),
            "negativ_gesamtsicht": (k["flag_gesamt"] == "negativ").mean(),
            "wechselquote": k["wechsel"].mean(),
            "nur_pkw_negativ": (k["gruppe"] == WECHSLER[0]).mean(),
            "nur_gesamt_negativ": (k["gruppe"] == WECHSLER[1]).mean(),
        })
    return pd.DataFrame(zeilen).set_index("schwelle")


def analysiere_sichten(df: pd.DataFrame, schwelle: float = 0.20,
                       pkw_produkte: Sequence[str] = ("PKW",),
                       kfz_rest_produkte: Sequence[str] = KFZ_REST_PRODUKTE
                       ) -> Dict[str, pd.DataFrame]:
    """Führt den kompletten Sichtvergleich aus."""
    k = kunden_drei_sichten(df, schwelle=schwelle, pkw_produkte=pkw_produkte,
                            kfz_rest_produkte=kfz_rest_produkte)
    erg = {
        "kunden": k,
        "eckwerte": eckwerte(k),
        # Werte und Reihenfolge - unabhängig von der Schwelle
        "wertvergleich": wertvergleich(k),
        "wertdelta": wertdelta(k),
        "rangkennzahlen": rangkennzahlen(k),
        "auswahlueberlappung": auswahlueberlappung(k),
        "dezil_matrix_pkw_gesamt": dezil_matrix(k, "pkw", "gesamt"),
        "dezil_matrix_pkw_kfz": dezil_matrix(k, "pkw", "kfz"),
        # Binäre Einstufung (nachrangig)
        "sichtstufen": sichtstufen(k),
        "stufenwechsel": stufenwechsel(k),
        "bestandsprofile": (k.groupby("bestandsprofil", observed=True)
                            .agg(kunden=("beitrag_pkw", "size"),
                                 wechselquote=("wechsel", "mean"),
                                 delta_marge_median=("delta_marge", "median"),
                                 beitrag_gesamt=("beitrag_gesamt", "sum"))
                            .sort_values("kunden", ascending=False)),
        "kreuztabelle": kreuztabelle(k),
        "wechselgruppen": wechselgruppen(k),
        "delta_verteilung": delta_verteilung(k),
        "treiber_husanteil": treiber_husanteil(k),
        "treiber_zweig": treiber_zweig(df, k, pkw_produkte=pkw_produkte,
                                       kfz_rest_produkte=kfz_rest_produkte),
        "rangvergleich": rangvergleich(k),
        "sensitivitaet": sensitivitaet_sichten(df, pkw_produkte=pkw_produkte,
                                               kfz_rest_produkte=kfz_rest_produkte),
    }
    erg["kunden"].attrs["schwelle"] = schwelle
    return erg


def drucke_sichtreport(erg: Dict[str, pd.DataFrame]) -> None:
    """Kompakte Konsolenausgabe."""
    pd.set_option("display.width", 200, "display.max_columns", 60,
                  "display.float_format", lambda x: f"{x:,.3f}")
    print("=" * 100)
    print("PKW-SICHT GEGEN GESAMTSICHT")
    print("=" * 100)
    print(erg["eckwerte"])
    print("\n--- Kreuztabelle der Einstufungen ---")
    print(erg["kreuztabelle"])
    print("\n--- Profil der Wechselgruppen ---")
    print(erg["wechselgruppen"])
    print("\n--- Wechselquote nach HUS-Beitragsanteil ---")
    print(erg["treiber_husanteil"])
    print("\n--- Überlappung der Auswahllisten ---")
    print(erg["rangvergleich"])
    print("\n--- Sensitivität ---")
    print(erg["sensitivitaet"])


# ======================================================================================
# Grafiken
# ======================================================================================
def _chart_dezilmatrix(m: pd.DataFrame, sicht_a: str = "PKW-Sicht",
                       sicht_b: str = "Gesamtsicht") -> plt.Figure:
    """Dezilwanderung als Matrix - die Diagonale zeigt Kunden, die ihren Rang halten."""
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    werte = m.values * 100
    ax.imshow(werte, cmap="Blues", vmin=0, vmax=max(werte.max(), 1))
    for i in range(werte.shape[0]):
        for j in range(werte.shape[1]):
            v = werte[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > werte.max() * 0.55 else MPL_PRIMAER)
    ax.set_xticks(range(len(m.columns)), [str(c) for c in m.columns], fontsize=7)
    ax.set_yticks(range(len(m.index)), [str(i) for i in m.index], fontsize=7)
    ax.set_xlabel(f"Dezil in der {sicht_b}", fontsize=8)
    ax.set_ylabel(f"Dezil in der {sicht_a}", fontsize=8)
    ax.set_title("Zeilenanteile in Prozent - Dezil 1 = schlechteste Kunden",
                 loc="left", fontsize=8, color=MPL_GRAU, pad=6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def _chart_ueberlappung(au: pd.DataFrame) -> plt.Figure:
    """Überlappung der Auswahllisten über verschiedene Auswahlgrößen."""
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(au.index, au["pkw_vs_kfz"], marker="o", color=MPL_GRAU, lw=1.8,
            label="PKW gegen KFZ")
    ax.plot(au.index, au["pkw_vs_gesamt"], marker="s", color=MPL_NEGATIV, lw=1.8,
            label="PKW gegen gesamt")
    ax.plot(au.index, au["kfz_vs_gesamt"], marker="^", color=MPL_AKZENT, lw=1.8, ls="--",
            label="KFZ gegen gesamt")
    ax.set_xlabel("Größe der Auswahl (schlechteste ... Prozent)", fontsize=8)
    ax.set_ylabel("Anteil gemeinsamer Kunden", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_wertdelta(k: pd.DataFrame) -> plt.Figure:
    """Wertunterschied je Kunde in Euro - wie weit liegen die Sichten auseinander?"""
    d = (k["profit_pkw"] - k["profit_gesamt"]).dropna()
    grenze = float(d.abs().quantile(0.98)) or 1.0
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(d.clip(-grenze, grenze), bins=61, color=MPL_AKZENT)
    ax.axvline(0, color=MPL_PRIMAER, lw=1.1)
    ax.set_xlabel("Ergebnis in PKW-Sicht minus Ergebnis gesamt (EUR je Kunde)", fontsize=8)
    ax.set_ylabel("Kunden", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _fmt(v, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_werte_je_sicht(wv: pd.DataFrame) -> plt.Figure:
    """Ergebnis je Kunde und Marge über die drei Sichten."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 2.5))
    x = np.arange(len(wv))
    for ax, spalte, titel, fmt in (
            (ax1, "profit_je_kunde_median", "Ergebnis je Kunde (Median)",
             lambda v: _eur(v)),
            (ax2, "marge_je_kunde_median", "Marge je Kunde (Median)",
             lambda v: _pct(v, 1))):
        farben = [MPL_NEGATIV if v < 0 else MPL_POSITIV for v in wv[spalte]]
        b = ax.bar(x, wv[spalte], color=farben, width=0.55)
        ax.bar_label(b, labels=[fmt(v) for v in wv[spalte]], fontsize=8, padding=3,
                     color=MPL_PRIMAER, fontweight="bold")
        ax.set_xticks(x, [str(i) for i in wv.index], fontsize=8.5)
        ax.set_title(titel, loc="left", fontsize=9.5, fontweight="bold",
                     color=MPL_PRIMAER, pad=8)
        ax.axhline(0, color=MPL_GRAU, lw=0.8)
        ax.set_yticks([])
        ax.margins(y=0.30)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    return fig


def _chart_streuung_sichten(k: pd.DataFrame, schwelle: float,
                            max_punkte: int = 6000) -> plt.Figure:
    """Marge in PKW-Sicht gegen Marge in Gesamtsicht, Quadranten durch die Grenze."""
    d = k[k["marge_pkw"].notna() & k["marge_gesamt"].notna()]
    if len(d) > max_punkte:
        d = d.sample(max_punkte, random_state=0)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    unveraendert = d[~d["wechsel"]]
    ax.scatter(unveraendert["marge_gesamt"].clip(-1.5, 1), unveraendert["marge_pkw"].clip(-1.5, 1),
               s=4, alpha=0.25, color=MPL_GRAU, linewidths=0, label="gleiche Einstufung")
    for gruppe, farbe in ((WECHSLER[0], MPL_NEGATIV), (WECHSLER[1], MPL_AKZENT)):
        t = d[d["gruppe"] == gruppe]
        ax.scatter(t["marge_gesamt"].clip(-1.5, 1), t["marge_pkw"].clip(-1.5, 1), s=7,
                   alpha=0.6, color=farbe, linewidths=0, label=gruppe)
    ax.axhline(-schwelle, color=C_PRIMAER.hexval()[2:] and MPL_PRIMAER, ls="--", lw=1)
    ax.axvline(-schwelle, color=MPL_PRIMAER, ls="--", lw=1)
    ax.plot([-1.5, 1], [-1.5, 1], color=MPL_GRAU, lw=0.8, ls=":")
    ax.set_xlabel("Marge in Gesamtsicht", fontsize=8)
    ax.set_ylabel("Marge in PKW-Sicht", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.set_xlim(-1.5, 1)
    ax.set_ylim(-1.5, 1)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=2.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_sichtstufen(ss: pd.DataFrame) -> plt.Figure:
    """Negativ-Quote und Marge über die drei Sichten - wo verschiebt sich was?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 2.5))
    x = np.arange(len(ss))
    for ax, spalte, titel, farbe in ((ax1, "negativ_quote", "Negativ-Quote", MPL_NEGATIV),
                                     (ax2, "marge", "Marge des Bestands", MPL_AKZENT)):
        b = ax.bar(x, ss[spalte], color=farbe, width=0.55)
        ax.bar_label(b, labels=[_pct(v, 1) for v in ss[spalte]], fontsize=8,
                     padding=3, color=MPL_PRIMAER, fontweight="bold")
        ax.set_xticks(x, [str(i) for i in ss.index], fontsize=8.5)
        ax.set_title(titel, loc="left", fontsize=9.5, fontweight="bold",
                     color=MPL_PRIMAER, pad=8)
        ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
        ax.axhline(0, color=MPL_GRAU, lw=0.8)
        ax.margins(y=0.22)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_gruppen(k: pd.DataFrame) -> plt.Figure:
    """Wie viele Kunden fallen in welche der vier Gruppen?"""
    g = k["gruppe"].value_counts().reindex(GRUPPEN).fillna(0)
    anteile = g / g.sum()
    farben = [MPL_NEGATIV, MPL_NEGATIV, MPL_AKZENT, MPL_POSITIV]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    y = np.arange(len(g))[::-1]
    ax.barh(y, anteile.values, color=farben, height=0.55)
    for yi, (name, anteil) in zip(y, anteile.items()):
        ax.text(anteil + 0.012, yi, f"{_pct(anteil, 1)}  ({_fmt(g[name])} Kunden)",
                va="center", fontsize=7.5, color=MPL_PRIMAER)
    ax.set_yticks(y, [n.replace(" Sichten", "\nSichten") for n in g.index], fontsize=8)
    ax.set_xlim(0, min(1.0, anteile.max() * 1.7))
    ax.set_xlabel("Anteil der PKW-Kunden", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_husanteil(th: pd.DataFrame) -> plt.Figure:
    """Wechselquote nach Gewicht des HUS-Geschäfts."""
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    x = np.arange(len(th))
    ax.bar(x, th["wechselquote"], color=MPL_NEGATIV, width=0.6)
    for xi, (_, r) in zip(x, th.iterrows()):
        ax.text(xi, r["wechselquote"] + 0.006, _pct(r["wechselquote"], 0), ha="center",
                fontsize=7.5, color=MPL_PRIMAER, fontweight="bold")
        ax.text(xi, 0.004, f"n={_fmt(r['kunden'])}", ha="center", va="bottom",
                fontsize=6.5, color="white")
    ax.set_xticks(x, [str(i) for i in th.index], fontsize=8)
    ax.set_xlabel("Anteil des Zusatzgeschäfts am Gesamtbeitrag", fontsize=8)
    ax.set_ylabel("Anteil mit abweichender Einstufung", fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.margins(y=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_delta_verteilung(k: pd.DataFrame) -> plt.Figure:
    """Verteilung des Margenunterschieds bei Verbundkunden."""
    d = k.loc[k["hat_zusatz"], "delta_marge"].dropna().clip(-1.0, 1.0)
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    ax.hist(d, bins=np.linspace(-1, 1, 61), color=MPL_AKZENT)
    ax.axvline(0, color=MPL_PRIMAER, lw=1.1)
    if len(d):
        ax.axvline(float(d.median()), color=MPL_NEGATIV, ls=":", lw=1.3)
        ax.text(float(d.median()), ax.get_ylim()[1] * 0.95,
                f" Median {_pct(float(d.median()), 1)}", fontsize=7.5,
                color=MPL_NEGATIV, va="top")
    ax.set_xlabel("Marge PKW-Sicht minus Marge Gesamtsicht", fontsize=8)
    ax.set_ylabel("Kunden mit Zusatzgeschäft", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_sensitivitaet_sichten(sens: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.plot(sens.index, sens["negativ_pkw_sicht"], marker="o", color=MPL_NEGATIV, lw=1.8,
            label="negativ in PKW-Sicht")
    ax.plot(sens.index, sens["negativ_kfz_sicht"], marker="D", color=MPL_GRAU, lw=1.5,
            label="negativ in KFZ-Sicht")
    ax.plot(sens.index, sens["negativ_gesamtsicht"], marker="s", color=MPL_POSITIV, lw=1.8,
            label="negativ in Gesamtsicht")
    ax.plot(sens.index, sens["wechselquote"], marker="^", color=MPL_AKZENT, lw=1.8,
            ls="--", label="abweichende Einstufung")
    ax.axvline(0.20, color=MPL_GRAU, ls="--", lw=1)
    ax.set_xlabel("Schwellenwert", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ======================================================================================
# Befunde
# ======================================================================================
def _befunde(erg: Dict[str, pd.DataFrame]) -> List[str]:
    k, ew = erg["kunden"], erg["eckwerte"]["wert"]
    wv, wd, rk = erg["wertvergleich"], erg["wertdelta"], erg["rangkennzahlen"]
    a: List[str] = []

    a.append(f"Von {_fmt(ew['PKW-Kunden gesamt'])} PKW-Kunden halten "
             f"<b>{_pct(ew['Anteil mit Zusatzgeschäft'], 1)}</b> weitere Verträge - "
             f"{_fmt(ew['davon mit KFZ_Rest'])} im übrigen KFZ, "
             f"{_fmt(ew['davon mit HUS'])} in HUS. Nur bei ihnen unterscheiden sich die "
             f"Sichten; für reine PKW-Kunden sind alle drei identisch.")

    a.append(f"Der Kundenwert selbst verschiebt sich deutlich: Das Ergebnis je Kunde "
             f"(Median) liegt bei <b>{_eur(wv.loc['PKW-Sicht', 'profit_je_kunde_median'])}</b> "
             f"in der PKW-Sicht, {_eur(wv.loc['KFZ-Sicht', 'profit_je_kunde_median'])} in "
             f"der KFZ-Sicht und {_eur(wv.loc['Gesamtsicht', 'profit_je_kunde_median'])} "
             f"in der Gesamtsicht; die Marge des betrachteten Bestands steigt von "
             f"{_pct(wv.loc['PKW-Sicht', 'marge'], 1)} auf "
             f"{_pct(wv.loc['Gesamtsicht', 'marge'], 1)}.")

    if "PKW minus gesamt" in wd.index:
        r = wd.loc["PKW minus gesamt"]
        a.append(f"Je Kunde weichen die Werte im Median um "
                 f"{_eur(r['median_eur'])} ab, bei einem Zehntel der Kunden um mehr als "
                 f"{_eur(abs(r['eur_p10']))} nach unten oder {_eur(r['eur_p90'])} nach "
                 f"oben. Bei <b>{_pct(r['anteil_ueber_100_eur'], 1)}</b> der Kunden liegt "
                 f"der Unterschied über 100 EUR, bei "
                 f"{_pct(r['anteil_ueber_10_pp'], 1)} über 10 Margenpunkten.")

    if "PKW gegen gesamt" in rk.index and "PKW gegen KFZ" in rk.index:
        rg, rk2 = rk.loc["PKW gegen gesamt"], rk.loc["PKW gegen KFZ"]
        a.append(f"Für die Reihenfolge ist der Sprung zur Gesamtsicht der entscheidende: "
                 f"Die Rangkorrelation beträgt <b>{_fmt(rg['spearman'], 2)}</b> zwischen "
                 f"PKW und gesamt, aber {_fmt(rk2['spearman'], 2)} zwischen PKW und KFZ. "
                 f"Nur {_pct(rg['gleiches_dezil'], 1)} der Kunden bleiben im selben Dezil, "
                 f"<b>{_pct(rg['sprung_2_plus'], 1)}</b> springen um zwei Dezile oder mehr.")

    au = erg.get("auswahlueberlappung")
    if au is not None and len(au) and 0.10 in au.index:
        r = au.loc[0.10]
        a.append(f"Konkret für eine Auswahl der schlechtesten 10 %: Von "
                 f"{_fmt(r['kunden_je_liste'])} Kunden je Liste stehen "
                 f"<b>{_pct(r['pkw_vs_gesamt'], 0)}</b> auf beiden. "
                 f"{_fmt(r['nur_in_pkw_liste'])} Kunden würden nur nach PKW-Sicht "
                 f"ausgewählt - sie tragen über alle Verträge zusammen "
                 f"{_eur_kurz(r['ergebnis_nur_pkw_liste'])} Ergebnis bei "
                 f"{_eur_kurz(r['beitrag_nur_pkw_liste'])} Beitrag.")

    th = erg.get("treiber_husanteil")
    if th is not None and len(th) > 1:
        oben, unten = th["delta_marge_median"].abs().idxmax(), th["delta_marge_median"].abs().idxmin()
        a.append(f"Der Unterschied wächst mit dem Gewicht des Zusatzgeschäfts: Macht es "
                 f"<b>{oben}</b> des Beitrags aus, liegt der Margenunterschied im Median "
                 f"bei {_pct(th.loc[oben, 'delta_marge_median'], 1)}, bei <b>{unten}</b> "
                 f"nur bei {_pct(th.loc[unten, 'delta_marge_median'], 1)}.")

    a.append(f"Legt man zusätzlich eine Negativ-Grenze an, ändert sich die Einstufung bei "
             f"{_pct(ew['Anteil an allen PKW-Kunden'], 1)} der Kunden - diese Zahl hängt "
             f"aber an der gewählten Schwelle und ist nur ein Ausschnitt der oben "
             f"gezeigten Wertverschiebung.")
    return a
# ======================================================================================
# PDF
# ======================================================================================
def erstelle_pkw_report(
    erg: Dict[str, pd.DataFrame],
    pfad: str = "PKW_Sichtvergleich.pdf",
    titel: str = "PKW-Sicht gegen Gesamtsicht",
    untertitel: str = "Aggregationsperspektive für PKW-Use-Cases",
    quelle: str = "",
    verfasser: str = "",
    stand: Optional[str] = None,
) -> str:
    """Erzeugt den PDF-Report zum Sichtvergleich."""
    st = _styles()
    stand = stand or date.today().strftime("%d.%m.%Y")
    k, ew = erg["kunden"], erg["eckwerte"]["wert"]
    schwelle = k.attrs.get("schwelle", 0.20)
    breite = A4[0] - 4.4 * cm

    doc = _Doc(pfad, titel, stand)
    s: List = []

    # ------------------------------------------------------------- Titelkopf
    s.append(Paragraph(titel, st["doc_titel"]))
    s.append(Paragraph(untertitel, st["doc_untertitel"]))
    meta = [f"Stand: {stand}",
            f"Grundgesamtheit: {_fmt(ew['PKW-Kunden gesamt'])} Kunden mit mindestens "
            f"einem PKW-Vertrag"]
    if quelle:
        meta.append(quelle)
    if verfasser:
        meta.append(verfasser)
    zeile = Table([[Paragraph("&nbsp;&nbsp;|&nbsp;&nbsp;".join(meta), st["klein"])]],
                  colWidths=[breite], hAlign="LEFT")
    zeile.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, C_LINIE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, C_LINIE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    s.append(Spacer(1, 6))
    s.append(zeile)
    s.append(Spacer(1, 16))

    # ------------------------------------------------------------ Fragestellung
    s.append(Paragraph("Fragestellung", st["h1"]))
    s.append(Paragraph(
        "Für PKW-Use-Cases lässt sich der Kundenwert auf drei Arten aggregieren: nur "
        "über die PKW-Verträge, über alle KFZ-Verträge (PKW und KFZ_Rest wie Kraftrad, "
        "LKW oder Anhänger) oder über den gesamten Kompositbestand. Verglichen werden die "
        "Werte selbst und die Reihenfolge, die sich daraus ergibt - denn für Auswahl und "
        "Priorisierung zählt, welcher Kunde vor welchem steht, nicht ob er eine Grenze "
        "reißt.", st["lead"]))

    wv, rk = erg["wertvergleich"], erg["rangkennzahlen"]
    s.append(_eckwerte([
        (_fmt(ew["PKW-Kunden gesamt"]), "PKW-Kunden"),
        (_pct(ew["Anteil mit Zusatzgeschäft"], 1), "mit weiteren Verträgen"),
        (_eur(wv.loc["PKW-Sicht", "profit_je_kunde_median"]), "Ergebnis je Kunde, PKW"),
        (_eur(wv.loc["Gesamtsicht", "profit_je_kunde_median"]), "Ergebnis je Kunde, gesamt"),
        (_fmt(rk.loc["PKW gegen gesamt", "spearman"], 2)
         if "PKW gegen gesamt" in rk.index else "n. v.", "Rangkorrelation"),
    ], breite, st))
    s.append(Spacer(1, 12))

    s.append(_bild(_chart_werte_je_sicht(wv), breite))
    s.append(Paragraph(
        "Der Kundenwert über die drei Sichten, jeweils als Median über alle PKW-Kunden.",
        st["klein"]))
    s.append(Spacer(1, 12))

    s.append(Paragraph("Befunde", st["h2"]))
    for b in _befunde(erg):
        s.append(Paragraph(b, st["bullet"], bulletText="\u25aa"))
    s.append(PageBreak())

    # ------------------------------------------------- Seite: Werte je Kunde
    s.append(Paragraph("Der Kundenwert in den drei Sichten", st["h1"]))
    s.append(Paragraph(
        "Wie weit liegen die Werte desselben Kunden auseinander, je nachdem welchen "
        "Bestand man aggregiert?", st["lead"]))

    wd = erg["wertdelta"]
    teiler, einheit = _skala(list(wv["beitrag"]) + list(wv["profit"]))
    zeilen = []
    for i, r in wv.iterrows():
        zeilen.append([str(i), _fmt(r["beitrag"] / teiler, 1), _fmt(r["profit"] / teiler, 1),
                       _pct(r["marge"], 1), _eur(r["profit_je_kunde_mittel"]),
                       _eur(r["profit_je_kunde_median"]),
                       _pct(r["marge_je_kunde_median"], 1),
                       _pct(r["kunden_mit_negativem_wert"], 1)])
    s.append(_tabelle(["Sicht", f"Beitrag<br/>({einheit})", f"Ergebnis<br/>({einheit})",
                       "Marge", "Ergebnis je<br/>Kunde (Ø)",
                       "Ergebnis je<br/>Kunde (Median)", "Marge je Kunde<br/>(Median)",
                       "Kunden mit<br/>Verlust"], zeilen,
                      [breite * 0.135, breite * 0.115, breite * 0.115, breite * 0.09,
                       breite * 0.135, breite * 0.15, breite * 0.15, breite * 0.11], st))
    s.append(Spacer(1, 14))

    s.append(_zwei_bilder(_chart_wertdelta(k), _chart_streuung_sichten(k, schwelle), breite))
    s.append(Paragraph(
        "Links: Wertunterschied je Kunde in Euro (PKW-Sicht minus Gesamtsicht), auf das "
        "98-Prozent-Quantil beschnitten. Rechts: Marge je Kunde in beiden Sichten - "
        "Punkte auf der Diagonale sind reine PKW-Kunden, bei ihnen sind alle Sichten "
        "identisch.", st["klein"]))
    s.append(Spacer(1, 14))

    zeilen = []
    for i, r in wd.iterrows():
        zeilen.append([str(i), _eur(r["median_eur"]), _eur(r["eur_p10"]),
                       _eur(r["eur_p90"]), _pct(r["median_marge"], 1),
                       _pct(r["anteil_ueber_100_eur"], 1), _pct(r["anteil_ueber_10_pp"], 1)])
    s.append(_tabelle(["Wertunterschied je Kunde", "Median", "10-%-<br/>Quantil",
                       "90-%-<br/>Quantil", "Median in<br/>Margenpunkten",
                       "über<br/>100 EUR", "über<br/>10 Punkte"], zeilen,
                      [breite * 0.21, breite * 0.10, breite * 0.125, breite * 0.125,
                       breite * 0.17, breite * 0.135, breite * 0.135], st))
    s.append(PageBreak())

    # ------------------------------------------------- Seite: Reihenfolge
    s.append(Paragraph("Die Reihenfolge der Kunden", st["h1"]))
    s.append(Paragraph(
        "Für Auswahl und Priorisierung zählt die Rangfolge. Die Matrix zeigt, wo ein "
        "Kunde landet, wenn man ihn statt nach PKW-Marge nach Gesamtmarge sortiert - "
        "die Diagonale sind Kunden, die ihr Dezil halten.", st["lead"]))

    dm = erg.get("dezil_matrix_pkw_gesamt")
    au = erg.get("auswahlueberlappung")
    if dm is not None and len(dm):
        s.append(_zwei_bilder(_chart_dezilmatrix(dm), _chart_ueberlappung(au)
                              if au is not None and len(au) else None, breite))
        s.append(Paragraph(
            "Links: Dezilwanderung von der PKW- zur Gesamtsicht, Zeilenanteile in Prozent. "
            "Rechts: Anteil gemeinsamer Kunden, wenn man nach beiden Sichten jeweils die "
            "schlechtesten X Prozent auswählt.", st["klein"]))
        s.append(Spacer(1, 14))

    if len(rk):
        zeilen = [[str(i), _fmt(r["spearman"], 2),
                   _fmt(r.get("kendall", np.nan), 2),
                   _pct(r["perzentilverschiebung_mittel"], 1),
                   _pct(r["gleiches_dezil"], 1), _pct(r["sprung_1_dezil"], 1),
                   _pct(r["sprung_2_plus"], 1), _pct(r["sprung_3_plus"], 1)]
                  for i, r in rk.iterrows()]
        s.append(_tabelle(["Sichtpaar", "Spearman", "Kendall",
                           "Rang-<br/>verschiebung", "gleiches<br/>Dezil",
                           "1 Dezil", "2 Dezile<br/>und mehr", "3 Dezile<br/>und mehr"],
                          zeilen,
                          [breite * 0.17, breite * 0.11, breite * 0.10, breite * 0.15,
                           breite * 0.12, breite * 0.10, breite * 0.13, breite * 0.12], st))
        s.append(Spacer(1, 6))
        s.append(Paragraph(
            "Die Rangverschiebung ist der mittlere Abstand der Perzentilränge desselben "
            "Kunden zwischen beiden Sichten.", st["klein"]))
        s.append(Spacer(1, 14))

    if au is not None and len(au):
        zeilen = [[f"schlechteste {_pct(i, 0)}", _fmt(r["kunden_je_liste"]),
                   _pct(r["pkw_vs_kfz"], 1), _pct(r["kfz_vs_gesamt"], 1),
                   _pct(r["pkw_vs_gesamt"], 1), _fmt(r["nur_in_pkw_liste"]),
                   _eur_kurz(r["ergebnis_nur_pkw_liste"])]
                  for i, r in au.iterrows()]
        s.append(Paragraph("Überlappung der Auswahllisten", st["h2"]))
        s.append(_tabelle(["Auswahl", "Kunden<br/>je Liste", "PKW gegen<br/>KFZ",
                           "KFZ gegen<br/>gesamt", "PKW gegen<br/>gesamt",
                           "nur in<br/>PKW-Liste", "deren Ergebnis<br/>gesamt"], zeilen,
                          [breite * 0.19, breite * 0.12, breite * 0.13, breite * 0.13,
                           breite * 0.13, breite * 0.12, breite * 0.18], st))
        s.append(Spacer(1, 6))
        s.append(Paragraph(
            "Die letzte Spalte zeigt, welches Gesamtergebnis hinter den Kunden steht, die "
            "nur die PKW-Sicht auf die Liste bringt - je positiver, desto teurer wäre eine "
            "Maßnahme, die sie fälschlich trifft.", st["klein"]))
    s.append(PageBreak())

    # ------------------------------------------------------------ Kreuztabelle
    s.append(Paragraph("Anhang: Wirkung auf eine binäre Einstufung", st["h1"]))
    s.append(Paragraph(
        f"Legt man zusätzlich eine Negativ-Grenze an - Ergebnis negativ und betragsmäßig "
        f"mindestens {_pct(schwelle, 0)} des Beitrags -, entstehen aus den Werten vier "
        f"Gruppen. Diese Zahlen hängen an der gewählten Schwelle und sind deshalb "
        f"nachrangig zu den Wert- und Rangunterschieden.", st["lead"]))
    s.append(_zwei_bilder(_chart_gruppen(k), _chart_ueberlappung(au)
                          if False else None, breite))
    s.append(Spacer(1, 8))
    s.append(Paragraph("Einstufung in beiden Sichten", st["h2"]))
    s.append(Paragraph(
        "Jede Zeile steht für eine Kombination aus PKW-Sicht und Gesamtsicht. Die beiden "
        "gemischten Zeilen sind die Fälle, in denen die Perspektive die Entscheidung "
        "ändert.", st["text"]))

    kt = erg["kreuztabelle"]
    teiler, einheit = _skala(list(kt["beitrag_gesamt"]) + list(kt["profit_gesamt"]))
    zeilen, fett = [], []
    for flag_pkw in ("negativ", "positiv"):
        for flag_ges in ("negativ", "positiv"):
            if (flag_pkw, flag_ges) not in kt.index:
                continue
            r = kt.loc[(flag_pkw, flag_ges)]
            zeilen.append([f"{flag_pkw} / {flag_ges}", _fmt(r["kunden"]),
                           _pct(r["anteil_kunden"], 1),
                           _fmt(r["beitrag_pkw"] / teiler, 1),
                           _fmt(r["beitrag_gesamt"] / teiler, 1),
                           _fmt(r["profit_pkw"] / teiler, 1),
                           _fmt(r["profit_gesamt"] / teiler, 1)])
            if flag_pkw != flag_ges:
                fett.append(len(zeilen))
    s.append(_tabelle([f"PKW-Sicht / Gesamtsicht", "Kunden", "Anteil",
                       f"Beitrag PKW<br/>({einheit})", f"Beitrag gesamt<br/>({einheit})",
                       f"Ergebnis PKW<br/>({einheit})", f"Ergebnis gesamt<br/>({einheit})"],
                      zeilen,
                      [breite * 0.22, breite * 0.10, breite * 0.09, breite * 0.1475,
                       breite * 0.1475, breite * 0.1475, breite * 0.1475],
                      st, hervorheben=fett))
    s.append(Spacer(1, 6))
    s.append(Paragraph("Fett hervorgehoben die beiden Gruppen, in denen die Perspektive "
                       "die Einstufung ändert.", st["klein"]))
    s.append(Spacer(1, 16))

    s.append(Paragraph("Profil der vier Gruppen", st["h2"]))
    wg = erg["wechselgruppen"]
    zeilen = [[KURZ.get(str(i), str(i)), _fmt(r["kunden"]), _pct(r["anteil_kunden"], 1),
               _fmt(r["vertraege_kfz_rest"], 2), _fmt(r["vertraege_hus"], 2),
               _pct(r["zusatz_beitragsanteil"], 0), _eur(r["profit_zusatz_je_kunde"]),
               _pct(r["marge_pkw"], 1), _pct(r["marge_kfz"], 1),
               _pct(r["marge_gesamt"], 1)]
              for i, r in wg.iterrows() if r["kunden"] > 0]
    s.append(_tabelle(["Gruppe", "Kunden", "Anteil", "KFZ_Rest-<br/>Verträge",
                       "HUS-<br/>Verträge", "Zusatz-<br/>anteil", "Zusatz-Ergebnis<br/>je Kunde",
                       "Marge PKW<br/>(Median)", "Marge KFZ<br/>(Median)",
                       "Marge gesamt<br/>(Median)"], zeilen,
                      [breite * 0.155, breite * 0.075, breite * 0.07, breite * 0.10,
                       breite * 0.09, breite * 0.085, breite * 0.125, breite * 0.10,
                       breite * 0.10, breite * 0.10], st))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "Verträge, Zusatzanteil und Zusatz-Ergebnis sind Mittelwerte je Kunde, die Margen "
        "Mediane. Zusatz = alles außerhalb des PKW, also KFZ_Rest und HUS zusammen. Das "
        "Zusatz-Ergebnis je Kunde zeigt, in welche Richtung es die Gesamtsicht verschiebt.",
        st["klein"]))
    s.append(PageBreak())

    # ------------------------------------------------------------ Wo entsteht das Delta
    s.append(Paragraph("Wo das Delta entsteht", st["h1"]))
    ss, sw = erg["sichtstufen"], erg["stufenwechsel"]
    s.append(Paragraph("Erst der Schritt von einer Sicht zur nächsten zeigt, welcher "
                       "Bestandsteil die Einstufung verschiebt.", st["lead"]))
    zeilen = [[str(i), _fmt(r["betroffene_kunden"]), _fmt(r["kunden"]),
               _pct(r["anteil"], 1), _pct(r["quote_in_betroffenen"], 1)]
              for i, r in sw.iterrows()]
    s.append(_tabelle(["Schritt", "betroffene Kunden", "Einstufung ändert sich",
                       "Anteil aller<br/>PKW-Kunden", "Anteil der<br/>Betroffenen"], zeilen,
                      [breite * 0.26, breite * 0.18, breite * 0.20, breite * 0.18,
                       breite * 0.18], st))
    s.append(Spacer(1, 6))
    s.append(Paragraph("Betroffene Kunden = Kunden, die für den jeweiligen Schritt "
                       "überhaupt zusätzliche Verträge halten.", st["klein"]))
    s.append(Spacer(1, 16))
    s.append(Paragraph("Je schwerer das Geschäft außerhalb des PKW wiegt, desto größer "
                       "der Unterschied.", st["h2"]))

    th = erg.get("treiber_husanteil")
    if th is not None and len(th):
        s.append(_zwei_bilder(_chart_husanteil(th), _chart_delta_verteilung(k), breite))
        s.append(Paragraph(
            "Links: Anteil der Kunden mit abweichender Einstufung, gestaffelt nach dem "
            "Anteil des Zusatzgeschäfts am Gesamtbeitrag. Rechts: Verteilung des Margenunterschieds "
            "(PKW-Sicht minus Gesamtsicht). Positive Werte heißen, dass der Kunde im PKW "
            "besser dasteht als insgesamt.", st["klein"]))
        s.append(Spacer(1, 14))

        zeilen = [[str(i), _fmt(r["kunden"]), _pct(r["wechselquote"], 1),
                   _pct(r["delta_marge_median"], 1), _eur_kurz(r["beitrag_gesamt"])]
                  for i, r in th.iterrows()]
        s.append(_tabelle(["Zusatzanteil am Beitrag", "Kunden", "abweichende<br/>Einstufung",
                           "Delta Marge<br/>(Median)", "Beitragsvolumen"], zeilen,
                          [breite * 0.24, breite * 0.16, breite * 0.20, breite * 0.20,
                           breite * 0.20], st))
        s.append(Spacer(1, 16))

    tz = erg.get("treiber_zweig")
    if tz is not None and len(tz):
        s.append(Paragraph("Welche Zweige außerhalb des PKW die Verschiebung tragen",
                           st["h2"]))
        s.append(Paragraph(
            "Durchdringung = Anteil der Kunden einer Gruppe mit mindestens einem Vertrag "
            "dieses Zweigs, verglichen mit allen Kunden, die Zusatzgeschäft halten.",
            st["text"]))
        zeilen = []
        for i, r in tz.head(10).iterrows():
            zeilen.append([str(i)[:26], _pct(r.get("durchdringung_alle_zusatz"), 0),
                           _pct(r.get(f"durchdringung__{WECHSLER[0]}"), 0),
                           _pct(r.get(f"delta__{WECHSLER[0]}"), 1),
                           _pct(r.get(f"durchdringung__{WECHSLER[1]}"), 0),
                           _pct(r.get(f"delta__{WECHSLER[1]}"), 1)])
        s.append(_tabelle(["Zweig", "alle mit<br/>Zusatzgeschäft", "nur PKW-Sicht<br/>negativ",
                           "Delta", "nur Gesamtsicht<br/>negativ", "Delta"], zeilen,
                          [breite * 0.22, breite * 0.16, breite * 0.17, breite * 0.11,
                           breite * 0.17, breite * 0.11], st))
    s.append(PageBreak())

    # ------------------------------------------------------------ Auswirkung
    s.append(Paragraph("Auswirkung auf die Auswahl", st["h1"]))
    s.append(Paragraph(
        "Für Use Cases, die Kunden nach Wert priorisieren, zählt weniger die Einstufung "
        "als die Reihenfolge. Verglichen werden die schlechtesten Kunden nach beiden "
        "Sichten.", st["lead"]))

    rv = erg.get("rangvergleich")
    if rv is not None and len(rv):
        zeilen = [[f"schlechteste {_pct(i, 0)}", _fmt(r["kunden_je_liste"]),
                   _pct(r["pkw_vs_kfz"], 1), _pct(r["kfz_vs_gesamt"], 1),
                   _pct(r["pkw_vs_gesamt"], 1), _fmt(r["nur_in_pkw_liste"])]
                  for i, r in rv.iterrows()]
        s.append(_tabelle(["Auswahl", "Kunden<br/>je Liste", "PKW gegen<br/>KFZ",
                           "KFZ gegen<br/>gesamt", "PKW gegen<br/>gesamt",
                           "nur in<br/>PKW-Liste"], zeilen,
                          [breite * 0.22, breite * 0.14, breite * 0.16, breite * 0.16,
                           breite * 0.16, breite * 0.16], st))
        s.append(Spacer(1, 6))
        s.append(Paragraph("Angegeben ist der Anteil der Kunden, die in beiden Listen "
                           "derselben Auswahlgröße stehen.", st["klein"]))
        sp = rv.attrs.get("spearman")
        if sp is not None:
            s.append(Spacer(1, 8))
            s.append(_hinweis(
                f"Die Rangkorrelation der Kundenmargen zwischen PKW-Sicht und Gesamtsicht "
                f"beträgt <b>{_fmt(sp, 2)}</b>. Je näher an 1, desto ähnlicher die Reihenfolge - "
                f"für Priorisierungen ist das die relevantere Größe als die Trefferquote "
                f"der binären Einstufung.", st))
        s.append(Spacer(1, 16))

    s.append(Paragraph("Sensitivität der Segmentgrenze", st["h2"]))
    s.append(Paragraph("Wie stark hängen Segmentgrößen und Abweichungsquote an der "
                       "gewählten Schwelle?", st["text"]))
    s.append(_bild(_chart_sensitivitaet_sichten(erg["sensitivitaet"]), breite * 0.78))
    s.append(Spacer(1, 10))

    sens = erg["sensitivitaet"]
    zeilen = [[_pct(i, 0), _pct(r["negativ_pkw_sicht"], 1), _pct(r["negativ_kfz_sicht"], 1),
               _pct(r["negativ_gesamtsicht"], 1), _pct(r["wechselquote"], 1),
               _pct(r["nur_pkw_negativ"], 1), _pct(r["nur_gesamt_negativ"], 1)]
              for i, r in sens.iterrows()]
    s.append(_tabelle(["Schwelle", "negativ<br/>PKW", "negativ<br/>KFZ", "negativ<br/>gesamt",
                       "abweichend<br/>PKW zu gesamt", "nur PKW<br/>negativ",
                       "nur gesamt<br/>negativ"], zeilen,
                      [breite * 0.115, breite * 0.125, breite * 0.125, breite * 0.135,
                       breite * 0.19, breite * 0.155, breite * 0.155], st))

    doc.build(s)
    return pfad
