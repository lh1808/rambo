"""
pkw_sichtvergleich.py
=====================

Vergleicht für KFZ-Kunden zwei Aggregationsperspektiven:

    KFZ-Sicht    : je Kunde werden nur die KFZ-Verträge aggregiert
                   (PKW und KFZ_Rest wie Kraftrad, LKW, Anhänger zusammen)
    Gesamtsicht  : je Kunde wird der komplette Kompositbestand aggregiert (KFZ und HUS)

Im Mittelpunkt stehen die Werte selbst und die Reihenfolge, die sich daraus ergibt -
für Auswahl und Priorisierung zählt, welcher Kunde vor welchem steht. Eine binäre
Einstufung mit Schwellenwert kommt hier bewusst nicht vor.

Hält ein Kunde nur KFZ-Verträge, sind beide Sichten identisch; das Delta stammt
ausschließlich von Kunden mit zusätzlichem HUS-Geschäft.

Verwendung:
    from pkw_sichtvergleich import analysiere_sichten, erstelle_pkw_report

    erg = analysiere_sichten(df)
    erstelle_pkw_report(erg, "KFZ_Sichtvergleich.pdf")

Ergebnis-Dict:
    kunden              Kundenebene mit beiden Sichten
    eckwerte            Kernzahlen der Grundgesamtheit
    wertvergleich       Beitrag, Ergebnis und Marge je Sicht
    wertdelta           Quantile des Wertunterschieds je Kunde (EUR und Margenpunkte)
    rangkennzahlen      Rangkorrelation, Perzentilverschiebung, Dezilwanderung
    dezil_matrix        Wanderung zwischen den Dezilen beider Sichten
    auswahlueberlappung Überlappung der Auswahllisten je Auswahlgröße
    treiber_husanteil   Delta gestaffelt nach Gewicht des HUS-Geschäfts
    treiber_zweig       HUS-Zweige der Kunden mit großem Delta
"""

import io
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
def kunden_zwei_sichten(df: pd.DataFrame,
                        pkw_produkte: Sequence[str] = ("PKW",),
                        kfz_rest_produkte: Sequence[str] = KFZ_REST_PRODUKTE,
                        nur_pkw_kunden: bool = False) -> pd.DataFrame:
    """
    Kundenebene für alle Kunden mit mindestens einem KFZ-Vertrag, jeweils mit
    KFZ-Sicht (PKW und KFZ_Rest) und Gesamtsicht (zusätzlich HUS).

    nur_pkw_kunden=True schränkt die Grundgesamtheit auf Kunden mit mindestens einem
    PKW-Vertrag ein.
    """
    d = ergaenze_ebenen(bereinige(df), pkw_produkte=pkw_produkte,
                        kfz_rest_produkte=kfz_rest_produkte)
    v = vertragsaggregat(d)

    gesamt = v.groupby(ID, observed=True).agg(
        n_vertraege_gesamt=(VERTRAG, "nunique"),
        beitrag_gesamt=("beitrag", "sum"),
        profit_gesamt=("profit", "sum"))
    je_bereich = v.pivot_table(index=ID, columns=BEREICH, values=["beitrag", "profit"],
                               aggfunc="sum", observed=True).fillna(0.0)
    anzahl = v.pivot_table(index=ID, columns=BEREICH, values=VERTRAG,
                           aggfunc="nunique", observed=True).fillna(0)

    k = gesamt.copy()
    for b, sp in (("PKW", "pkw"), ("KFZ_Rest", "kfz_rest"), ("HUS", "hus")):
        k[f"beitrag_{sp}"] = (je_bereich[("beitrag", b)]
                              if ("beitrag", b) in je_bereich.columns else 0.0)
        k[f"profit_{sp}"] = (je_bereich[("profit", b)]
                             if ("profit", b) in je_bereich.columns else 0.0)
        k[f"n_vertraege_{sp}"] = anzahl[b] if b in anzahl.columns else 0

    # KFZ-Sicht = PKW und KFZ_Rest zusammen
    k["beitrag_kfz"] = k["beitrag_pkw"] + k["beitrag_kfz_rest"]
    k["profit_kfz"] = k["profit_pkw"] + k["profit_kfz_rest"]
    k["n_vertraege_kfz"] = k["n_vertraege_pkw"] + k["n_vertraege_kfz_rest"]

    k = k[k["n_vertraege_pkw"] > 0] if nur_pkw_kunden else k[k["n_vertraege_kfz"] > 0]
    k = k.copy()

    k["hat_pkw"] = k["n_vertraege_pkw"] > 0
    k["hat_kfz_rest"] = k["n_vertraege_kfz_rest"] > 0
    k["hat_hus"] = k["n_vertraege_hus"] > 0
    k["bestandsprofil"] = np.where(k["hat_hus"], "KFZ und HUS", "nur KFZ")
    k["hus_beitragsanteil"] = np.where(k["beitrag_gesamt"] > 0,
                                       k["beitrag_hus"] / k["beitrag_gesamt"], np.nan)

    for sicht in ("kfz", "gesamt"):
        k[f"marge_{sicht}"] = np.where(k[f"beitrag_{sicht}"] > 0,
                                       k[f"profit_{sicht}"] / k[f"beitrag_{sicht}"], np.nan)
    k["delta_marge"] = k["marge_kfz"] - k["marge_gesamt"]
    k["delta_profit"] = k["profit_kfz"] - k["profit_gesamt"]
    return k


def eckwerte(k: pd.DataFrame) -> pd.DataFrame:
    """Kernzahlen der Grundgesamtheit."""
    zeilen = {
        "KFZ-Kunden gesamt": len(k),
        "davon mit PKW": int(k["hat_pkw"].sum()),
        "davon mit KFZ_Rest": int(k["hat_kfz_rest"].sum()),
        "davon zusätzlich HUS": int(k["hat_hus"].sum()),
        "Anteil mit HUS-Geschäft": k["hat_hus"].mean(),
        "KFZ-Verträge je Kunde (Mittel)": k["n_vertraege_kfz"].mean(),
        "HUS-Verträge je Kunde (Mittel)": k["n_vertraege_hus"].mean(),
        "HUS-Anteil am Beitrag (Median)": k["hus_beitragsanteil"].median(),
    }
    return pd.DataFrame({"wert": zeilen})


def wertvergleich(k: pd.DataFrame) -> pd.DataFrame:
    """Der Kundenwert je Sicht: aggregiertes Ergebnis, Ergebnis je Kunde, Marge."""
    zeilen = []
    for sicht, name in (("kfz", "KFZ-Sicht"), ("gesamt", "Gesamtsicht")):
        beitrag, profit = k[f"beitrag_{sicht}"].sum(), k[f"profit_{sicht}"].sum()
        zeilen.append({
            "sicht": name,
            "vertraege": k[f"n_vertraege_{sicht}"].sum(),
            "beitrag": beitrag,
            "profit": profit,
            "marge": profit / beitrag if beitrag else np.nan,
            "profit_je_kunde_mittel": k[f"profit_{sicht}"].mean(),
            "profit_je_kunde_median": k[f"profit_{sicht}"].median(),
            "marge_je_kunde_median": k[f"marge_{sicht}"].median(),
            "kunden_mit_verlust": float((k[f"profit_{sicht}"] < 0).mean()),
        })
    return pd.DataFrame(zeilen).set_index("sicht")


def wertdelta(k: pd.DataFrame) -> pd.DataFrame:
    """
    Verteilung des Wertunterschieds je Kunde. Positiv heißt: In der KFZ-Sicht steht
    der Kunde besser da als in der Gesamtsicht.
    """
    verbund = k[k["hat_hus"]]
    q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    return pd.DataFrame({
        "quantil": q,
        "delta_eur_alle": k["delta_profit"].quantile(q).values,
        "delta_eur_mit_hus": (verbund["delta_profit"].quantile(q).values
                              if len(verbund) else np.nan),
        "delta_marge_mit_hus": (verbund["delta_marge"].quantile(q).values
                                if len(verbund) else np.nan),
    }).set_index("quantil")


def delta_kennzahlen(k: pd.DataFrame) -> pd.DataFrame:
    """Verdichtete Kennzahlen zum Wertunterschied, insgesamt und für Kunden mit HUS."""
    verbund = k[k["hat_hus"]]
    zeilen = {}
    for name, d in (("alle KFZ-Kunden", k), ("Kunden mit HUS", verbund)):
        if not len(d):
            continue
        zeilen[name] = {
            "kunden": len(d),
            "median_eur": d["delta_profit"].median(),
            "mittel_eur": d["delta_profit"].mean(),
            "median_marge": d["delta_marge"].median(),
            "anteil_ueber_50_eur": float((d["delta_profit"].abs() > 50).mean()),
            "anteil_ueber_100_eur": float((d["delta_profit"].abs() > 100).mean()),
            "anteil_ueber_10_pp": float((d["delta_marge"].abs() > 0.10).mean()),
            "anteil_vorzeichenwechsel": float(
                ((d["profit_kfz"] < 0) != (d["profit_gesamt"] < 0)).mean()),
        }
    return pd.DataFrame(zeilen).T


def dezil_matrix(k: pd.DataFrame, groesse: str = "marge", n: int = 10) -> pd.DataFrame:
    """
    Wanderung zwischen den Dezilen: Wo landet ein Kunde aus Dezil X der KFZ-Sicht,
    wenn nach der Gesamtsicht sortiert wird? Dezil 1 = schlechteste Kunden,
    Werte sind Zeilenanteile.
    """
    d = k[[f"{groesse}_kfz", f"{groesse}_gesamt"]].dropna()
    if d.empty:
        return pd.DataFrame()
    labels = list(range(1, n + 1))
    da = pd.qcut(d[f"{groesse}_kfz"].rank(method="first"), n, labels=labels)
    db = pd.qcut(d[f"{groesse}_gesamt"].rank(method="first"), n, labels=labels)
    m = pd.crosstab(da, db, normalize="index")
    m.index.name = "Dezil KFZ-Sicht"
    m.columns.name = "Dezil Gesamtsicht"
    return m


def rangkennzahlen(k: pd.DataFrame, groesse: str = "marge", n: int = 10) -> pd.DataFrame:
    """Wie stabil ist die Reihenfolge zwischen beiden Sichten?"""
    zeilen = {}
    for name, d in (("alle KFZ-Kunden", k), ("nur Kunden mit HUS", k[k["hat_hus"]])):
        d = d[[f"{groesse}_kfz", f"{groesse}_gesamt"]].dropna()
        if len(d) < n:
            continue
        ra = d[f"{groesse}_kfz"].rank(pct=True, method="first")
        rb = d[f"{groesse}_gesamt"].rank(pct=True, method="first")
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
    Überlappung der Auswahllisten: Wie viele Kunden stehen auf beiden Listen, wenn man
    die schlechtesten X Prozent nach der jeweiligen Sicht auswählt? Zusätzlich wird
    ausgewiesen, welches Volumen hinter den nur einseitig ausgewählten Kunden steht.
    """
    d = k[[f"{groesse}_kfz", f"{groesse}_gesamt", "profit_gesamt", "beitrag_gesamt",
           "profit_kfz"]].dropna(subset=[f"{groesse}_kfz", f"{groesse}_gesamt"])
    if d.empty:
        return pd.DataFrame()
    zeilen = []
    for a in anteile:
        n = max(1, int(round(len(d) * a)))
        liste_kfz = set(d[f"{groesse}_kfz"].nsmallest(n).index)
        liste_ges = set(d[f"{groesse}_gesamt"].nsmallest(n).index)
        nur_kfz = list(liste_kfz - liste_ges)
        nur_ges = list(liste_ges - liste_kfz)
        zeilen.append({
            "auswahl": a,
            "kunden_je_liste": n,
            "uebereinstimmung": len(liste_kfz & liste_ges),
            "anteil_uebereinstimmung": len(liste_kfz & liste_ges) / n,
            "nur_kfz_liste": len(nur_kfz),
            "ergebnis_nur_kfz_liste": d.loc[nur_kfz, "profit_gesamt"].sum(),
            "beitrag_nur_kfz_liste": d.loc[nur_kfz, "beitrag_gesamt"].sum(),
            "ergebnis_nur_gesamtliste": d.loc[nur_ges, "profit_gesamt"].sum(),
        })
    return pd.DataFrame(zeilen).set_index("auswahl")


def treiber_husanteil(k: pd.DataFrame) -> pd.DataFrame:
    """Wertunterschied und Rangverschiebung, gestaffelt nach Gewicht des HUS-Geschäfts."""
    verbund = k[k["hat_hus"]].copy()
    if verbund.empty:
        return pd.DataFrame()
    kanten = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    labels = ["bis 10 %", "10-25 %", "25-50 %", "50-75 %", "über 75 %"]
    verbund["klasse"] = pd.cut(verbund["hus_beitragsanteil"], bins=kanten,
                               labels=labels, include_lowest=True)

    rang_kfz = k["marge_kfz"].rank(pct=True, method="first")
    rang_ges = k["marge_gesamt"].rank(pct=True, method="first")
    verbund["rangverschiebung"] = (rang_kfz - rang_ges).abs().reindex(verbund.index)

    g = verbund.groupby("klasse", observed=True).agg(
        kunden=("delta_profit", "size"),
        delta_eur_median=("delta_profit", "median"),
        delta_marge_median=("delta_marge", "median"),
        anteil_ueber_10_pp=("delta_marge", lambda s: float((s.abs() > 0.10).mean())),
        rangverschiebung=("rangverschiebung", "mean"),
        beitrag_gesamt=("beitrag_gesamt", "sum"))
    return g[g["kunden"] > 0]


def treiber_zweig(df: pd.DataFrame, k: pd.DataFrame,
                  pkw_produkte: Sequence[str] = ("PKW",),
                  kfz_rest_produkte: Sequence[str] = KFZ_REST_PRODUKTE,
                  top_n: int = 10) -> pd.DataFrame:
    """
    Welche HUS-Zweige verschieben den Wert? Je Zweig das Ergebnis, das die Halter
    aus diesem Zweig ziehen, und wie stark ihre Reihenfolge sich verschiebt.
    """
    d = ergaenze_ebenen(bereinige(df), pkw_produkte=pkw_produkte,
                        kfz_rest_produkte=kfz_rest_produkte, melde_zuordnung=False)
    hus = d[(d[BEREICH] == "HUS") & (d[ID].isin(k.index[k["hat_hus"]]))]
    if hus.empty:
        return pd.DataFrame()

    rang_kfz = k["marge_kfz"].rank(pct=True, method="first")
    rang_ges = k["marge_gesamt"].rank(pct=True, method="first")
    verschiebung = (rang_kfz - rang_ges).abs()

    g = hus.groupby(ZWEIG, observed=True).agg(
        kunden=(ID, "nunique"), vertraege=(VERTRAG, "nunique"),
        beitrag=(BEITRAG, "sum"), profit=(PROFIT, "sum"))
    g["marge"] = np.where(g["beitrag"] > 0, g["profit"] / g["beitrag"], np.nan)
    g["profit_je_halter"] = g["profit"] / g["kunden"]
    g["durchdringung"] = g["kunden"] / int(k["hat_hus"].sum())
    g["rangverschiebung_halter"] = (
        hus.groupby(ZWEIG, observed=True)[ID]
        .apply(lambda s: float(verschiebung.reindex(s.unique()).mean())))
    return g.sort_values("kunden", ascending=False).head(top_n)


def analysiere_sichten(df: pd.DataFrame,
                       pkw_produkte: Sequence[str] = ("PKW",),
                       kfz_rest_produkte: Sequence[str] = KFZ_REST_PRODUKTE,
                       nur_pkw_kunden: bool = False) -> Dict[str, pd.DataFrame]:
    """Führt den kompletten Sichtvergleich aus."""
    k = kunden_zwei_sichten(df, pkw_produkte=pkw_produkte,
                            kfz_rest_produkte=kfz_rest_produkte,
                            nur_pkw_kunden=nur_pkw_kunden)
    return {
        "kunden": k,
        "eckwerte": eckwerte(k),
        "wertvergleich": wertvergleich(k),
        "wertdelta": wertdelta(k),
        "delta_kennzahlen": delta_kennzahlen(k),
        "rangkennzahlen": rangkennzahlen(k),
        "dezil_matrix": dezil_matrix(k),
        "auswahlueberlappung": auswahlueberlappung(k),
        "treiber_husanteil": treiber_husanteil(k),
        "treiber_zweig": treiber_zweig(df, k, pkw_produkte=pkw_produkte,
                                       kfz_rest_produkte=kfz_rest_produkte),
    }


def drucke_sichtreport(erg: Dict[str, pd.DataFrame]) -> None:
    """Kompakte Konsolenausgabe."""
    pd.set_option("display.width", 200, "display.max_columns", 60,
                  "display.float_format", lambda x: f"{x:,.3f}")
    print("=" * 100)
    print("KFZ-SICHT GEGEN GESAMTSICHT")
    print("=" * 100)
    print(erg["eckwerte"])
    print("\n--- Kundenwert je Sicht ---")
    print(erg["wertvergleich"])
    print("\n--- Wertunterschied je Kunde ---")
    print(erg["delta_kennzahlen"])
    print("\n--- Reihenfolge ---")
    print(erg["rangkennzahlen"])
    print("\n--- Überlappung der Auswahllisten ---")
    print(erg["auswahlueberlappung"])
    print("\n--- Dezilwanderung (Zeilenanteile) ---")
    print((erg["dezil_matrix"] * 100).round(0))
    print("\n--- Nach Gewicht des HUS-Geschäfts ---")
    print(erg["treiber_husanteil"])


# ======================================================================================
# Grafiken
# ======================================================================================
def _chart_ueberlappung(au: pd.DataFrame) -> plt.Figure:
    """Überlappung der Auswahllisten über verschiedene Auswahlgrößen."""
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    ax.plot(au.index, au["anteil_uebereinstimmung"], marker="o", color=MPL_NEGATIV, lw=2)
    for x, y in zip(au.index, au["anteil_uebereinstimmung"]):
        ax.text(x, y + 0.03, _pct(y, 0), ha="center", fontsize=7.5, color=MPL_PRIMAER,
                fontweight="bold")
    ax.set_xlabel("Größe der Auswahl (schlechteste ... Prozent)", fontsize=8)
    ax.set_ylabel("Anteil gemeinsamer Kunden", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.set_ylim(0, 1.08)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_dezilmatrix(m: pd.DataFrame) -> plt.Figure:
    """Dezilwanderung - die Diagonale sind Kunden, die ihren Rang halten."""
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
    ax.set_xlabel("Dezil in der Gesamtsicht", fontsize=8)
    ax.set_ylabel("Dezil in der KFZ-Sicht", fontsize=8)
    ax.set_title("Zeilenanteile in Prozent - Dezil 1 = schlechteste Kunden",
                 loc="left", fontsize=8, color=MPL_GRAU, pad=6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def _chart_streuung(k: pd.DataFrame, max_punkte: int = 6000) -> plt.Figure:
    """Marge je Kunde in beiden Sichten, getrennt nach Kunden mit und ohne HUS."""
    d = k[k["marge_kfz"].notna() & k["marge_gesamt"].notna()]
    if len(d) > max_punkte:
        d = d.sample(max_punkte, random_state=0)
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    ohne = d[~d["hat_hus"]]
    mit = d[d["hat_hus"]]
    ax.scatter(ohne["marge_gesamt"].clip(-1.5, 1), ohne["marge_kfz"].clip(-1.5, 1), s=4,
               alpha=0.30, color=MPL_GRAU, linewidths=0, label="nur KFZ")
    ax.scatter(mit["marge_gesamt"].clip(-1.5, 1), mit["marge_kfz"].clip(-1.5, 1), s=5,
               alpha=0.40, color=MPL_AKZENT, linewidths=0, label="KFZ und HUS")
    ax.plot([-1.5, 1], [-1.5, 1], color=MPL_PRIMAER, lw=1, ls="--")
    ax.set_xlabel("Marge in der Gesamtsicht", fontsize=8)
    ax.set_ylabel("Marge in der KFZ-Sicht", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.set_xlim(-1.5, 1)
    ax.set_ylim(-1.5, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_wertdelta(k: pd.DataFrame, breit: bool = False) -> plt.Figure:
    """Wertunterschied je Kunde in Euro - nur Kunden mit HUS-Geschäft."""
    d = k.loc[k["hat_hus"], "delta_profit"].dropna()
    grenze = float(d.abs().quantile(0.98)) or 1.0
    fig, ax = plt.subplots(figsize=(8.6, 2.5) if breit else (4.3, 3.1))
    ax.hist(d.clip(-grenze, grenze), bins=61, color=MPL_AKZENT)
    ax.axvline(0, color=MPL_PRIMAER, lw=1.1)
    if len(d):
        med = float(d.median())
        ax.axvline(med, color=MPL_NEGATIV, ls=":", lw=1.3)
        ax.text(med, ax.get_ylim()[1] * 0.95, f" Median {_eur(med)}", fontsize=7.5,
                color=MPL_NEGATIV, va="top")
    ax.set_xlabel("Ergebnis KFZ-Sicht minus Ergebnis gesamt (EUR je Kunde)", fontsize=8)
    ax.set_ylabel("Kunden mit HUS", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _fmt(v, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_husanteil(th: pd.DataFrame) -> plt.Figure:
    """Rangverschiebung nach Gewicht des HUS-Geschäfts."""
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    x = np.arange(len(th))
    ax.bar(x, th["rangverschiebung"], color=MPL_NEGATIV, width=0.6)
    for xi, (_, r) in zip(x, th.iterrows()):
        ax.text(xi, r["rangverschiebung"] + 0.004, _pct(r["rangverschiebung"], 1),
                ha="center", fontsize=7.5, color=MPL_PRIMAER, fontweight="bold")
        ax.text(xi, 0.003, f"n={_fmt(r['kunden'])}", ha="center", va="bottom",
                fontsize=6.5, color="white")
    ax.set_xticks(x, [str(i) for i in th.index], fontsize=8)
    ax.set_xlabel("HUS-Anteil am Gesamtbeitrag des Kunden", fontsize=8)
    ax.set_ylabel("mittlere Rangverschiebung", fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.margins(y=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ======================================================================================
# Befunde
# ======================================================================================
def _befunde(erg: Dict[str, pd.DataFrame]) -> List[str]:
    ew, wv = erg["eckwerte"]["wert"], erg["wertvergleich"]
    dk, rk = erg["delta_kennzahlen"], erg["rangkennzahlen"]
    au, th = erg.get("auswahlueberlappung"), erg.get("treiber_husanteil")
    a: List[str] = []

    a.append(f"Von {_fmt(ew['KFZ-Kunden gesamt'])} KFZ-Kunden halten "
             f"<b>{_pct(ew['Anteil mit HUS-Geschäft'], 1)}</b> zusätzlich HUS-Verträge. "
             f"Nur bei ihnen unterscheiden sich die Sichten; bei reinen KFZ-Kunden sind "
             f"sie identisch. Das HUS-Geschäft macht bei ihnen im Median "
             f"{_pct(ew['HUS-Anteil am Beitrag (Median)'], 0)} des Beitrags aus.")

    if "alle KFZ-Kunden" in rk.index:
        r = rk.loc["alle KFZ-Kunden"]
        a.append(f"Die Reihenfolge bleibt nur grob erhalten: Die Rangkorrelation beträgt "
                 f"<b>{_fmt(r['spearman'], 2)}</b>, {_pct(r['gleiches_dezil'], 1)} der "
                 f"Kunden bleiben im selben Dezil, <b>{_pct(r['sprung_2_plus'], 1)}</b> "
                 f"springen um zwei Dezile oder mehr. Im Mittel verschiebt sich ein Kunde "
                 f"um {_pct(r['perzentilverschiebung_mittel'], 1)} seiner Rangposition.")

    if au is not None and len(au) and 0.10 in au.index:
        r = au.loc[0.10]
        a.append(f"Für eine Auswahl der schlechtesten 10 %: Von "
                 f"{_fmt(r['kunden_je_liste'])} Kunden je Liste stehen "
                 f"<b>{_pct(r['anteil_uebereinstimmung'], 0)}</b> auf beiden. "
                 f"{_fmt(r['nur_kfz_liste'])} Kunden kämen nur über die KFZ-Sicht auf die "
                 f"Liste; über alle Verträge tragen sie "
                 f"{_eur_kurz(r['ergebnis_nur_kfz_liste'])} Ergebnis bei "
                 f"{_eur_kurz(r['beitrag_nur_kfz_liste'])} Beitrag.")

    if "Kunden mit HUS" in dk.index:
        r = dk.loc["Kunden mit HUS"]
        a.append(f"Bei den Kunden mit HUS-Geschäft liegt der Wertunterschied im Median bei "
                 f"{_eur(r['median_eur'])} und {_pct(r['median_marge'], 1)} Marge. Bei "
                 f"<b>{_pct(r['anteil_ueber_100_eur'], 1)}</b> von ihnen beträgt er über "
                 f"100 EUR, bei {_pct(r['anteil_vorzeichenwechsel'], 1)} dreht sich sogar "
                 f"das Vorzeichen des Ergebnisses um.")

    if th is not None and len(th) > 1:
        oben, unten = th["rangverschiebung"].idxmax(), th["rangverschiebung"].idxmin()
        a.append(f"Der Effekt hängt am Gewicht des HUS-Geschäfts: Macht es <b>{oben}</b> "
                 f"des Beitrags aus, verschiebt sich der Rang im Mittel um "
                 f"{_pct(th.loc[oben, 'rangverschiebung'], 1)}, bei <b>{unten}</b> nur um "
                 f"{_pct(th.loc[unten, 'rangverschiebung'], 1)}.")

    a.append(f"Auf Bestandsebene verschiebt sich das Bild ebenfalls: Die Marge liegt in "
             f"der KFZ-Sicht bei {_pct(wv.loc['KFZ-Sicht', 'marge'], 1)}, in der "
             f"Gesamtsicht bei {_pct(wv.loc['Gesamtsicht', 'marge'], 1)}; der Anteil "
             f"Kunden mit negativem Ergebnis geht von "
             f"{_pct(wv.loc['KFZ-Sicht', 'kunden_mit_verlust'], 1)} auf "
             f"{_pct(wv.loc['Gesamtsicht', 'kunden_mit_verlust'], 1)} zurück.")
    return a


# ======================================================================================
# PDF
# ======================================================================================
def erstelle_pkw_report(
    erg: Dict[str, pd.DataFrame],
    pfad: str = "KFZ_Sichtvergleich.pdf",
    titel: str = "KFZ-Sicht gegen Gesamtsicht",
    untertitel: str = "Aggregationsperspektive für KFZ-Use-Cases",
    quelle: str = "",
    verfasser: str = "",
    stand: Optional[str] = None,
) -> str:
    """Erzeugt den PDF-Report zum Sichtvergleich."""
    st = _styles()
    stand = stand or date.today().strftime("%d.%m.%Y")
    k, ew = erg["kunden"], erg["eckwerte"]["wert"]
    wv, dk, rk = erg["wertvergleich"], erg["delta_kennzahlen"], erg["rangkennzahlen"]
    au, th = erg.get("auswahlueberlappung"), erg.get("treiber_husanteil")
    breite = A4[0] - 4.4 * cm

    doc = _Doc(pfad, titel, stand)
    s: List = []

    # ------------------------------------------------------------- Titelkopf
    s.append(Paragraph(titel, st["doc_titel"]))
    s.append(Paragraph(untertitel, st["doc_untertitel"]))
    meta = [f"Stand: {stand}",
            f"Grundgesamtheit: {_fmt(ew['KFZ-Kunden gesamt'])} Kunden mit mindestens "
            f"einem KFZ-Vertrag"]
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

    # ------------------------------------------------ Seite 1: Frage und Befunde
    s.append(Paragraph("Fragestellung", st["h1"]))
    s.append(Paragraph(
        "Für KFZ-Use-Cases lässt sich der Kundenwert auf zwei Arten aggregieren: nur über "
        "die KFZ-Verträge (PKW und KFZ_Rest zusammen) oder über den gesamten "
        "Kompositbestand. Die Frage ist nicht, ob ein Kunde eine Grenze reißt, sondern wie "
        "weit die beiden Sichten denselben Kunden auseinander bewerten - und ob sie zu "
        "derselben Auswahl führen.", st["lead"]))

    rk_alle = rk.loc["alle KFZ-Kunden"] if "alle KFZ-Kunden" in rk.index else None
    s.append(_eckwerte([
        (_fmt(ew["KFZ-Kunden gesamt"]), "KFZ-Kunden"),
        (_pct(ew["Anteil mit HUS-Geschäft"], 1), "davon mit HUS-Verträgen"),
        (_fmt(rk_alle["spearman"], 2) if rk_alle is not None else "n. v.",
         "Rangkorrelation der Sichten"),
        (_pct(rk_alle["sprung_2_plus"], 1) if rk_alle is not None else "n. v.",
         "Sprung ab 2 Dezilen"),
        (_pct(au.loc[0.10, "anteil_uebereinstimmung"], 0)
         if au is not None and 0.10 in au.index else "n. v.",
         "Übereinstimmung Top-10-Auswahl"),
    ], breite, st))
    s.append(Spacer(1, 14))

    s.append(_zwei_bilder(_chart_streuung(k),
                          _chart_ueberlappung(au) if au is not None and len(au) else None,
                          breite))
    s.append(Paragraph(
        "Links: Marge je Kunde in beiden Sichten. Punkte auf der Diagonale sind reine "
        "KFZ-Kunden - bei ihnen sind die Sichten identisch. Rechts: Anteil gemeinsamer "
        "Kunden, wenn nach beiden Sichten jeweils die schlechtesten X Prozent ausgewählt "
        "werden.", st["klein"]))
    s.append(Spacer(1, 12))

    s.append(Paragraph("Befunde", st["h2"]))
    for b in _befunde(erg):
        s.append(Paragraph(b, st["bullet"], bulletText="\u25aa"))
    s.append(PageBreak())

    # ------------------------------------------------ Seite 2: Reihenfolge
    s.append(Paragraph("Die Reihenfolge der Kunden", st["h1"]))
    s.append(Paragraph(
        "Für Auswahl und Priorisierung zählt die Rangfolge. Die Matrix zeigt, wo ein "
        "Kunde landet, wenn man ihn statt nach KFZ-Marge nach Gesamtmarge sortiert - die "
        "Diagonale sind Kunden, die ihr Dezil halten.", st["lead"]))

    dm = erg.get("dezil_matrix")
    if dm is not None and len(dm):
        s.append(_zwei_bilder(_chart_dezilmatrix(dm),
                              _chart_husanteil(th) if th is not None and len(th) else None,
                              breite))
        s.append(Paragraph(
            "Links: Dezilwanderung von der KFZ- zur Gesamtsicht, Zeilenanteile in Prozent. "
            "Rechts: mittlere Rangverschiebung, gestaffelt nach dem HUS-Anteil am "
            "Gesamtbeitrag des Kunden.", st["klein"]))
        s.append(Spacer(1, 14))

    if len(rk):
        zeilen = [[str(i), _fmt(r["kunden"]), _fmt(r["spearman"], 2),
                   _fmt(r.get("kendall", np.nan), 2),
                   _pct(r["perzentilverschiebung_mittel"], 1),
                   _pct(r["gleiches_dezil"], 1), _pct(r["sprung_2_plus"], 1),
                   _pct(r["sprung_3_plus"], 1)]
                  for i, r in rk.iterrows()]
        s.append(_tabelle(["Grundgesamtheit", "Kunden", "Spearman", "Kendall",
                           "Rang-<br/>verschiebung", "gleiches<br/>Dezil",
                           "2 Dezile<br/>und mehr", "3 Dezile<br/>und mehr"], zeilen,
                          [breite * 0.20, breite * 0.11, breite * 0.10, breite * 0.09,
                           breite * 0.15, breite * 0.12, breite * 0.12, breite * 0.11],
                          st))
        s.append(Spacer(1, 6))
        s.append(Paragraph(
            "Die Rangverschiebung ist der mittlere Abstand der Perzentilränge desselben "
            "Kunden zwischen beiden Sichten.", st["klein"]))
        s.append(Spacer(1, 16))

    if au is not None and len(au):
        s.append(Paragraph("Überlappung der Auswahllisten", st["h2"]))
        teiler, einheit = _skala(list(au["ergebnis_nur_kfz_liste"]) +
                                 list(au["beitrag_nur_kfz_liste"]))
        zeilen = [[f"schlechteste {_pct(i, 0)}", _fmt(r["kunden_je_liste"]),
                   _fmt(r["uebereinstimmung"]), _pct(r["anteil_uebereinstimmung"], 1),
                   _fmt(r["nur_kfz_liste"]),
                   _fmt(r["beitrag_nur_kfz_liste"] / teiler, 1),
                   _fmt(r["ergebnis_nur_kfz_liste"] / teiler, 1)]
                  for i, r in au.iterrows()]
        s.append(_tabelle(["Auswahl", "Kunden<br/>je Liste", "in beiden<br/>Listen",
                           "Über-<br/>einstimmung", "nur in<br/>KFZ-Liste",
                           f"deren Beitrag<br/>({einheit})",
                           f"deren Ergebnis<br/>({einheit})"], zeilen,
                          [breite * 0.19, breite * 0.115, breite * 0.115, breite * 0.13,
                           breite * 0.115, breite * 0.165, breite * 0.17], st))
        s.append(Spacer(1, 6))
        s.append(Paragraph(
            "Die letzten beiden Spalten zeigen, welches Volumen hinter den Kunden steht, "
            "die nur die KFZ-Sicht auf die Liste bringt - gemessen über ihren gesamten "
            "Bestand. Ein positives Ergebnis bedeutet: Diese Kunden sind insgesamt "
            "profitabel und würden zu Unrecht angefasst.", st["klein"]))
    s.append(PageBreak())

    # ------------------------------------------------ Seite 3: Werte und Treiber
    s.append(Paragraph("Wie weit die Werte auseinanderliegen", st["h1"]))
    s.append(Paragraph(
        "Hinter der Rangverschiebung stehen konkrete Wertunterschiede. Sie entstehen "
        "ausschließlich beim HUS-Geschäft der Kunden.", st["lead"]))

    teiler, einheit = _skala(list(wv["beitrag"]) + list(wv["profit"]))
    zeilen = [[str(i), _fmt(r["vertraege"]), _fmt(r["beitrag"] / teiler, 1),
               _fmt(r["profit"] / teiler, 1), _pct(r["marge"], 1),
               _eur(r["profit_je_kunde_median"]), _pct(r["marge_je_kunde_median"], 1),
               _pct(r["kunden_mit_verlust"], 1)]
              for i, r in wv.iterrows()]
    s.append(_tabelle(["Sicht", "Verträge", f"Beitrag<br/>({einheit})",
                       f"Ergebnis<br/>({einheit})", "Marge",
                       "Ergebnis je Kunde<br/>(Median)", "Marge je Kunde<br/>(Median)",
                       "Kunden mit<br/>Verlust"], zeilen,
                      [breite * 0.13, breite * 0.11, breite * 0.115, breite * 0.115,
                       breite * 0.09, breite * 0.16, breite * 0.15, breite * 0.13], st))
    s.append(Spacer(1, 14))

    s.append(_bild(_chart_wertdelta(k, breit=True), breite))
    s.append(Paragraph(
        "Wertunterschied je Kunde in Euro (KFZ-Sicht minus Gesamtsicht), nur Kunden mit "
        "HUS-Geschäft, auf das 98-Prozent-Quantil beschnitten. Negative Werte heißen: Der "
        "Kunde steht über den Gesamtbestand besser da.", st["klein"]))
    s.append(Spacer(1, 14))

    if len(dk):
        zeilen = [[str(i), _fmt(r["kunden"]), _eur(r["median_eur"]), _eur(r["mittel_eur"]),
                   _pct(r["median_marge"], 1), _pct(r["anteil_ueber_100_eur"], 1),
                   _pct(r["anteil_ueber_10_pp"], 1), _pct(r["anteil_vorzeichenwechsel"], 1)]
                  for i, r in dk.iterrows()]
        s.append(_tabelle(["Wertunterschied", "Kunden", "Median", "Mittel",
                           "Median in<br/>Margenpunkten", "über<br/>100 EUR",
                           "über<br/>10 Punkte", "Vorzeichen<br/>dreht"], zeilen,
                          [breite * 0.18, breite * 0.10, breite * 0.11, breite * 0.11,
                           breite * 0.16, breite * 0.115, breite * 0.115, breite * 0.13],
                          st))
        s.append(Spacer(1, 16))

    tz = erg.get("treiber_zweig")
    if tz is not None and len(tz):
        s.append(Paragraph("Welche HUS-Zweige den Unterschied tragen", st["h2"]))
        zeilen = [[str(i)[:24], _fmt(r["kunden"]), _pct(r["durchdringung"], 0),
                   _eur(r["profit_je_halter"]), _pct(r["marge"], 1),
                   _pct(r["rangverschiebung_halter"], 1)]
                  for i, r in tz.iterrows()]
        s.append(_tabelle(["HUS-Zweig", "Kunden", "Durch-<br/>dringung",
                           "Ergebnis je<br/>Halter", "Marge<br/>des Zweigs",
                           "Rangverschiebung<br/>der Halter"], zeilen,
                          [breite * 0.20, breite * 0.12, breite * 0.14, breite * 0.16,
                           breite * 0.16, breite * 0.22], st))
        s.append(Spacer(1, 6))
        s.append(Paragraph(
            "Durchdringung = Anteil der Kunden mit HUS-Geschäft, die diesen Zweig halten. "
            "Ergebnis je Halter zeigt, wie viel Ergebnis der Zweig zum Gesamtwert dieser "
            "Kunden beisteuert.", st["klein"]))

    doc.build(s)
    return pfad
