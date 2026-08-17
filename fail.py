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
    erg["bereich_mix"]          # Ebene 1: PKW vs. HUS
    erg["zweig_mix"]            # Ebene 2: PKW-Sparten und HUS-Produkte (vergleichbar)
    erg["zweig_je_bereich"]     # Zweige INNERHALB eines Bereichs
    erg["vertraege"]            # Vertragsebene: Positionen je ve_id zusammengefasst
    erg["bereichs_kombination"] # nur PKW / nur HUS / beides je Segment
    erg["vertragsprofile_pkw"]  # Deckungskombinationen innerhalb eines PKW-Vertrags
    erg["beitragsverwendung"]   # Schaden-/Kostenquote: wodurch wird ein Segment negativ?
    erg["sanierungsbedarf"]     # je negativem Kunden: wie viele Verträge treiben den Verlust?
    ...

Fachliche Hierarchie (wird von ergaenze_ebenen() erzeugt):
    Ebene 1  ve_bereich : PKW vs. HUS - alles, was nicht PKW ist, gehört zu HUS.
    Ebene 2  ve_zweig   : innerhalb PKW die ve_sparte, innerhalb HUS das ve_produkt.
Erst dadurch liegen die verglichenen Einheiten auf derselben Ebene.

Doppelte ve_ids sind normal: Ein Vertrag kann mehrere Positionen (Deckungen) haben,
z. B. Haftpflicht und Teilkasko in einem PKW-Vertrag. Beiträge und Ergebnisse werden
über alle Positionen summiert, Verträge dedupliziert auf ve_id gezählt.
Andere PKW-Produktschlüssel: analysiere_kundenwert(df, pkw_produkte=("PKW", "KRAD")).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

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
    """Typen sichern, fehlende Strings auffüllen, Struktur der Daten melden."""
    d = df.copy() if kopie else df

    for c in NUM_SPALTEN:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in STR_SPALTEN:
        if c in d.columns:
            d[c] = d[c].astype("string").fillna("(unbekannt)").str.strip()

    if VERTRAG in d.columns:
        n_zeilen, n_vertraege = len(d), d[VERTRAG].nunique()
        if n_zeilen > n_vertraege:
            print(f"[Struktur] {n_zeilen:,} Zeilen (Positionen) auf {n_vertraege:,} Verträge "
                  f"(ve_id), im Schnitt {n_zeilen / n_vertraege:.2f} Positionen je Vertrag. "
                  f"Vertragszählungen erfolgen dedupliziert auf ve_id.")

    fehlend = d[BEITRAG].isna().sum() + d[PROFIT].isna().sum()
    if fehlend:
        print(f"[Hinweis] {int(fehlend):,} fehlende Werte in Beitrag/Profit -> als 0 gewertet.")
        d[[BEITRAG, PROFIT]] = d[[BEITRAG, PROFIT]].fillna(0)

    return d


def vertragsaggregat(df: pd.DataFrame, mit_profil: bool = True) -> pd.DataFrame:
    """
    Fasst die Positionen (Zeilen) einer ve_id zu EINEM Vertrag zusammen.

    Ein PKW-Vertrag kann mehrere Positionen enthalten (z. B. Teilkasko und
    Haftpflicht); Beiträge und Ergebnisse werden summiert, Zählungen erfolgen
    anschließend auf Vertragsebene.

    Zusatzspalten:
      n_positionen : Anzahl Zeilen des Vertrags
      profil       : Kombination der Zweige im Vertrag, z. B. 'Haftpflicht + Teilkasko'
      bereich      : PKW/HUS; 'gemischt', falls ein Vertrag beide Bereiche berührt
    """
    hat_bereich = BEREICH in df.columns
    agg_spec = dict(
        n_positionen=(BEITRAG, "size"),
        beitrag=(BEITRAG, "sum"),
        profit=(PROFIT, "sum"),
    )
    if SCHADEN in df.columns:
        agg_spec["schaden"] = (SCHADEN, "sum")
    if KOSTEN in df.columns:
        agg_spec["kosten"] = (KOSTEN, "sum")
    if hat_bereich:
        agg_spec["n_bereiche"] = (BEREICH, "nunique")
        agg_spec["n_zweige"] = (ZWEIG, "nunique")

    v = df.groupby([ID, VERTRAG], observed=True).agg(**agg_spec)

    erster = df.drop_duplicates([ID, VERTRAG]).set_index([ID, VERTRAG])
    spalten = [PRODUKT, GESELLSCHAFT] + ([BEREICH, ZWEIG] if hat_bereich else [])
    v = v.join(erster[[c for c in spalten if c in erster.columns]])

    if hat_bereich:
        v[BEREICH] = np.where(v["n_bereiche"] > 1, "gemischt", v[BEREICH])
        if mit_profil:
            z = df[[ID, VERTRAG, ZWEIG]].drop_duplicates().sort_values([ID, VERTRAG, ZWEIG])
            v["profil"] = z.groupby([ID, VERTRAG], observed=True)[ZWEIG].agg(" + ".join)
        gemischt = int((v[BEREICH] == "gemischt").sum())
        if gemischt:
            print(f"[Hinweis] {gemischt:,} Verträge berühren PKW und HUS gleichzeitig und "
                  f"laufen als 'gemischt'.")

    v["marge"] = np.where(v["beitrag"] > 0, v["profit"] / v["beitrag"], np.nan)
    return v.reset_index()


# ======================================================================================
# 2) Kundenaggregat + Klassifikation
# ======================================================================================
def kunden_aggregat(df: pd.DataFrame, schwelle: float = 0.20,
                    vertraege: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregiert auf vn_partner_id und klassifiziert in 'negativ' / 'positiv'.

    Regel (einzige):
        negativ  <=>  sum(ve_profit) < 0
                      UND  |sum(ve_profit)| >= schwelle * sum(ve_bestandsjahresnettobeitrag)

    Beträge werden über alle Zeilen summiert, Verträge dagegen dedupliziert auf ve_id
    gezählt. `n_positionen` gibt die Zeilenzahl aus, `n_vertraege` die echte
    Vertragsanzahl.
    """
    d = df
    v = vertragsaggregat(d) if vertraege is None else vertraege

    agg = d.groupby(ID, observed=True).agg(
        n_positionen=(VERTRAG, "size"),
        n_vertraege=(VERTRAG, "nunique"),
        beitrag_sum=(BEITRAG, "sum"),
        profit_sum=(PROFIT, "sum"),
        schaden_sum=(SCHADEN, "sum") if SCHADEN in d.columns else (BEITRAG, "size"),
        kosten_sum=(KOSTEN, "sum") if KOSTEN in d.columns else (BEITRAG, "size"),
        n_sparten=(SPARTE, "nunique"),
        n_produkte=(PRODUKT, "nunique"),
        n_gesellschaften=(GESELLSCHAFT, "nunique"),
    )

    agg["positionen_je_vertrag"] = agg["n_positionen"] / agg["n_vertraege"]
    agg["beitrag_je_vertrag"] = agg["beitrag_sum"] / agg["n_vertraege"]
    agg["profit_je_vertrag"] = agg["profit_sum"] / agg["n_vertraege"]
    agg["marge"] = np.where(agg["beitrag_sum"] > 0,
                            agg["profit_sum"] / agg["beitrag_sum"], np.nan)
    agg["cr_gesamt"] = np.where(agg["beitrag_sum"] > 0,
                                (agg["schaden_sum"] + agg["kosten_sum"]) / agg["beitrag_sum"],
                                np.nan)

    # Defizitäre Verträge auf VERTRAGSEBENE (nicht je Position)
    agg["anteil_verlustvertraege"] = (v.assign(_neg=v["profit"] < 0)
                                      .groupby(ID, observed=True)["_neg"].mean())

    # Vertragszahl je Bereich - PKW und HUS getrennt
    if BEREICH in v.columns:
        je_bereich = v.pivot_table(index=ID, columns=BEREICH, values=VERTRAG,
                                   aggfunc="nunique", observed=True).fillna(0)
        for b in ("PKW", "HUS"):
            agg[f"n_vertraege_{b.lower()}"] = je_bereich[b] if b in je_bereich.columns else 0
        beitrag_bereich = v.pivot_table(index=ID, columns=BEREICH, values="beitrag",
                                        aggfunc="sum", observed=True).fillna(0)
        for b in ("PKW", "HUS"):
            agg[f"beitrag_{b.lower()}"] = (beitrag_bereich[b]
                                           if b in beitrag_bereich.columns else 0.0)
        agg["anteil_pkw_vertraege"] = agg["n_vertraege_pkw"] / agg["n_vertraege"]
        agg["bereichs_profil"] = np.select(
            [(agg["n_vertraege_pkw"] > 0) & (agg["n_vertraege_hus"] > 0),
             agg["n_vertraege_pkw"] > 0,
             agg["n_vertraege_hus"] > 0],
            ["PKW und HUS", "nur PKW", "nur HUS"], default="sonstige")

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
            "Positionen (Zeilen) gesamt": k["n_positionen"].sum(),
            "Positionen je Vertrag (Mittel)": k["positionen_je_vertrag"].mean(),
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
        if "n_vertraege_pkw" in k.columns:
            werte.update({
                "PKW-Verträge je Kunde (Mittel)": k["n_vertraege_pkw"].mean(),
                "HUS-Verträge je Kunde (Mittel)": k["n_vertraege_hus"].mean(),
                "Anteil Kunden mit PKW-Vertrag": (k["n_vertraege_pkw"] > 0).mean(),
                "Anteil Kunden mit HUS-Vertrag": (k["n_vertraege_hus"] > 0).mean(),
                "Anteil Kunden mit PKW und HUS": ((k["n_vertraege_pkw"] > 0) &
                                                  (k["n_vertraege_hus"] > 0)).mean(),
                "PKW-Anteil an den Verträgen (Mittel)": k["anteil_pkw_vertraege"].mean(),
            })
        zeilen[g] = {kk: float(vv) for kk, vv in werte.items()}

    out = pd.DataFrame(zeilen)
    out["differenz_neg_minus_pos"] = out["negativ"] - out["positiv"]
    out["verhaeltnis_neg_zu_pos"] = np.where(out["positiv"] != 0,
                                             out["negativ"] / out["positiv"], np.nan)
    out.index.name = "kennzahl"
    return out


# ======================================================================================
# 4) Ebenen und Mix-Analysen
#
#    Fachliche Hierarchie:
#      Ebene 1  ve_bereich : PKW vs. HUS. Alles, was nicht PKW ist, gehoert zu HUS.
#      Ebene 2  ve_zweig   : innerhalb PKW die ve_sparte (Vollkasko, Teilkasko, ...),
#                            innerhalb HUS das ve_produkt (PH, RS, UNFALL, ...).
#    Erst dadurch liegen die verglichenen Einheiten auf derselben Ebene - vorher
#    stand eine PKW-Sparte neben einem HUS-Produkt.
# ======================================================================================
BEREICH = "ve_bereich"
ZWEIG = "ve_zweig"


def ergaenze_ebenen(df: pd.DataFrame, pkw_produkte: Sequence[str] = ("PKW",)) -> pd.DataFrame:
    """Fügt ve_bereich (PKW/HUS) und ve_zweig (vergleichbare Unterebene) hinzu."""
    d = df.copy()
    ziel = {str(p).strip().upper() for p in pkw_produkte}
    ist_pkw = d[PRODUKT].astype(str).str.strip().str.upper().isin(ziel)
    if not ist_pkw.any():
        print(f"[Hinweis] Kein Vertrag mit ve_produkt in {sorted(ziel)} gefunden - "
              f"der Bereich PKW bleibt leer. Parameter pkw_produkte prüfen.")
    d[BEREICH] = np.where(ist_pkw, "PKW", "HUS")
    d[ZWEIG] = np.where(ist_pkw, d[SPARTE].astype(str), d[PRODUKT].astype(str))
    return d


def ebenen_uebersicht(df: pd.DataFrame) -> pd.DataFrame:
    """Welche Zweige gehören zu welchem Bereich, und aus welcher Spalte stammen sie?"""
    g = df.groupby([BEREICH, ZWEIG], observed=True).agg(
        vertraege=(BEITRAG, "size"),
        kunden=(ID, "nunique"),
        beitrag=(BEITRAG, "sum"),
    )
    g["herkunft"] = np.where(g.index.get_level_values(0) == "PKW", "ve_sparte", "ve_produkt")
    g["anteil_vertraege"] = g["vertraege"] / g["vertraege"].sum()
    return g.sort_values([BEREICH, "vertraege"], ascending=[True, False])


def mix_analyse(
    df: pd.DataFrame,
    kunden: pd.DataFrame,
    spalten: Union[str, Sequence[str]] = ZWEIG,
    sortiere_nach: str = "anteil_vertraege_negativ",
    min_vertraege: int = 0,
) -> pd.DataFrame:
    """
    Zusammensetzung der Kundenbestände nach Kategorie, bezogen auf den GESAMTEN
    Vertragsbestand des Kunden. Sinnvoll für ve_bereich, ve_zweig, ve_gesellschaft.
    Für die Aufteilung innerhalb eines Bereichs `unter_mix()` verwenden.

    Geliefert werden je Gruppe (negativ/positiv):
      - anteil_vertraege_*  : mittlerer kundenindividueller Anteil dieser Kategorie
                              an allen Verträgen des Kunden ("im Mittel zusammengesetzt aus")
      - anteil_beitrag_*    : dito, aber beitragsgewichtet
      - penetration_*       : Anteil der Kunden der Gruppe mit mind. 1 Vertrag der Kategorie
      - vertraege_*, beitrag_*, profit_*, marge_*
      - delta_*             : negativ minus positiv (in Prozentpunkten)

    Verträge werden auf ve_id dedupliziert gezählt. Berührt ein Vertrag mehrere
    Kategorien (z. B. Teilkasko und Haftpflicht in einem PKW-Vertrag), zählt er in
    jeder davon; die kundenindividuellen Anteile summieren sich dann auf über 100 %.
    Auf Bereichsebene (PKW/HUS) tritt das praktisch nicht auf.
    """
    keys: List[str] = [spalten] if isinstance(spalten, str) else list(spalten)

    d = df[[ID, VERTRAG, *keys, BEITRAG, PROFIT]].copy()
    d["wert_flag"] = d[ID].map(kunden["wert_flag"])
    d = d[d["wert_flag"].notna()]

    # Kunde x Kategorie: Verträge dedupliziert, Beträge summiert
    kk = d.groupby([ID, *keys], observed=True).agg(
        n=(VERTRAG, "nunique"), beitrag=(BEITRAG, "sum"), profit=(PROFIT, "sum")
    ).reset_index()

    ges = d.groupby(ID, observed=True).agg(n_ges=(VERTRAG, "nunique"),
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


def unter_mix(
    df: pd.DataFrame,
    kunden: pd.DataFrame,
    ober: str = BEREICH,
    unter: str = ZWEIG,
    min_vertraege: int = 0,
) -> pd.DataFrame:
    """
    Aufteilung der Unterebene INNERHALB jeder Oberebene (Standard: Zweige je Bereich).

    Die Anteile beziehen sich auf die Verträge des jeweiligen Bereichs im jeweiligen
    Segment und addieren sich je Bereich und Segment auf 100 %. Damit wird
    'Vollkasko vs. Teilkasko innerhalb PKW' verglichen und nicht Vollkasko gegen PH.
    """
    d = df[[ID, VERTRAG, ober, unter, BEITRAG, PROFIT]].copy()
    d["wert_flag"] = d[ID].map(kunden["wert_flag"])
    d = d[d["wert_flag"].notna()]
    if d.empty:
        return pd.DataFrame()

    g = d.groupby(["wert_flag", ober, unter], observed=True).agg(
        vertraege=(VERTRAG, "nunique"), kunden_mit=(ID, "nunique"),
        beitrag=(BEITRAG, "sum"), profit=(PROFIT, "sum")).reset_index()
    oben = d.groupby(["wert_flag", ober], observed=True).agg(
        vertraege_ober=(VERTRAG, "nunique")).reset_index()
    g = g.merge(oben, on=["wert_flag", ober], how="left")

    g["anteil_im_bereich"] = g["vertraege"] / g["vertraege_ober"]
    g["beitrag_je_vertrag"] = g["beitrag"] / g["vertraege"]
    g["marge"] = np.where(g["beitrag"] > 0, g["profit"] / g["beitrag"], np.nan)

    metriken = ["anteil_im_bereich", "vertraege", "kunden_mit",
                "beitrag", "profit", "beitrag_je_vertrag", "marge"]
    wide = g.pivot_table(index=[ober, unter], columns="wert_flag",
                         values=metriken, observed=True)
    wide.columns = [f"{m}_{f}" for m, f in wide.columns]
    wide = wide.reindex(columns=[f"{m}_{f}" for m in metriken for f in ("negativ", "positiv")])
    wide = wide.fillna({c: 0 for c in wide.columns if not c.startswith("marge")})

    for m in ["anteil_im_bereich", "beitrag_je_vertrag", "marge"]:
        wide[f"delta_{m}"] = wide[f"{m}_negativ"] - wide[f"{m}_positiv"]
    wide["vertraege_gesamt"] = wide["vertraege_negativ"] + wide["vertraege_positiv"]

    if min_vertraege:
        wide = wide[wide["vertraege_gesamt"] >= min_vertraege]
    return wide.sort_values([ober, "vertraege_gesamt"], ascending=[True, False])


def bereichs_kombination(kunden: pd.DataFrame) -> pd.DataFrame:
    """
    Hält der Kunde nur PKW, nur HUS oder beides - und wie unterscheidet sich das
    zwischen den Segmenten? Beantwortet die Frage, ob Verbundkunden werthaltiger sind.
    """
    if "bereichs_profil" not in kunden.columns:
        return pd.DataFrame()
    g = kunden.groupby(["bereichs_profil", "wert_flag"], observed=True).agg(
        kunden=("beitrag_sum", "size"),
        vertraege=("n_vertraege", "sum"),
        beitrag=("beitrag_sum", "sum"),
        profit=("profit_sum", "sum"),
        vertraege_je_kunde=("n_vertraege", "mean"),
        beitrag_je_kunde=("beitrag_sum", "mean"),
    ).reset_index()
    g["marge"] = np.where(g["beitrag"] > 0, g["profit"] / g["beitrag"], np.nan)

    metriken = ["kunden", "vertraege", "beitrag", "profit", "vertraege_je_kunde",
                "beitrag_je_kunde", "marge"]
    wide = g.pivot_table(index="bereichs_profil", columns="wert_flag", values=metriken,
                         observed=True)
    wide.columns = [f"{m}_{f}" for m, f in wide.columns]
    wide = wide.reindex(columns=[f"{m}_{f}" for m in metriken
                                 for f in ("negativ", "positiv")])
    wide = wide.fillna({c: 0 for c in wide.columns if not c.startswith("marge")})
    wide["kunden_gesamt"] = wide["kunden_negativ"] + wide["kunden_positiv"]
    # Anteil negativer Kunden innerhalb der Kombination - die zentrale Vergleichszahl
    wide["negativ_quote"] = wide["kunden_negativ"] / wide["kunden_gesamt"]
    wide["anteil_kunden_negativ"] = wide["kunden_negativ"] / wide["kunden_negativ"].sum()
    wide["anteil_kunden_positiv"] = wide["kunden_positiv"] / wide["kunden_positiv"].sum()
    wide["delta_anteil_kunden"] = wide["anteil_kunden_negativ"] - wide["anteil_kunden_positiv"]
    return wide.sort_values("kunden_gesamt", ascending=False)


def vertragsprofile(vertraege: pd.DataFrame, kunden: pd.DataFrame,
                    bereich: Optional[str] = None, top_n: int = 12,
                    min_vertraege: int = 0) -> pd.DataFrame:
    """
    Deckungsprofile auf Vertragsebene: Welche Zweig-Kombinationen stecken in einem
    Vertrag (z. B. 'Haftpflicht + Teilkasko'), und wie schlagen sie sich?

    bereich='PKW' beschränkt auf PKW-Verträge, wo Mehrfachdeckungen typisch sind.
    """
    if vertraege is None or "profil" not in vertraege.columns:
        return pd.DataFrame()
    v = vertraege.copy()
    v["wert_flag"] = v[ID].map(kunden["wert_flag"])
    v = v[v["wert_flag"].notna()]
    if bereich is not None and BEREICH in v.columns:
        v = v[v[BEREICH] == bereich]
    if v.empty:
        return pd.DataFrame()

    g = v.groupby(["wert_flag", "profil"], observed=True).agg(
        vertraege=("beitrag", "size"), kunden_mit=(ID, "nunique"),
        beitrag=("beitrag", "sum"), profit=("profit", "sum"),
        positionen=("n_positionen", "mean")).reset_index()
    ges = v.groupby("wert_flag", observed=True).size().rename("vertraege_segment")
    g = g.merge(ges, left_on="wert_flag", right_index=True, how="left")
    g["anteil_vertraege"] = g["vertraege"] / g["vertraege_segment"]
    g["beitrag_je_vertrag"] = g["beitrag"] / g["vertraege"]
    g["marge"] = np.where(g["beitrag"] > 0, g["profit"] / g["beitrag"], np.nan)

    metriken = ["anteil_vertraege", "vertraege", "kunden_mit", "beitrag", "profit",
                "beitrag_je_vertrag", "marge", "positionen"]
    wide = g.pivot_table(index="profil", columns="wert_flag", values=metriken, observed=True)
    wide.columns = [f"{m}_{f}" for m, f in wide.columns]
    wide = wide.reindex(columns=[f"{m}_{f}" for m in metriken
                                 for f in ("negativ", "positiv")])
    wide = wide.fillna({c: 0 for c in wide.columns if not c.startswith("marge")})
    for m in ["anteil_vertraege", "beitrag_je_vertrag", "marge"]:
        wide[f"delta_{m}"] = wide[f"{m}_negativ"] - wide[f"{m}_positiv"]
    wide["vertraege_gesamt"] = wide["vertraege_negativ"] + wide["vertraege_positiv"]
    if min_vertraege:
        wide = wide[wide["vertraege_gesamt"] >= min_vertraege]
    return wide.sort_values("vertraege_gesamt", ascending=False).head(top_n)


# ======================================================================================
# 5) Vertragsebene
# ======================================================================================
def vertragsebene(vertraege: pd.DataFrame, kunden: pd.DataFrame,
                  ebene: Optional[str] = None) -> pd.DataFrame:
    """
    Durchschnittliche Kennzahlen je VERTRAG (ve_id, Positionen bereits summiert),
    optional zusätzlich je Ebene (z. B. ve_bereich oder ve_zweig).

    Erwartet das Ergebnis von `vertragsaggregat()`. Ein Vertrag mit mehreren
    Positionen zählt hier einmal; sein Beitrag ist die Summe seiner Positionen.
    """
    v = vertraege.copy()
    v["wert_flag"] = v[ID].map(kunden["wert_flag"])
    v = v[v["wert_flag"].notna()]

    keys = ["wert_flag", ebene] if ebene else ["wert_flag"]
    out = v.groupby(keys, observed=True).agg(
        vertraege=("beitrag", "size"),
        positionen_je_vertrag=("n_positionen", "mean"),
        beitrag_mittel=("beitrag", "mean"),
        beitrag_median=("beitrag", "median"),
        beitrag_summe=("beitrag", "sum"),
        profit_mittel=("profit", "mean"),
        profit_median=("profit", "median"),
        profit_summe=("profit", "sum"),
        anteil_verlustvertraege=("profit", lambda s: float((s < 0).mean())),
    )
    out["marge_gewichtet"] = np.where(out["beitrag_summe"] > 0,
                                      out["profit_summe"] / out["beitrag_summe"], np.nan)
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
                   ebene: Union[str, Sequence[str]] = PRODUKT) -> pd.DataFrame:
    """
    Welche Einheiten erzeugen den Verlust innerhalb der negativen Kunden?
    ebene=PRODUKT liefert die obere Ebene, ebene=[PRODUKT, SPARTE] die Feingliederung.
    """
    keys = [ebene] if isinstance(ebene, str) else list(ebene)
    d = df.copy()
    d["wert_flag"] = d[ID].map(kunden["wert_flag"])
    neg = d[d["wert_flag"] == "negativ"]
    if neg.empty:
        return pd.DataFrame()
    out = neg.groupby(keys if len(keys) > 1 else keys[0], observed=True).agg(
        vertraege=(VERTRAG, "nunique"),
        positionen=(BEITRAG, "size"),
        beitrag=(BEITRAG, "sum"),
        profit=(PROFIT, "sum"),
    )
    out["marge"] = np.where(out["beitrag"] > 0, out["profit"] / out["beitrag"], np.nan)
    verlust = out.loc[out["profit"] < 0, "profit"].sum()
    out["anteil_am_gesamtverlust"] = np.where(out["profit"] < 0,
                                              out["profit"] / verlust if verlust else np.nan, 0.0)
    return out.sort_values("profit").head(top_n)


def beitragsverwendung(df: pd.DataFrame, kunden: pd.DataFrame) -> pd.DataFrame:
    """
    Wofür wird der Beitrag verbraucht? Schaden- und Kostenquote je Segment,
    jeweils als Anteil am Beitrag. Zeigt unmittelbar, wodurch ein Segment negativ wird.
    """
    d = df.copy()
    d["wert_flag"] = d[ID].map(kunden["wert_flag"])
    d = d[d["wert_flag"].notna()]
    g = d.groupby("wert_flag", observed=True).agg(
        beitrag=(BEITRAG, "sum"),
        schaden=(SCHADEN, "sum") if SCHADEN in d.columns else (BEITRAG, "size"),
        kosten=(KOSTEN, "sum") if KOSTEN in d.columns else (BEITRAG, "size"),
        profit=(PROFIT, "sum"),
    )
    g["schadenquote"] = g["schaden"] / g["beitrag"]
    g["kostenquote"] = g["kosten"] / g["beitrag"]
    g["combined_ratio"] = g["schadenquote"] + g["kostenquote"]
    g["ergebnisquote"] = g["profit"] / g["beitrag"]
    return g


def sanierungsbedarf(vertraege: pd.DataFrame, kunden: pd.DataFrame,
                     schwelle: float = 0.20) -> pd.DataFrame:
    """
    Je negativem Kunden: Wie viele der schlechtesten VERTRÄGE müssten aus dem
    Bestand fallen (oder saniert werden), damit der Kunde die Negativ-Grenze verlässt?

    Erwartet das Ergebnis von `vertragsaggregat()` - ein Vertrag mit mehreren
    Positionen wird als Ganzes betrachtet, nicht positionsweise zerlegt.

    Ergebnis je Kunde:
      k_noetig                : Anzahl der schlechtesten Verträge (NaN = kein Teilbestand
                                des Kunden erfüllt die Grenze)
      anteil_vertraege_noetig : k_noetig / Vertragsanzahl
      anteil_verlust_top1     : Anteil des schlechtesten Vertrags am Kundenverlust
    """
    neg = kunden.index[kunden["wert_flag"] == "negativ"]
    d = vertraege[vertraege[ID].isin(neg)][[ID, VERTRAG, "beitrag", "profit"]].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values([ID, "profit"])
    g = d.groupby(ID, observed=True)
    d["k"] = g.cumcount() + 1
    d["cum_profit"] = g["profit"].cumsum()
    d["cum_beitrag"] = g["beitrag"].cumsum()
    d = d.merge(kunden.loc[neg, ["beitrag_sum", "profit_sum", "n_vertraege"]],
                left_on=ID, right_index=True, how="left")

    rest_profit = d["profit_sum"] - d["cum_profit"]
    rest_beitrag = d["beitrag_sum"] - d["cum_beitrag"]
    # Restbestand erfüllt die Negativ-Definition nicht mehr
    ok = (rest_beitrag > 0) & ~((rest_profit < 0) &
                                (rest_profit.abs() >= schwelle * rest_beitrag))
    k_noetig = d.loc[ok].groupby(ID, observed=True)["k"].min()

    top1 = g["profit"].min()
    out = kunden.loc[neg, ["n_vertraege", "beitrag_sum", "profit_sum", "marge"]].copy()
    out["k_noetig"] = k_noetig
    out["anteil_vertraege_noetig"] = out["k_noetig"] / out["n_vertraege"]
    out["anteil_verlust_top1"] = np.where(out["profit_sum"] < 0,
                                          top1.reindex(out.index) / out["profit_sum"], np.nan)
    return out


def sanierung_verteilung(san: pd.DataFrame) -> pd.DataFrame:
    """Verdichtung von `sanierungsbedarf`: Wie viele Kunden brauchen wie viele Eingriffe?"""
    if san is None or san.empty:
        return pd.DataFrame()
    k = san["k_noetig"]
    einzel = san["n_vertraege"] == 1
    klasse = pd.Series("kein Teilbestand reicht", index=san.index, dtype=object)
    klasse[k == 1] = "1 von mehreren Verträgen"
    klasse[k == 2] = "2 Verträge"
    klasse[k >= 3] = "3 oder mehr Verträge"
    klasse[einzel] = "Einvertragskunde"
    reihenfolge = ["Einvertragskunde", "1 von mehreren Verträgen", "2 Verträge",
                   "3 oder mehr Verträge", "kein Teilbestand reicht"]
    g = san.assign(klasse=klasse).groupby("klasse", observed=True).agg(
        kunden=("profit_sum", "size"),
        verlust=("profit_sum", "sum"),
        beitrag=("beitrag_sum", "sum"),
    ).reindex(reihenfolge).fillna(0)
    g["anteil_kunden"] = g["kunden"] / len(san)
    g["anteil_verlust"] = g["verlust"] / san["profit_sum"].sum()
    return g


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
    if not zeilen:
        return pd.DataFrame()
    return pd.DataFrame(zeilen).set_index("merkmal")


# ======================================================================================
# 7) Orchestrierung
# ======================================================================================
def analysiere_kundenwert(
    df: pd.DataFrame,
    schwelle: float = 0.20,
    pkw_produkte: Sequence[str] = ("PKW",),
    min_vertraege_zweig: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Führt die komplette Analyse aus und gibt ein Dict von DataFrames zurück."""
    d = ergaenze_ebenen(bereinige(df), pkw_produkte=pkw_produkte)
    vert = vertragsaggregat(d)
    kunden = kunden_aggregat(d, schwelle=schwelle, vertraege=vert)
    san = sanierungsbedarf(vert, kunden, schwelle=schwelle)

    erg: Dict[str, pd.DataFrame] = {
        "kunden": kunden,
        "vertraege": vert,
        "kennzahlen": kennzahlen_vergleich(kunden),
        # Ebene 1: PKW vs. HUS
        "bereich_mix": mix_analyse(d, kunden, BEREICH),
        "bereichs_kombination": bereichs_kombination(kunden),
        # Ebene 2: vergleichbare Zweige (PKW-Sparten und HUS-Produkte)
        "zweig_mix": mix_analyse(d, kunden, ZWEIG, min_vertraege=min_vertraege_zweig),
        "zweig_je_bereich": unter_mix(d, kunden, BEREICH, ZWEIG,
                                      min_vertraege=min_vertraege_zweig),
        "ebenen_uebersicht": ebenen_uebersicht(d),
        # Vertragsebene: Deckungskombinationen innerhalb einer ve_id
        "vertragsprofile_pkw": vertragsprofile(vert, kunden, bereich="PKW"),
        "gesellschaft_mix": mix_analyse(d, kunden, GESELLSCHAFT),
        "vertragsebene": vertragsebene(vert, kunden),
        "vertragsebene_je_bereich": vertragsebene(vert, kunden, ebene=BEREICH),
        "vertragsebene_je_zweig": vertragsebene(vert, kunden, ebene=ZWEIG),
        "verteilung_vertragsanzahl": verteilung_vertragsanzahl(kunden),
        "margen_verteilung": margen_verteilung(kunden),
        # Anatomie der negativen Kunden
        "beitragsverwendung": beitragsverwendung(d, kunden),
        "sanierungsbedarf": san,
        "sanierung_verteilung": sanierung_verteilung(san),
        "verlusttreiber_zweig": verlusttreiber(d, kunden, ebene=ZWEIG, top_n=25),
        "verlusttreiber_bereich_zweig": verlusttreiber(d, kunden, ebene=[BEREICH, ZWEIG],
                                                       top_n=25),
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

    print("\n--- 2) Ebene 1: Bereiche (PKW vs. HUS) ---")
    print(erg["bereich_mix"][["anteil_vertraege_negativ", "anteil_vertraege_positiv",
                              "delta_anteil_vertraege", "penetration_negativ",
                              "penetration_positiv", "marge_negativ", "marge_positiv"]])

    print("\n--- 3) Ebene 2: Zweige (PKW-Sparten und HUS-Produkte) ---")
    print(erg["zweig_mix"][["anteil_vertraege_negativ", "anteil_vertraege_positiv",
                            "delta_anteil_vertraege", "beitrag_je_vertrag_negativ",
                            "marge_negativ", "marge_positiv"]].head(top))

    print("\n--- 3b) Zweige INNERHALB des Bereichs ---")
    print(erg["zweig_je_bereich"][["anteil_im_bereich_negativ", "anteil_im_bereich_positiv",
                                   "delta_anteil_im_bereich", "marge_negativ",
                                   "marge_positiv"]].head(2 * top))

    print("\n--- 3c) Verbund: nur PKW / nur HUS / beides ---")
    bk = erg.get("bereichs_kombination")
    if bk is not None and len(bk):
        print(bk[["kunden_negativ", "kunden_positiv", "negativ_quote",
                  "vertraege_je_kunde_negativ", "vertraege_je_kunde_positiv",
                  "marge_negativ", "marge_positiv"]])

    print("\n--- 3d) Deckungsprofile innerhalb der PKW-Verträge ---")
    vp = erg.get("vertragsprofile_pkw")
    if vp is not None and len(vp):
        print(vp[["vertraege_negativ", "vertraege_positiv", "anteil_vertraege_negativ",
                  "anteil_vertraege_positiv", "beitrag_je_vertrag_negativ",
                  "marge_negativ", "marge_positiv"]])

    print("\n--- 4) Vertragsebene (ve_id dedupliziert) ---")
    print(erg["vertragsebene"])
    print(erg["vertragsebene_je_bereich"])

    print("\n--- 5) Verteilung Vertragsanzahl ---")
    print(erg["verteilung_vertragsanzahl"])

    print("\n--- 6) Wie werden Kunden negativ? Beitragsverwendung ---")
    print(erg["beitragsverwendung"][["schadenquote", "kostenquote", "combined_ratio",
                                     "ergebnisquote"]])

    print("\n--- 6b) Sanierungsbedarf je negativem Kunden ---")
    print(erg["sanierung_verteilung"])

    print("\n--- 7) Größte Verlusttreiber (Zweige, negative Kunden) ---")
    print(erg["verlusttreiber_zweig"].head(top))

    print("\n--- 8) Verlustkonzentration ---")
    print(erg["konzentration"])

    print("\n--- 9) Sensitivität der Schwelle ---")
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






"""
kundenwert_report.py
====================

Erzeugt aus dem Ergebnis-Dict von `analysiere_kundenwert()` eine präsentationsreife
PDF-Vorlage für die Vorstandsebene.

Verwendung:
    from kundenwert_analyse import analysiere_kundenwert
    from kundenwert_report import erstelle_pdf

    erg = analysiere_kundenwert(df)
    erstelle_pdf(erg, "Kundenwertanalyse.pdf",
                 titel="Kundenwertanalyse Bestand",
                 untertitel="Segmentierung nach aggregierter Profitabilität",
                 quelle="Bestandsdaten Stichtag TT.MM.JJJJ")

Aufbau des Dokuments (Seitenzahl variiert je Datenstand):
    1  Fragestellung, Eckwerte und Befunde
    2  Kundenstruktur im Vergleich
    3  Wie Kunden negativ werden (Beitragsverwendung, Margenverteilung)
    4  Wo der Verlust beim einzelnen Kunden sitzt (Sanierungsbedarf)
    5  Portfoliostruktur: Bereiche (PKW/HUS) und Zweige
    6  Verlusttreiber und Konzentration
    7  Methodik, Ebenenaufbau und Sensitivität
"""

from __future__ import annotations

import io
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

# --------------------------------------------------------------------------------------
# Farbwelt (zentral anpassbar, z. B. auf CI-Farben)
# --------------------------------------------------------------------------------------
C_PRIMAER = colors.HexColor("#123A5E")   # dunkelblau, Überschriften/Deckblatt
C_AKZENT = colors.HexColor("#2E7DA8")    # mittelblau
C_POSITIV = colors.HexColor("#2E7D5B")   # gruen  -> wertvolle Kunden
C_NEGATIV = colors.HexColor("#B3402F")   # rot    -> nicht wertvolle Kunden
C_GRAU = colors.HexColor("#5A6570")
C_HELLGRAU = colors.HexColor("#EDF1F4")
C_LINIE = colors.HexColor("#C9D3DB")

MPL_POSITIV = "#2E7D5B"
MPL_NEGATIV = "#B3402F"
MPL_PRIMAER = "#123A5E"
MPL_AKZENT = "#2E7DA8"
MPL_GRAU = "#5A6570"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#C9D3DB",
    "axes.labelcolor": "#3A444E",
    "text.color": "#1F2933",
    "xtick.color": "#3A444E",
    "ytick.color": "#3A444E",
    "axes.grid": True,
    "grid.color": "#E4E9ED",
    "grid.linewidth": 0.7,
    "figure.dpi": 200,
})


# ======================================================================================
# Formatierung (deutsche Konvention)
# ======================================================================================
def _fmt(x: float, dez: int = 0, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n. v."
    s = f"{x:,.{dez}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return s + suffix


def _pct(x: float, dez: int = 1) -> str:
    return "n. v." if x is None or not np.isfinite(x) else _fmt(100 * x, dez, " %")


def _eur(x: float, dez: int = 0) -> str:
    return "n. v." if x is None or not np.isfinite(x) else _fmt(x, dez, " EUR")


def _eur_kurz(x: float) -> str:
    """Große Beträge vorstandstauglich verkürzen: 1,2 Mio. EUR."""
    if x is None or not np.isfinite(x):
        return "n. v."
    a = abs(x)
    if a >= 1e9:
        return _fmt(x / 1e9, 2, " Mrd. EUR")
    if a >= 1e6:
        return _fmt(x / 1e6, 2, " Mio. EUR")
    if a >= 1e4:
        return _fmt(x / 1e3, 1, " Tsd. EUR")
    return _fmt(x, 0, " EUR")


# ======================================================================================
# Styles
# ======================================================================================
def _styles() -> Dict[str, ParagraphStyle]:
    s = getSampleStyleSheet()
    return {
        "doc_titel": ParagraphStyle("doc_titel", parent=s["Title"], fontName="Helvetica-Bold",
                                    fontSize=20, leading=24, textColor=C_PRIMAER,
                                    alignment=TA_LEFT, spaceAfter=2),
        "doc_untertitel": ParagraphStyle("doc_untertitel", parent=s["Normal"], fontSize=11,
                                         leading=15, textColor=C_GRAU, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontName="Helvetica-Bold",
                             fontSize=16, leading=20, textColor=C_PRIMAER,
                             spaceBefore=0, spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontName="Helvetica-Bold",
                             fontSize=11.5, leading=15, textColor=C_PRIMAER,
                             spaceBefore=12, spaceAfter=4),
        "lead": ParagraphStyle("lead", parent=s["Normal"], fontSize=10, leading=14.5,
                               textColor=C_GRAU, spaceAfter=8),
        "text": ParagraphStyle("text", parent=s["Normal"], fontSize=9.5, leading=13.5,
                               textColor=colors.HexColor("#1F2933"), spaceAfter=4),
        "bullet": ParagraphStyle("bullet", parent=s["Normal"], fontSize=9.5, leading=13.5,
                                 leftIndent=12, bulletIndent=2, spaceAfter=5,
                                 textColor=colors.HexColor("#1F2933")),
        "klein": ParagraphStyle("klein", parent=s["Normal"], fontSize=8, leading=11,
                                textColor=C_GRAU),
        "kpi_wert": ParagraphStyle("kpi_wert", parent=s["Normal"], fontSize=15.5, leading=19,
                                   fontName="Helvetica-Bold", textColor=C_PRIMAER),
        "kpi_label": ParagraphStyle("kpi_label", parent=s["Normal"], fontSize=7.8, leading=10,
                                    textColor=C_GRAU),
        "eck_wert": ParagraphStyle("eck_wert", parent=s["Normal"], fontSize=12.5, leading=15,
                                   fontName="Helvetica-Bold",
                                   textColor=colors.HexColor("#1F2933")),
        "eck_label": ParagraphStyle("eck_label", parent=s["Normal"], fontSize=7.6, leading=10,
                                    textColor=C_GRAU),
        "th": ParagraphStyle("th", parent=s["Normal"], fontSize=8.5, leading=11,
                             fontName="Helvetica-Bold", textColor=colors.white),
        "td": ParagraphStyle("td", parent=s["Normal"], fontSize=8.5, leading=11),
    }


# ======================================================================================
# Bausteine
# ======================================================================================
def _bild(fig, breite: float) -> Image:
    """matplotlib-Figure -> reportlab-Image mit fester Zielbreite."""
    puffer = io.BytesIO()
    fig.savefig(puffer, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    puffer.seek(0)
    img = Image(puffer)
    faktor = breite / img.drawWidth
    img.drawWidth = breite
    img.drawHeight *= faktor
    return img


def _eckwerte(eintraege: Sequence[Tuple[str, str]], breite: float, st) -> Table:
    """Nüchterne Kennzahlenzeile: Liste aus (Wert, Label), ohne Dashboard-Optik."""
    zellen = [[Paragraph(w, st["eck_wert"]) for w, _ in eintraege],
              [Paragraph(l, st["eck_label"]) for _, l in eintraege]]
    n = len(eintraege)
    t = Table(zellen, colWidths=[breite / n] * n, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
        ("TOPPADDING", (0, 1), (-1, 1), 1),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("LEFTPADDING", (1, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, C_LINIE),
        ("LINEBELOW", (0, 1), (-1, 1), 0.6, C_LINIE),
    ]))
    return t


def _zwei_bilder(fig_links, fig_rechts, breite: float) -> Table:
    """Zwei Grafiken nebeneinander; rechts darf None sein."""
    links = _bild(fig_links, breite / 2 - 6)
    rechts = _bild(fig_rechts, breite / 2 - 6) if fig_rechts is not None else ""
    return Table([[links, rechts]], colWidths=[breite / 2, breite / 2], hAlign="LEFT",
                 style=[("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("VALIGN", (0, 0), (-1, -1), "TOP")])


def _kpi_kacheln(eintraege: Sequence[Tuple[str, str]], breite: float, st) -> Table:
    """Kachelreihe: Liste aus (Wert, Label)."""
    zellen = [[Paragraph(w, st["kpi_wert"]) for w, _ in eintraege],
              [Paragraph(l, st["kpi_label"]) for _, l in eintraege]]
    n = len(eintraege)
    t = Table(zellen, colWidths=[breite / n] * n, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_HELLGRAU),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
        ("TOPPADDING", (0, 1), (-1, 1), 1),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBEFORE", (1, 0), (-1, -1), 0.8, colors.white),
        ("LINEABOVE", (0, 0), (-1, 0), 2.2, C_AKZENT),
    ]))
    return t


def _tabelle(kopf: Sequence[str], zeilen: Sequence[Sequence[str]], breiten: Sequence[float],
             st, rechts_ab: int = 1, hervorheben: Optional[Sequence[int]] = None,
             text_spalten: Sequence[int] = ()) -> Table:
    th_r = ParagraphStyle("th_r", parent=st["th"], alignment=TA_RIGHT)
    daten = [[Paragraph(h, st["th"] if (i < rechts_ab or i in text_spalten) else th_r)
              for i, h in enumerate(kopf)]]
    daten += [[str(z) for z in zeile] for zeile in zeilen]
    t = Table(daten, colWidths=list(breiten), repeatRows=1, hAlign="LEFT")
    stil = [
        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMAER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1F2933")),
        ("ALIGN", (rechts_ab, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, C_LINIE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, C_PRIMAER),
    ]
    for i in range(1, len(daten)):
        if i % 2 == 0:
            stil.append(("BACKGROUND", (0, i), (-1, i), C_HELLGRAU))
    for i in (hervorheben or []):
        stil.append(("FONTNAME", (0, i), (-1, i), "Helvetica-Bold"))
    for c in text_spalten:
        stil.append(("ALIGN", (c, 1), (c, -1), "LEFT"))
    t.setStyle(TableStyle(stil))
    return t


def _hinweis(text: str, st) -> Table:
    t = Table([[Paragraph(text, st["klein"])]], colWidths=[16.4 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, C_AKZENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# ======================================================================================
# Grafiken
# ======================================================================================
def _chart_anteile(kz: pd.DataFrame) -> plt.Figure:
    """Gestapelte 100-%-Balken: Kunden, Verträge, Beitrag + Profitbeitrag in EUR."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 2.9),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    kat = ["Kunden", "Verträge", "Beitrag"]
    neg = [kz.loc["Kunden (Anteil)", "negativ"],
           kz.loc["Verträge (Anteil am Bestand)", "negativ"],
           kz.loc["Beitrag (Anteil am Gesamtbeitrag)", "negativ"]]
    pos = [1 - n for n in neg]
    y = np.arange(len(kat))
    ax1.barh(y, neg, color=MPL_NEGATIV, height=0.55, label="nicht wertvoll")
    ax1.barh(y, pos, left=neg, color=MPL_POSITIV, height=0.55, label="wertvoll")
    for i, n in enumerate(neg):
        ax1.text(n / 2, i, _pct(n, 0), ha="center", va="center", color="white",
                 fontsize=8.5, fontweight="bold")
        ax1.text(n + (1 - n) / 2, i, _pct(1 - n, 0), ha="center", va="center",
                 color="white", fontsize=8.5, fontweight="bold")
    ax1.set_yticks(y, kat)
    ax1.set_xlim(0, 1)
    ax1.invert_yaxis()
    ax1.set_xticks([])
    ax1.grid(False)
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.set_title("Verteilung des Bestands", loc="left", fontsize=9.5, fontweight="bold",
                  color=MPL_PRIMAER, pad=8)
    ax1.legend(frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.05), ncol=2)

    p_neg = kz.loc["Profit gesamt", "negativ"]
    p_pos = kz.loc["Profit gesamt", "positiv"]
    ax2.bar([0, 1], [p_neg, p_pos], color=[MPL_NEGATIV, MPL_POSITIV], width=0.5)
    ax2.axhline(0, color="#8C97A0", lw=0.9)
    for x, v in zip([0, 1], [p_neg, p_pos]):
        ax2.text(x, v, _eur_kurz(v), ha="center", fontsize=8.5, fontweight="bold",
                 va="bottom" if v >= 0 else "top", color=MPL_PRIMAER)
    ax2.set_xticks([0, 1], ["nicht wertvoll", "wertvoll"], fontsize=8.5)
    ax2.set_yticks([])
    ax2.grid(False)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.margins(y=0.28)
    ax2.set_title("Ergebnisbeitrag", loc="left", fontsize=9.5, fontweight="bold",
                  color=MPL_PRIMAER, pad=8)
    fig.tight_layout()
    return fig


def _chart_vertragsverteilung(vv: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    idx = [str(i) for i in vv.index]
    x = np.arange(len(idx))
    ax.bar(x - 0.2, vv.get("anteil_negativ", pd.Series(0, index=vv.index)), width=0.4,
           color=MPL_NEGATIV, label="nicht wertvoll")
    ax.bar(x + 0.2, vv.get("anteil_positiv", pd.Series(0, index=vv.index)), width=0.4,
           color=MPL_POSITIV, label="wertvoll")
    ax.set_xticks(x, idx, fontsize=8.5)
    ax.set_ylabel("Anteil Kunden")
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.set_xlabel("Verträge je Kunde", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_marge_streuung(kunden: pd.DataFrame, schwelle: float = 0.20,
                          max_punkte: int = 6000, breit: bool = False) -> plt.Figure:
    """Streudiagramm: Beitrag je Kunde gegen Marge - wo genau liegen die negativen Kunden?"""
    d = kunden[(kunden["beitrag_sum"] > 0) & kunden["marge"].notna()]
    if len(d) > max_punkte:
        d = d.sample(max_punkte, random_state=0)
    fig, ax = plt.subplots(figsize=(8.6, 2.9) if breit else (4.2, 2.7))
    for flag, farbe in (("positiv", MPL_POSITIV), ("negativ", MPL_NEGATIV)):
        t = d[d["wert_flag"] == flag]
        ax.scatter(t["beitrag_sum"], t["marge"].clip(-2, 1.5), s=5 if breit else 4,
                   alpha=0.35, color=farbe, linewidths=0,
                   label="wertvoll" if flag == "positiv" else "nicht wertvoll")
    ax.axhline(-schwelle, color=MPL_PRIMAER, ls="--", lw=1.1)
    ax.text(d["beitrag_sum"].min(), -schwelle - 0.04, f"Grenze {_pct(-schwelle, 0)}",
            fontsize=7.5, color=MPL_PRIMAER, va="top", ha="left")
    ax.axhline(0, color=MPL_GRAU, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Beitrag je Kunde (logarithmisch)", fontsize=8)
    ax.set_ylabel("Marge des Kunden", fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.legend(frameon=False, fontsize=8, loc="upper right", markerscale=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_margen_histogramm(kunden: pd.DataFrame, schwelle: float = 0.20,
                             breit: bool = False) -> plt.Figure:
    """Wie weit unter der Grenze liegen die negativen Kunden?"""
    fig, ax = plt.subplots(figsize=(8.6, 2.7) if breit else (4.2, 2.7))
    kanten = np.linspace(-1.5, 1.0, 76 if breit else 51)
    neg = kunden.loc[kunden["wert_flag"] == "negativ", "marge"].dropna().clip(-1.5, 1.0)
    pos = kunden.loc[kunden["wert_flag"] == "positiv", "marge"].dropna().clip(-1.5, 1.0)
    ax.hist([neg, pos], bins=kanten, stacked=True, color=[MPL_NEGATIV, MPL_POSITIV],
            label=["nicht wertvoll", "wertvoll"])
    ax.axvline(-schwelle, color=MPL_PRIMAER, ls="--", lw=1.2)
    ax.text(-schwelle - 0.02, ax.get_ylim()[1] * 0.95, f"Grenze {_pct(-schwelle, 0)}",
            fontsize=7.5, color=MPL_PRIMAER, ha="right", va="top")
    if breit and len(neg):
        median_neg = float(neg.median())
        ax.axvline(median_neg, color=MPL_NEGATIV, ls=":", lw=1.2)
        ax.text(median_neg - 0.02, ax.get_ylim()[1] * 0.72,
                f"Median nicht wertvoll {_pct(median_neg, 0)}", fontsize=7.5,
                color=MPL_NEGATIV, ha="right", va="top")
    ax.set_xlabel("Marge des Kunden", fontsize=8)
    ax.set_ylabel("Kunden", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_verlustvertraege(kunden: pd.DataFrame) -> plt.Figure:
    """Wie viele Verträge eines Kunden sind selbst defizitär?"""
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    kanten = np.linspace(0, 1, 11)
    neg = kunden.loc[kunden["wert_flag"] == "negativ", "anteil_verlustvertraege"].dropna()
    pos = kunden.loc[kunden["wert_flag"] == "positiv", "anteil_verlustvertraege"].dropna()
    ax.hist(neg, bins=kanten, weights=np.ones(len(neg)) / max(len(neg), 1),
            color=MPL_NEGATIV, alpha=0.85, label="nicht wertvoll")
    ax.hist(pos, bins=kanten, weights=np.ones(len(pos)) / max(len(pos), 1),
            color=MPL_POSITIV, alpha=0.55, label="wertvoll")
    ax.set_xlabel("Anteil defizitärer Verträge am Bestand des Kunden", fontsize=8)
    ax.set_ylabel("Anteil der Kunden", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_sanierung(sv: pd.DataFrame) -> plt.Figure:
    """Wie viele Verträge müssten je Kunde korrigiert werden?"""
    d = sv[sv["kunden"] > 0]
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["anteil_kunden"], color=MPL_NEGATIV, height=0.55)
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["anteil_kunden"] + 0.012, yi,
                f"{_pct(r['anteil_kunden'], 0)}  ({_pct(r['anteil_verlust'], 0)} d. Verlusts)",
                va="center", fontsize=7.5, color=MPL_PRIMAER)
    ax.set_yticks(y, [str(i) for i in d.index], fontsize=8.5)
    ax.set_xlim(0, min(1.35, d["anteil_kunden"].max() * 2.1))
    ax.set_xlabel("Anteil der nicht wertvollen Kunden", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_bereichskombination(bk: pd.DataFrame) -> plt.Figure:
    """Hält der Kunde nur PKW, nur HUS oder beides - und wie oft ist er dann negativ?"""
    d = bk.sort_values("kunden_gesamt", ascending=False)
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["negativ_quote"], color=MPL_NEGATIV, height=0.5)
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["negativ_quote"] + 0.008, yi,
                f"{_pct(r['negativ_quote'], 1)}  ({_fmt(r['kunden_gesamt'])} Kunden)",
                va="center", fontsize=7.5, color=MPL_PRIMAER)
    ax.set_yticks(y, [str(i) for i in d.index], fontsize=9)
    ax.set_xlim(0, min(1.0, d["negativ_quote"].max() * 2.0))
    ax.set_xlabel("Anteil nicht wertvoller Kunden in der Gruppe", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_vertragsanzahl_bereich(kunden: pd.DataFrame) -> plt.Figure:
    """Verträge je Kunde, getrennt nach PKW und HUS."""
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    gruppen = [("negativ", "nicht wertvoll", MPL_NEGATIV),
               ("positiv", "wertvoll", MPL_POSITIV)]
    x = np.arange(2)
    for i, (flag, name, farbe) in enumerate(gruppen):
        k = kunden[kunden["wert_flag"] == flag]
        werte = [k["n_vertraege_pkw"].mean(), k["n_vertraege_hus"].mean()]
        b = ax.bar(x + (i - 0.5) * 0.38, werte, width=0.36, color=farbe, label=name)
        ax.bar_label(b, fmt=lambda v: _fmt(v, 2), fontsize=7.5, padding=2,
                     color=MPL_PRIMAER)
    ax.set_xticks(x, ["PKW-Verträge", "HUS-Verträge"], fontsize=9)
    ax.set_ylabel("Verträge je Kunde (Mittel)", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.margins(y=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_bereichsmix(bm: pd.DataFrame) -> plt.Figure:
    """Ebene 1: PKW vs. HUS im Bestand des durchschnittlichen Kunden."""
    d = bm.sort_index()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["anteil_vertraege_negativ"], height=0.4, color=MPL_NEGATIV,
            label="nicht wertvoll")
    ax.barh(y + 0.2, d["anteil_vertraege_positiv"], height=0.4, color=MPL_POSITIV,
            label="wertvoll")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["anteil_vertraege_negativ"] + 0.01, yi - 0.2,
                _pct(r["anteil_vertraege_negativ"], 1), va="center", fontsize=7.5,
                color=MPL_PRIMAER)
        ax.text(r["anteil_vertraege_positiv"] + 0.01, yi + 0.2,
                _pct(r["anteil_vertraege_positiv"], 1), va="center", fontsize=7.5,
                color=MPL_PRIMAER)
    ax.set_yticks(y, [str(i) for i in d.index], fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.set_xlabel("mittlerer Anteil am Vertragsbestand des Kunden", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_produktmix(pm: pd.DataFrame, top: int = 8) -> plt.Figure:
    """Obere Ebene: mittlere Produktzusammensetzung des Kundenbestands je Segment."""
    d = pm.copy()
    d["basis"] = d["anteil_vertraege_negativ"] + d["anteil_vertraege_positiv"]
    d = d.sort_values("basis", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["anteil_vertraege_negativ"], height=0.4, color=MPL_NEGATIV,
            label="nicht wertvoll")
    ax.barh(y + 0.2, d["anteil_vertraege_positiv"], height=0.4, color=MPL_POSITIV,
            label="wertvoll")
    ax.set_yticks(y, [str(i)[:22] for i in d.index], fontsize=8.5)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.set_xlabel("mittlerer Anteil am Vertragsbestand des Kunden", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_konzentration(kunden: pd.DataFrame) -> plt.Figure:
    """Lorenzkurve: Wie stark konzentriert sich der Verlust auf wenige Kunden?"""
    neg = kunden[kunden["wert_flag"] == "negativ"].sort_values("profit_sum")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    if len(neg):
        anteil_k = np.arange(1, len(neg) + 1) / len(neg)
        kum = neg["profit_sum"].cumsum() / neg["profit_sum"].sum()
        ax.plot(anteil_k, kum, color=MPL_NEGATIV, lw=2)
        ax.fill_between(anteil_k, 0, kum, color=MPL_NEGATIV, alpha=0.12)
        ax.plot([0, 1], [0, 1], color=MPL_GRAU, lw=1, ls="--")
        i10 = max(0, int(0.1 * len(neg)) - 1)
        ax.annotate(f"Top 10 % der Kunden = {_pct(kum.iloc[i10], 0)} des Verlusts",
                    xy=(0.1, kum.iloc[i10]), xytext=(0.30, max(0.08, kum.iloc[i10] - 0.30)),
                    fontsize=7.5, color=MPL_PRIMAER,
                    arrowprops=dict(arrowstyle="->", color=MPL_GRAU, lw=0.9))
    ax.set_xlabel("kumulierter Anteil der Kunden (nach Verlusthöhe)", fontsize=8)
    ax.set_ylabel("kumulierter Verlust", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_verlusttreiber(vt: pd.DataFrame, top: int = 8) -> plt.Figure:
    d = vt.head(top).iloc[::-1]
    skala, einheit = (1e6, "Mio. EUR") if d["profit"].abs().max() >= 1e6 else (1e3, "Tsd. EUR")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    y = np.arange(len(d))
    ax.barh(y, d["profit"] / skala, color=MPL_NEGATIV, height=0.6)
    ax.set_yticks(y, [str(i)[:22] for i in d.index], fontsize=8.5)
    ax.set_xlabel(f"Ergebnisbeitrag im Segment „nicht wertvoll“ in {einheit}", fontsize=8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.xaxis.set_major_formatter(lambda v, p: _fmt(v, 1))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_sensitivitaet(sens: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.plot(sens.index, sens["anteil_kunden_negativ"], marker="o", color=MPL_NEGATIV,
            lw=1.8, label="Anteil Kunden")
    ax.plot(sens.index, sens["anteil_beitrag_negativ"], marker="s", color=MPL_AKZENT,
            lw=1.8, label="Anteil Beitrag")
    ax.axvline(0.20, color=MPL_GRAU, ls="--", lw=1)
    ax.text(0.205, ax.get_ylim()[1] * 0.95, "gewählte Grenze", fontsize=7.5,
            color=MPL_GRAU, va="top")
    ax.set_xlabel("Schwellenwert (negative Marge)", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.yaxis.set_major_formatter(lambda v, p: _pct(v, 0))
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ======================================================================================
# Kernaussagen (automatisch aus den Zahlen abgeleitet)
# ======================================================================================
def _aufwandsquote(marge: float) -> float:
    """
    Aufwand in Prozent des Beitrags, abgeleitet aus Beitrag und Ergebnis: 100 % - Marge.

    Bewusst NICHT aus ve_expected_claim_amount + ve_total_cost gerechnet - diese Felder
    sind nicht durchgängig befüllt, sodass Aufwand und Marge sich sonst nicht zu 100 %
    ergänzen würden.
    """
    return np.nan if marge is None or not np.isfinite(marge) else 1.0 - marge


def _kernaussagen(erg: Dict[str, pd.DataFrame]) -> List[str]:
    """Befunde in Prioritätsreihenfolge; Formulierung richtet sich nach den Zahlen."""
    kz, kunden = erg["kennzahlen"], erg["kunden"]
    a: List[str] = []

    # 1) Größe und Ergebniswirkung des Segments
    anteil_k = kz.loc["Kunden (Anteil)", "negativ"]
    anteil_b = kz.loc["Beitrag (Anteil am Gesamtbeitrag)", "negativ"]
    verlust = kz.loc["Profit gesamt", "negativ"]
    gewinn = kz.loc["Profit gesamt", "positiv"]
    quote = abs(verlust) / gewinn if gewinn else np.nan
    a.append(f"<b>{_pct(anteil_k, 1)} der Kunden binden {_pct(anteil_b, 1)} des Beitrags "
             f"und kosten {_eur_kurz(abs(verlust))}</b> - das entspricht {_pct(quote, 0)} "
             f"des Ergebnisses aus dem wertvollen Segment ({_eur_kurz(gewinn)}).")

    # 2) Preis oder Aufwand?
    b_neg = kz.loc["Beitrag je Vertrag (gepoolt)", "negativ"]
    b_pos = kz.loc["Beitrag je Vertrag (gepoolt)", "positiv"]
    diff = b_neg / b_pos - 1 if b_pos else np.nan
    cr_neg = _aufwandsquote(kz.loc["Marge gepoolt (Profit/Beitrag)", "negativ"])
    cr_pos = _aufwandsquote(kz.loc["Marge gepoolt (Profit/Beitrag)", "positiv"])
    if abs(diff) < 0.05:
        preis = (f"Der Beitrag je Vertrag ist in beiden Segmenten praktisch gleich "
                 f"({_eur(b_neg)} zu {_eur(b_pos)})")
    else:
        preis = (f"Der Beitrag je Vertrag liegt bei nicht wertvollen Kunden mit "
                 f"{_eur(b_neg)} sogar {_pct(abs(diff), 1)} "
                 f"{'über' if diff > 0 else 'unter'} dem wertvollen Segment "
                 f"({_eur(b_pos)})")
    a.append(f"{preis}. Der Aufwand dagegen liegt bei <b>{_pct(cr_neg, 1)} des Beitrags "
             f"gegenüber {_pct(cr_pos, 1)}</b>.")

    # 3) Verbund und Bereichsgewichtung
    bk = erg.get("bereichs_kombination")
    if bk is not None and len(bk) > 1 and "negativ_quote" in bk.columns:
        d = bk.sort_values("negativ_quote", ascending=False)
        if d["negativ_quote"].max() - d["negativ_quote"].min() > 0.02:
            teile = [f"<b>{i}</b> {_pct(r['negativ_quote'], 1)}" for i, r in d.iterrows()]
            a.append("Negativ-Quote je Verbundtyp: " + ", ".join(teile) + ".")

    bm = erg.get("bereich_mix")
    if bm is not None and len(bm) > 1 and "PKW" in bm.index:
        r = bm.loc["PKW"]
        a.append(f"<b>PKW</b> macht bei nicht wertvollen Kunden "
                 f"{_pct(r['anteil_vertraege_negativ'], 1)} des Vertragsbestands aus, bei "
                 f"wertvollen {_pct(r['anteil_vertraege_positiv'], 1)}; PKW-Marge "
                 f"{_pct(r['marge_negativ'], 1)} zu {_pct(r['marge_positiv'], 1)}.")

    # 4) Größter Verlusttreiber
    vt = erg.get("verlusttreiber_zweig")
    if vt is not None and len(vt):
        r = vt.iloc[0]
        a.append(f"Größter Verlusttreiber ist der Zweig <b>{vt.index[0]}</b>: "
                 f"{_eur_kurz(r['profit'])} bei einer Marge von {_pct(r['marge'], 1)}, "
                 f"{_pct(r['anteil_am_gesamtverlust'], 0)} des Segmentverlusts.")

    # 5) Wie tief sitzt der Verlust beim einzelnen Kunden?
    vv = kz.loc["Anteil Verlustverträge je Kunde (Mittel)", "negativ"]
    sv = erg.get("sanierung_verteilung")
    zusatz = ""
    if sv is not None and len(sv) and sv["kunden"].sum() > 0:
        teile = []
        if "1 von mehreren Verträgen" in sv.index and sv.loc["1 von mehreren Verträgen", "kunden"]:
            teile.append(f"bei <b>{_pct(sv.loc['1 von mehreren Verträgen', 'anteil_kunden'], 0)}"
                         f"</b> genügt die Korrektur des schlechtesten Vertrags")
        if "Einvertragskunde" in sv.index and sv.loc["Einvertragskunde", "kunden"]:
            teile.append(f"{_pct(sv.loc['Einvertragskunde', 'anteil_kunden'], 0)} halten nur "
                         f"einen Vertrag")
        if teile:
            zusatz = " Ansatzpunkte: " + ", ".join(teile) + "."
    a.append(f"<b>{_pct(vv, 0)} der Verträge</b> eines nicht wertvollen Kunden sind einzeln "
             f"defizitär, im wertvollen Segment "
             f"{_pct(kz.loc['Anteil Verlustverträge je Kunde (Mittel)', 'positiv'], 0)}."
             f"{zusatz}")

    # 6) Konzentration
    konz = erg.get("konzentration")
    if konz is not None and 0.10 in konz.index:
        z10 = konz.loc[0.10]
        anteil10 = z10["anteil_am_negativen_profit"]
        deutung = ("Maßnahmen lassen sich eng auf diese Kunden fokussieren."
                   if anteil10 >= 0.40 else
                   "Der Verlust streut damit breit - Einzelfallsteuerung greift zu kurz.")
        a.append(f"Die {_fmt(z10['kunden'])} verlustreichsten Kunden (10 % des Segments) "
                 f"verantworten <b>{_pct(anteil10, 0)} des negativen Ergebnisses</b>. "
                 f"{deutung}")

    return a[:6]


# ======================================================================================
# Seitenrahmen
# ======================================================================================
class _Doc(BaseDocTemplate):
    def __init__(self, pfad: str, titel: str, fusszeile: str, **kw):
        super().__init__(pfad, pagesize=A4, title=titel, author="Data Science",
                         leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                         topMargin=2.0 * cm, bottomMargin=1.8 * cm, **kw)
        self.titel_kurz = titel
        self.fusszeile = fusszeile
        rahmen = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                       id="normal", leftPadding=0, rightPadding=0,
                       topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="Inhalt", frames=[rahmen], onPage=self._kopf_fuss),
        ])

    def _kopf_fuss(self, canv, doc):
        canv.saveState()
        canv.setStrokeColor(C_LINIE)
        canv.setLineWidth(0.6)
        canv.line(self.leftMargin, A4[1] - 1.35 * cm,
                  A4[0] - self.rightMargin, A4[1] - 1.35 * cm)
        canv.setFont("Helvetica", 7.5)
        canv.setFillColor(C_GRAU)
        canv.drawString(self.leftMargin, A4[1] - 1.15 * cm, self.titel_kurz)
        canv.drawRightString(A4[0] - self.rightMargin, A4[1] - 1.15 * cm, self.fusszeile)
        canv.line(self.leftMargin, 1.35 * cm, A4[0] - self.rightMargin, 1.35 * cm)
        canv.drawString(self.leftMargin, 1.0 * cm, "Vertraulich - nur zur internen Verwendung")
        canv.drawRightString(A4[0] - self.rightMargin, 1.0 * cm, f"Seite {doc.page}")
        canv.restoreState()


# ======================================================================================
# Hauptfunktion
# ======================================================================================
def erstelle_pdf(
    erg: Dict[str, pd.DataFrame],
    pfad: str = "Kundenwertanalyse.pdf",
    titel: str = "Kundenwertanalyse",
    untertitel: str = "Segmentierung des Bestands nach aggregierter Profitabilität",
    quelle: str = "",
    verfasser: str = "",
    stand: Optional[str] = None,
    schwelle: float = 0.20,
) -> str:
    """Erzeugt die PDF-Vorlage und gibt den Dateipfad zurück."""
    st = _styles()
    stand = stand or date.today().strftime("%d.%m.%Y")
    kz, kunden = erg["kennzahlen"], erg["kunden"]
    regel = str(kunden["regel"].iloc[0])
    breite = A4[0] - 4.4 * cm

    doc = _Doc(pfad, titel, stand)
    s: List = []

    # ------------------------------------------------- Titelkopf auf Seite 1
    s.append(Paragraph(titel, st["doc_titel"]))
    s.append(Paragraph(untertitel, st["doc_untertitel"]))
    meta = [f"Stand: {stand}",
            f"Datenbasis: {_fmt(len(kunden))} Kunden, "
            f"{_fmt(kunden['n_vertraege'].sum())} Verträge"]
    if quelle:
        meta.append(quelle)
    if verfasser:
        meta.append(verfasser)
    zeile = Table([[Paragraph("&nbsp;&nbsp;|&nbsp;&nbsp;".join(meta), st["klein"])]],
                  colWidths=[breite], hAlign="LEFT")
    zeile.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, C_LINIE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, C_LINIE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    s.append(Spacer(1, 6))
    s.append(zeile)
    s.append(Spacer(1, 16))

    # ------------------------------------------- Seite 1: Fragestellung und Befunde
    s.append(Paragraph("Fragestellung", st["h1"]))
    s.append(Paragraph(
        "Worin unterscheiden sich Kunden mit deutlich negativem Ergebnisbeitrag von den "
        "übrigen? Beitrag und Ergebnis werden je Kunde über alle Verträge aggregiert. "
        "<b>Nicht wertvoll</b> heißt: aggregiertes Ergebnis negativ und betragsmäßig "
        "mindestens 20 Prozent des Bestandsjahresnettobeitrags.",
        st["lead"]))

    eckwerte = [
        (_fmt(kz.loc["Kunden (Anzahl)", "negativ"]), "nicht wertvolle Kunden"),
        (_pct(kz.loc["Kunden (Anteil)", "negativ"], 1), "Anteil am Kundenbestand"),
        (_pct(kz.loc["Beitrag (Anteil am Gesamtbeitrag)", "negativ"], 1), "Anteil am Beitrag"),
        (_eur_kurz(kz.loc["Profit gesamt", "negativ"]).replace(" EUR", ""),
         "Ergebnisbeitrag (EUR)"),
        (_pct(kz.loc["Marge gepoolt (Profit/Beitrag)", "negativ"], 1), "Marge"),
    ]
    s.append(_eckwerte(eckwerte, breite, st))
    s.append(Spacer(1, 12))
    s.append(_bild(_chart_anteile(kz), breite))
    s.append(Spacer(1, 10))
    s.append(Paragraph("Befunde", st["h2"]))
    for a in _kernaussagen(erg):
        s.append(Paragraph(a, st["bullet"], bulletText="\u25aa"))
    s.append(PageBreak())

    # ------------------------------------------------ Seite 2: Kundenstruktur
    s.append(Paragraph("Kundenstruktur im Vergleich", st["h1"]))
    s.append(Paragraph("Verträge, Beitragsvolumen und Ergebnis beider Segmente im "
                       "direkten Vergleich.", st["lead"]))

    auswahl = [
        ("Kunden (Anzahl)", "Kunden", _fmt),
        ("Verträge (Anzahl gesamt)", "Verträge (ve_id)", _fmt),
        ("Positionen (Zeilen) gesamt", "Positionen (Zeilen)", _fmt),
        ("Positionen je Vertrag (Mittel)", "Positionen je Vertrag", lambda x: _fmt(x, 2)),
        ("Verträge je Kunde (Mittel)", "Verträge je Kunde (Mittel)", lambda x: _fmt(x, 2)),
        ("Verträge je Kunde (Median)", "Verträge je Kunde (Median)", lambda x: _fmt(x, 1)),
        ("Anteil Kunden mit nur 1 Vertrag", "Anteil Einvertragskunden", _pct),
        ("PKW-Verträge je Kunde (Mittel)", "davon PKW-Verträge (Mittel)", lambda x: _fmt(x, 2)),
        ("HUS-Verträge je Kunde (Mittel)", "davon HUS-Verträge (Mittel)", lambda x: _fmt(x, 2)),
        ("Anteil Kunden mit PKW-Vertrag", "Anteil Kunden mit PKW", _pct),
        ("Anteil Kunden mit PKW und HUS", "Anteil Kunden mit PKW und HUS", _pct),
        ("Beitrag gesamt", "Beitrag gesamt", _eur_kurz),
        ("Beitrag je Kunde (Mittel)", "Beitrag je Kunde (Mittel)", _eur),
        ("Beitrag je Kunde (Median)", "Beitrag je Kunde (Median)", _eur),
        ("Beitrag je Vertrag (gepoolt)", "Beitrag je Vertrag", _eur),
        ("Profit gesamt", "Ergebnis gesamt", _eur_kurz),
        ("Profit je Kunde (Mittel)", "Ergebnis je Kunde (Mittel)", _eur),
        ("Marge gepoolt (Profit/Beitrag)", "Marge (gepoolt)", _pct),
        ("Anteil Verlustverträge je Kunde (Mittel)", "Anteil defizitärer Verträge", _pct),
    ]
    zeilen = [[label, f(kz.loc[k, "negativ"]), f(kz.loc[k, "positiv"])]
              for k, label, f in auswahl if k in kz.index]
    # Aufwand aus Beitrag und Ergebnis abgeleitet, damit er exakt zur Marge passt
    marge_zeile = "Marge gepoolt (Profit/Beitrag)"
    if marge_zeile in kz.index:
        pos = next(i for i, z in enumerate(zeilen) if z[0] == "Marge (gepoolt)")
        zeilen.insert(pos, ["Aufwand in % des Beitrags",
                            _pct(_aufwandsquote(kz.loc[marge_zeile, "negativ"]), 1),
                            _pct(_aufwandsquote(kz.loc[marge_zeile, "positiv"]), 1)])
    s.append(_tabelle(["Kennzahl", "nicht wertvoll", "wertvoll"], zeilen,
                      [breite * 0.5, breite * 0.25, breite * 0.25], st))
    s.append(Spacer(1, 14))

    s.append(_zwei_bilder(
        _chart_vertragsverteilung(erg["verteilung_vertragsanzahl"]),
        _chart_vertragsanzahl_bereich(kunden)
        if "n_vertraege_pkw" in kunden.columns else _chart_bereichsmix(erg["bereich_mix"]),
        breite))
    s.append(Paragraph("Links: Kunden nach Anzahl ihrer Verträge (ve_id). "
                       "Rechts: Verträge je Kunde, getrennt nach PKW und HUS.",
                       st["klein"]))
    s.append(PageBreak())

    # ---------------------------------- Seite 3: Wie Kunden negativ werden
    bv = erg.get("beitragsverwendung")
    s.append(Paragraph("Verteilung der Kundenmargen", st["h1"]))
    s.append(Paragraph(
        "Wie weit liegen die Kunden auseinander, und wie weit reicht das negative Segment "
        "unter die Grenze? Darunter: dieselbe Marge gegen das Beitragsvolumen je Kunde.",
        st["lead"]))

    s.append(_bild(_chart_margen_histogramm(kunden, schwelle, breit=True), breite))
    s.append(Paragraph("Alles links der gestrichelten Grenze zählt als nicht wertvoll. "
                       "Der Abstand zur Grenze zeigt, wie knapp oder wie deutlich.",
                       st["klein"]))
    s.append(Spacer(1, 14))

    s.append(_bild(_chart_marge_streuung(kunden, schwelle, breit=True), breite))
    s.append(Paragraph("Jeder Punkt ein Kunde. Verteilen sich die roten Punkte über die "
                       "gesamte Beitragsachse, sind auch beitragsstarke Kunden betroffen.",
                       st["klein"]))
    s.append(Spacer(1, 14))

    if bv is not None and len(bv):
        margen = {f: (bv.loc[f, "profit"] / bv.loc[f, "beitrag"]
                      if f in bv.index and bv.loc[f, "beitrag"] else np.nan)
                  for f in ("negativ", "positiv")}
        zeilen = [["Aufwand in Prozent des Beitrags",
                   _pct(_aufwandsquote(margen["negativ"]), 1),
                   _pct(_aufwandsquote(margen["positiv"]), 1)],
                  ["Ergebnisquote (Marge)",
                   _pct(margen["negativ"], 1), _pct(margen["positiv"], 1)]]
        zeilen.append(["Beitragsvolumen des Segments",
                       _eur_kurz(kz.loc["Beitrag gesamt", "negativ"]),
                       _eur_kurz(kz.loc["Beitrag gesamt", "positiv"])])
        zeilen.append(["Ergebnis des Segments",
                       _eur_kurz(kz.loc["Profit gesamt", "negativ"]),
                       _eur_kurz(kz.loc["Profit gesamt", "positiv"])])
        s.append(_tabelle(["Beitrag und Aufwand", "nicht wertvoll", "wertvoll"], zeilen,
                          [breite * 0.5, breite * 0.25, breite * 0.25], st))
        s.append(Spacer(1, 6))
        s.append(Paragraph(
            "Aufwand = 100 % minus Marge, abgeleitet aus Beitrag und Ergebnis. Damit "
            "unabhängig davon, wie vollständig die Schaden- und Kostenfelder befüllt sind.",
            st["klein"]))
    s.append(PageBreak())

    # ------------------------- Seite 4: Wo der Verlust beim einzelnen Kunden sitzt
    s.append(Paragraph("Wo der Verlust beim einzelnen Kunden sitzt", st["h1"]))
    s.append(Paragraph(
        "Hängt der Verlust an einzelnen Verträgen oder am gesamten Bestand des Kunden? "
        "Je nicht wertvollem Kunden werden die Verträge nach Ergebnis sortiert und "
        "geprüft, wie viele der schlechtesten korrigiert werden müssten, damit er die "
        "Negativ-Grenze verlässt.", st["lead"]))

    s.append(_zwei_bilder(
        _chart_verlustvertraege(kunden),
        _chart_sanierung(erg["sanierung_verteilung"])
        if len(erg.get("sanierung_verteilung", [])) else None, breite))
    s.append(Paragraph("Links: Anteil der Verträge eines Kunden, die für sich genommen "
                       "defizitär sind. Rechts: notwendige Zahl an Korrekturen je Kunde.",
                       st["klein"]))
    s.append(Spacer(1, 14))

    sv = erg.get("sanierung_verteilung")
    if sv is not None and len(sv):
        zeilen = [[str(i), _fmt(r["kunden"]), _pct(r["anteil_kunden"], 1),
                   _eur_kurz(r["verlust"]), _pct(r["anteil_verlust"], 1),
                   _eur_kurz(r["beitrag"])]
                  for i, r in sv[sv["kunden"] > 0].iterrows()]
        s.append(Paragraph("Sanierungsbedarf je nicht wertvollem Kunden", st["h2"]))
        s.append(_tabelle(["notwendige Korrektur", "Kunden", "Anteil<br/>Kunden", "Ergebnis",
                           "Anteil am<br/>Verlust", "Beitrags-<br/>volumen"], zeilen,
                          [breite * 0.24, breite * 0.10, breite * 0.13, breite * 0.17,
                           breite * 0.16, breite * 0.20], st))
        s.append(Spacer(1, 8))
        s.append(_hinweis(
            "<b>Einvertragskunde:</b> nur ein Vertrag, eine Teilsanierung ist nicht "
            "möglich - die Entscheidung betrifft die gesamte Kundenbeziehung. "
            "<b>Kein Teilbestand reicht:</b> auch der verbleibende Bestand läge noch "
            "unter der Negativ-Grenze.", st))
    s.append(PageBreak())

    # ------------------------------------ Seite 5: Portfolio - Bereiche und Zweige
    bm, zjb = erg["bereich_mix"], erg.get("zweig_je_bereich")
    s.append(Paragraph("Portfoliostruktur: Bereiche und Zweige", st["h1"]))
    s.append(Paragraph(
        "Ebene 1 ist PKW gegenüber HUS. Ebene 2 teilt PKW nach Sparte und HUS nach "
        "Produkt auf - beides liegt damit auf derselben Ebene und ist vergleichbar.",
        st["lead"]))

    zeilen = [[str(i), _pct(r["anteil_vertraege_negativ"], 1),
               _pct(r["anteil_vertraege_positiv"], 1), _pct(r["delta_anteil_vertraege"], 1),
               _pct(r["penetration_negativ"], 0), _pct(r["penetration_positiv"], 0),
               _eur(r["beitrag_je_vertrag_negativ"]),
               _pct(r["marge_negativ"], 1), _pct(r["marge_positiv"], 1)]
              for i, r in bm.iterrows()]
    s.append(Paragraph("Ebene 1: Bereiche", st["h2"]))
    s.append(_tabelle(["Bereich", "Anteil<br/>n. wertv.", "Anteil<br/>wertvoll", "Delta",
                       "Durchdr.<br/>n. wertv.", "Durchdr.<br/>wertvoll",
                       "Beitrag je<br/>Vertrag", "Marge<br/>n. wertv.", "Marge<br/>wertvoll"],
                      zeilen,
                      [breite * 0.17, breite * 0.095, breite * 0.095, breite * 0.08,
                       breite * 0.12, breite * 0.12, breite * 0.11, breite * 0.105,
                       breite * 0.105], st))

    if zjb is not None and len(zjb):
        s.append(Paragraph("Ebene 2: Zweige innerhalb des Bereichs", st["h2"]))
        s.append(Paragraph("Anteile bezogen auf die Verträge des jeweiligen Bereichs; "
                           "sie addieren sich je Bereich und Segment auf 100 %.",
                           st["text"]))
        zeilen, fett = [], []
        for bereich, block in zjb.groupby(level=0, observed=True, sort=False):
            zeilen.append([str(bereich)[:26], "", "", "", "", "", ""])
            fett.append(len(zeilen))
            for (_, zweig), r in block.iterrows():
                zeilen.append([
                    f"     {str(zweig)[:24]}",
                    _pct(r["anteil_im_bereich_negativ"], 1),
                    _pct(r["anteil_im_bereich_positiv"], 1),
                    _pct(r["delta_anteil_im_bereich"], 1),
                    _eur(r["beitrag_je_vertrag_negativ"]),
                    _pct(r["marge_negativ"], 1),
                    _pct(r["marge_positiv"], 1)])
        s.append(_tabelle(["Bereich / Zweig", "Anteil im Bereich<br/>n. wertv.",
                           "Anteil im Bereich<br/>wertvoll", "Delta",
                           "Beitrag je Vertrag<br/>n. wertv.",
                           "Marge<br/>n. wertv.", "Marge<br/>wertvoll"], zeilen,
                          [breite * 0.24, breite * 0.145, breite * 0.145, breite * 0.085,
                           breite * 0.145, breite * 0.12, breite * 0.12], st,
                          hervorheben=fett))
    s.append(PageBreak())

    # ------------------- Seite: Bereichskombination und Deckungsprofile
    bk = erg.get("bereichs_kombination")
    vp = erg.get("vertragsprofile_pkw")
    if bk is not None and len(bk):
        s.append(Paragraph("Verbund und Deckungsumfang", st["h1"]))
        s.append(Paragraph(
            "Hält ein Kunde nur PKW, nur HUS oder beides - und wie oft ist er dann nicht "
            "wertvoll? Die Negativ-Quote misst den Anteil nicht wertvoller Kunden "
            "innerhalb jeder Gruppe.", st["lead"]))

        s.append(_zwei_bilder(_chart_bereichskombination(bk),
                              _chart_bereichsmix(erg["bereich_mix"]), breite))
        s.append(Paragraph("Links: Negativ-Quote je Kundengruppe. Rechts: Anteil von PKW "
                           "und HUS am Vertragsbestand des Kunden.", st["klein"]))
        s.append(Spacer(1, 12))

        zeilen = [[str(i), _fmt(r["kunden_gesamt"]), _pct(r["negativ_quote"], 1),
                   _fmt(r["vertraege_je_kunde_negativ"], 2),
                   _fmt(r["vertraege_je_kunde_positiv"], 2),
                   _eur(r["beitrag_je_kunde_negativ"]), _eur(r["beitrag_je_kunde_positiv"]),
                   _pct(r["marge_negativ"], 1), _pct(r["marge_positiv"], 1)]
                  for i, r in bk.iterrows()]
        s.append(_tabelle(["Kundengruppe", "Kunden", "Negativ-<br/>Quote",
                           "Verträge je Kunde<br/>n. wertv.",
                           "Verträge je Kunde<br/>wertvoll",
                           "Beitrag je Kunde<br/>n. wertv.",
                           "Beitrag je Kunde<br/>wertvoll",
                           "Marge<br/>n. wertv.", "Marge<br/>wertvoll"], zeilen,
                          [breite * 0.155, breite * 0.085, breite * 0.09, breite * 0.12,
                           breite * 0.12, breite * 0.12, breite * 0.12, breite * 0.095,
                           breite * 0.095], st))
        s.append(Spacer(1, 14))

    if vp is not None and len(vp):
        s.append(Paragraph("Deckungsprofile innerhalb der PKW-Verträge", st["h2"]))
        s.append(Paragraph(
            "Ein PKW-Vertrag (eine ve_id) kann mehrere Deckungen enthalten. Anteile "
            "bezogen auf alle PKW-Verträge des jeweiligen Segments.", st["text"]))
        zeilen = [[str(i)[:34], _fmt(r["vertraege_negativ"] + r["vertraege_positiv"]),
                   _pct(r["anteil_vertraege_negativ"], 1),
                   _pct(r["anteil_vertraege_positiv"], 1),
                   _pct(r["delta_anteil_vertraege"], 1),
                   _eur(r["beitrag_je_vertrag_negativ"]),
                   _pct(r["marge_negativ"], 1), _pct(r["marge_positiv"], 1)]
                  for i, r in vp.iterrows()]
        s.append(_tabelle(["Deckungsprofil", "Verträge", "Anteil<br/>n. wertv.",
                           "Anteil<br/>wertvoll", "Delta", "Beitrag je<br/>Vertrag",
                           "Marge<br/>n. wertv.", "Marge<br/>wertvoll"], zeilen,
                          [breite * 0.25, breite * 0.10, breite * 0.11, breite * 0.11,
                           breite * 0.08, breite * 0.12, breite * 0.115, breite * 0.115], st))
        s.append(PageBreak())
    s.append(Paragraph("Verlusttreiber und Konzentration", st["h1"]))
    s.append(Paragraph(
        "In welchen Zweigen entsteht der Verlust, und wie breit streut er über die "
        "Kunden? Davon hängt ab, ob eine Maßnahme breit ansetzen muss oder auf wenige "
        "Kunden fokussiert werden kann.", st["lead"]))

    s.append(_zwei_bilder(_chart_verlusttreiber(erg["verlusttreiber_zweig"]),
                          _chart_konzentration(kunden), breite))
    s.append(Paragraph("Links: Zweige mit dem größten Verlustbeitrag. Rechts: "
                       "Konzentration des Verlusts über die Kunden (Lorenzkurve).",
                       st["klein"]))
    s.append(Spacer(1, 14))

    konz = erg.get("konzentration")
    if konz is not None and len(konz):
        zeilen = [[f"Top {_pct(i, 0)} der Kunden", _fmt(r["kunden"]),
                   _eur_kurz(r["profit_summe"]), _pct(r["anteil_am_negativen_profit"], 1),
                   _eur_kurz(r["beitrag_summe"])]
                  for i, r in konz.iterrows()]
        s.append(Paragraph("Verlustkonzentration im Segment „nicht wertvoll“", st["h2"]))
        s.append(_tabelle(["Kundengruppe", "Kunden", "Ergebnis", "Anteil am Verlust",
                           "Beitragsvolumen"], zeilen,
                          [breite * 0.26, breite * 0.14, breite * 0.2, breite * 0.2,
                           breite * 0.2], st))

    vts = erg.get("verlusttreiber_bereich_zweig")
    if vts is not None and len(vts):
        s.append(Spacer(1, 14))
        s.append(Paragraph("Feingliederung: Bereich und Zweig", st["h2"]))
        zeilen = [[f"{str(i[0])[:16]} / {str(i[1])[:16]}", _fmt(r["vertraege"]),
                   _eur_kurz(r["beitrag"]), _eur_kurz(r["profit"]), _pct(r["marge"], 1),
                   _pct(r["anteil_am_gesamtverlust"], 1)]
                  for i, r in vts.head(10).iterrows()]
        s.append(_tabelle(["Bereich / Zweig", "Verträge", "Beitrag", "Ergebnis", "Marge",
                           "Anteil am Verlust"], zeilen,
                          [breite * 0.28, breite * 0.12, breite * 0.15, breite * 0.15,
                           breite * 0.13, breite * 0.17], st))
    s.append(PageBreak())

    # ---------------------------------------------------- Seite 7: Methodik
    s.append(Paragraph("Methodik und Robustheit", st["h1"]))
    s.append(Paragraph("Definition der Segmentierung", st["h2"]))
    s.append(_hinweis(
        f"Beitrag und Ergebnis werden je vn_partner_id über alle Verträge summiert. "
        f"<b>Nicht wertvoll</b>, wenn das aggregierte Ergebnis negativ ist <b>und</b> "
        f"betragsmäßig mindestens 20 Prozent des aggregierten Bestandsjahresnetto"
        f"beitrags erreicht - formal: {regel}. Alle übrigen Kunden gelten als wertvoll.",
        st))
    s.append(Spacer(1, 10))

    s.append(Paragraph("Zählweise von Verträgen", st["h2"]))
    s.append(_hinweis(
        f"Eine ve_id kann mehrere Zeilen umfassen, wenn ein Vertrag mehrere Deckungen "
        f"enthält (z. B. Haftpflicht und Teilkasko in einem PKW-Vertrag). "
        f"<b>Beiträge und Ergebnisse werden über alle Zeilen summiert, Verträge "
        f"dedupliziert auf ve_id gezählt.</b> Aktuell entfallen "
        f"{_fmt(kunden['n_positionen'].sum())} Positionen auf "
        f"{_fmt(kunden['n_vertraege'].sum())} Verträge "
        f"({_fmt(kunden['n_positionen'].sum() / max(kunden['n_vertraege'].sum(), 1), 2)} "
        f"Positionen je Vertrag).", st))
    s.append(Spacer(1, 10))

    s.append(Paragraph("Aufbau der Auswertungsebenen", st["h2"]))
    s.append(_hinweis(
        "<b>Ebene 1 (ve_bereich):</b> PKW gegenüber HUS - alles, was nicht PKW ist, "
        "zählt zu HUS. <b>Ebene 2 (ve_zweig):</b> innerhalb PKW die ve_sparte, innerhalb "
        "HUS das ve_produkt. So stehen sich nur gleichrangige Einheiten gegenüber.", st))
    s.append(Spacer(1, 10))

    eu = erg.get("ebenen_uebersicht")
    if eu is not None and len(eu):
        zeilen = [[f"{str(i[0])} / {str(i[1])[:24]}", str(r["herkunft"]),
                   _fmt(r["vertraege"]), _fmt(r["kunden"]), _pct(r["anteil_vertraege"], 1)]
                  for i, r in eu.iterrows()]
        s.append(_tabelle(["Bereich / Zweig", "Quelle", "Verträge", "Kunden",
                           "Anteil am Bestand"], zeilen,
                          [breite * 0.30, breite * 0.18, breite * 0.16, breite * 0.16,
                           breite * 0.20], st, text_spalten=(1,)))
        s.append(Spacer(1, 12))

    s.append(PageBreak())
    s.append(Paragraph("Sensitivität der Segmentgrenze", st["h1"]))
    s.append(Paragraph(
        "Die 20-Prozent-Grenze ist eine Setzung. Wie stark reagieren Segmentgröße und "
        "gebundenes Beitragsvolumen auf eine andere Wahl?", st["lead"]))
    s.append(_bild(_chart_sensitivitaet(erg["sensitivitaet"]), breite * 0.72))
    s.append(Spacer(1, 10))

    sens = erg.get("sensitivitaet")
    if sens is not None and len(sens):
        zeilen = [[_pct(i, 0), _fmt(r["kunden_negativ"]),
                   _pct(r["anteil_kunden_negativ"], 1), _pct(r["anteil_beitrag_negativ"], 1),
                   _eur_kurz(r["profit_negativ"])]
                  for i, r in sens.iterrows()]
        s.append(_tabelle(["Schwellenwert", "Kunden", "Anteil Kunden", "Anteil Beitrag",
                           "Ergebnis des Segments"], zeilen,
                          [breite * 0.18, breite * 0.16, breite * 0.20, breite * 0.20,
                           breite * 0.26], st))

    doc.build(s)
    return pfad
