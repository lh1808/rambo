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

Aufbau des Dokuments:
    1  Titelkopf und Management Summary (KPIs, Kernaussagen, Segmentanteile)
    2  Kundenstruktur (Verträge und Beiträge je Kunde)
    3  Portfoliomix (Sparten, Produkte)
    4  Verlusttreiber und Konzentration
    5  Methodik und Sensitivität
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
             st, rechts_ab: int = 1, hervorheben: Optional[Sequence[int]] = None) -> Table:
    th_r = ParagraphStyle("th_r", parent=st["th"], alignment=TA_RIGHT)
    daten = [[Paragraph(h, th_r if i >= rechts_ab else st["th"]) for i, h in enumerate(kopf)]]
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


def _chart_spartenmix(sm: pd.DataFrame, top: int = 8) -> plt.Figure:
    d = sm.copy()
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
def _kernaussagen(erg: Dict[str, pd.DataFrame]) -> List[str]:
    kz, kunden = erg["kennzahlen"], erg["kunden"]
    a: List[str] = []

    anteil_k = kz.loc["Kunden (Anteil)", "negativ"]
    anteil_b = kz.loc["Beitrag (Anteil am Gesamtbeitrag)", "negativ"]
    verlust = kz.loc["Profit gesamt", "negativ"]
    gewinn = kz.loc["Profit gesamt", "positiv"]
    a.append(f"<b>{_pct(anteil_k, 1)} der Kunden</b> ({_fmt(kz.loc['Kunden (Anzahl)', 'negativ'])}) "
             f"gelten nach der gewählten Definition als nicht wertvoll. Sie tragen "
             f"{_pct(anteil_b, 1)} des Beitragsvolumens und belasten das Ergebnis mit "
             f"<b>{_eur_kurz(verlust)}</b> gegenüber {_eur_kurz(gewinn)} aus dem wertvollen Segment.")

    v_neg = kz.loc["Verträge je Kunde (Mittel)", "negativ"]
    v_pos = kz.loc["Verträge je Kunde (Mittel)", "positiv"]
    richtung = "weniger" if v_neg < v_pos else "mehr"
    a.append(f"Nicht wertvolle Kunden haben im Mittel <b>{_fmt(v_neg, 2)} Verträge</b> "
             f"gegenüber {_fmt(v_pos, 2)} im wertvollen Segment - also {richtung} "
             f"Bindung über die Vertragsanzahl. Der Anteil der Einvertragskunden liegt bei "
             f"{_pct(kz.loc['Anteil Kunden mit nur 1 Vertrag', 'negativ'], 0)} gegenüber "
             f"{_pct(kz.loc['Anteil Kunden mit nur 1 Vertrag', 'positiv'], 0)}.")

    b_neg = kz.loc["Beitrag je Vertrag (gepoolt)", "negativ"]
    b_pos = kz.loc["Beitrag je Vertrag (gepoolt)", "positiv"]
    diff = b_neg / b_pos - 1 if b_pos else np.nan
    a.append(f"Der durchschnittliche Beitrag je Vertrag unterscheidet sich mit "
             f"<b>{_eur(b_neg)}</b> gegenüber {_eur(b_pos)} um {_pct(abs(diff), 1)}. "
             f"Der Ergebnisunterschied entsteht damit "
             f"{'kaum' if abs(diff) < 0.1 else 'auch'} über die Beitragshöhe, sondern "
             f"primär über Schaden- und Kostenquote "
             f"(CR {_fmt(kz.loc['CR gepoolt', 'negativ'], 2)} vs. "
             f"{_fmt(kz.loc['CR gepoolt', 'positiv'], 2)}).")

    konz = erg.get("konzentration")
    if konz is not None and len(konz):
        z10 = konz.loc[0.10]
        a.append(f"Der Verlust ist konzentriert: <b>Die 10 Prozent verlustreichsten Kunden "
                 f"({_fmt(z10['kunden'])}) verantworten {_pct(z10['anteil_am_negativen_profit'], 0)} "
                 f"des negativen Ergebnisses.</b> Maßnahmen können entsprechend eng "
                 f"fokussiert werden.")

    vt = erg.get("verlusttreiber_sparte")
    if vt is not None and len(vt):
        top = vt.index[0]
        a.append(f"Größter Ergebnistreiber im negativen Segment ist die Sparte "
                 f"<b>{top}</b> mit {_eur_kurz(vt.iloc[0]['profit'])} "
                 f"({_pct(vt.iloc[0]['anteil_am_gesamtverlust'], 0)} des Segmentverlusts) bei "
                 f"einer Marge von {_pct(vt.iloc[0]['marge'], 1)}.")

    anteil_vv = kz.loc["Anteil Verlustverträge je Kunde (Mittel)", "negativ"]
    a.append(f"Im Mittel sind {_pct(anteil_vv, 0)} der Verträge eines nicht wertvollen "
             f"Kunden für sich genommen defizitär (wertvolles Segment: "
             f"{_pct(kz.loc['Anteil Verlustverträge je Kunde (Mittel)', 'positiv'], 0)}). "
             f"Das zeigt, ob es sich um Einzelverträge oder um durchgängige "
             f"Fehlbepreisung handelt.")
    return a


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
    top_produkte: int = 10,
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

    # ------------------------------------------------------- Management Summary
    s.append(Paragraph("Management Summary", st["h1"]))
    s.append(Paragraph(
        "Der Bestand wurde je Kunde über alle Verträge aggregiert und in zwei Segmente "
        "geteilt: Kunden mit deutlich negativem Ergebnisbeitrag („nicht wertvoll“) und "
        "alle übrigen („wertvoll“). Die folgenden Seiten zeigen, worin sich beide "
        "Segmente strukturell unterscheiden.", st["lead"]))

    kpis = [
        (_pct(kz.loc["Kunden (Anteil)", "negativ"], 1), "Anteil nicht wertvoller Kunden"),
        (_pct(kz.loc["Beitrag (Anteil am Gesamtbeitrag)", "negativ"], 1),
         "davon gebundenes Beitragsvolumen"),
        (_eur_kurz(kz.loc["Profit gesamt", "negativ"]).replace(" EUR", ""),
         "Ergebnisbeitrag des Segments in EUR"),
        (_pct(kz.loc["Marge gepoolt (Profit/Beitrag)", "negativ"], 1), "Marge des Segments"),
    ]
    s.append(_kpi_kacheln(kpis, breite, st))
    s.append(Spacer(1, 12))
    s.append(_bild(_chart_anteile(kz), breite))
    s.append(Spacer(1, 10))
    s.append(Paragraph("Kernaussagen", st["h2"]))
    for a in _kernaussagen(erg):
        s.append(Paragraph(a, st["bullet"], bulletText="\u25aa"))
    s.append(PageBreak())

    # ---------------------------------------------------------- Kundenstruktur
    s.append(Paragraph("Kundenstruktur im Vergleich", st["h1"]))
    s.append(Paragraph("Anzahl der Verträge, Beitragsvolumen und Ergebnis je Segment.",
                       st["lead"]))

    auswahl = [
        ("Kunden (Anzahl)", "Kunden", _fmt),
        ("Verträge (Anzahl gesamt)", "Verträge", _fmt),
        ("Verträge je Kunde (Mittel)", "Verträge je Kunde (Mittel)", lambda x: _fmt(x, 2)),
        ("Verträge je Kunde (Median)", "Verträge je Kunde (Median)", lambda x: _fmt(x, 1)),
        ("Anteil Kunden mit nur 1 Vertrag", "Anteil Einvertragskunden", _pct),
        ("Sparten je Kunde (Mittel)", "Sparten je Kunde (Mittel)", lambda x: _fmt(x, 2)),
        ("Beitrag gesamt", "Beitrag gesamt", _eur_kurz),
        ("Beitrag je Kunde (Mittel)", "Beitrag je Kunde (Mittel)", _eur),
        ("Beitrag je Kunde (Median)", "Beitrag je Kunde (Median)", _eur),
        ("Beitrag je Vertrag (gepoolt)", "Beitrag je Vertrag", _eur),
        ("Profit gesamt", "Ergebnis gesamt", _eur_kurz),
        ("Profit je Kunde (Mittel)", "Ergebnis je Kunde (Mittel)", _eur),
        ("Marge gepoolt (Profit/Beitrag)", "Marge (gepoolt)", _pct),
        ("CR gepoolt", "Combined Ratio", lambda x: _fmt(x, 2)),
        ("Anteil Verlustverträge je Kunde (Mittel)", "Anteil defizitärer Verträge", _pct),
    ]
    zeilen = [[label, f(kz.loc[k, "negativ"]), f(kz.loc[k, "positiv"])]
              for k, label, f in auswahl if k in kz.index]
    s.append(_tabelle(["Kennzahl", "nicht wertvoll", "wertvoll"], zeilen,
                      [breite * 0.5, breite * 0.25, breite * 0.25], st))
    s.append(Spacer(1, 14))

    nebeneinander = Table(
        [[_bild(_chart_vertragsverteilung(erg["verteilung_vertragsanzahl"]), breite / 2 - 6),
          _bild(_chart_spartenmix(erg["sparten_mix"]), breite / 2 - 6)]],
        colWidths=[breite / 2, breite / 2], hAlign="LEFT",
        style=[("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
               ("VALIGN", (0, 0), (-1, -1), "TOP")])
    s.append(nebeneinander)
    s.append(Paragraph("Links: Verteilung der Kunden nach Vertragsanzahl. "
                       "Rechts: mittlere Zusammensetzung des Kundenbestands nach Sparten.",
                       st["klein"]))
    s.append(PageBreak())

    # ------------------------------------------------------------- Portfoliomix
    s.append(Paragraph("Zusammensetzung des Portfolios", st["h1"]))
    s.append(Paragraph(
        "Anteil einer Sparte am Vertragsbestand eines Kunden, gemittelt über alle Kunden "
        "des Segments. Die Durchdringung zeigt, welcher Anteil der Kunden mindestens einen "
        "Vertrag der Sparte hält.", st["lead"]))

    sm = erg["sparten_mix"]
    zeilen = [[str(i)[:28],
               _pct(r["anteil_vertraege_negativ"], 1), _pct(r["anteil_vertraege_positiv"], 1),
               _pct(r["delta_anteil_vertraege"], 1),
               _pct(r["penetration_negativ"], 0), _pct(r["penetration_positiv"], 0),
               _pct(r["marge_negativ"], 1), _pct(r["marge_positiv"], 1)]
              for i, r in sm.iterrows()]
    s.append(Paragraph("Sparten", st["h2"]))
    s.append(_tabelle(["Sparte", "Anteil<br/>n. wertv.", "Anteil<br/>wertvoll", "Delta",
                       "Durchdr.<br/>n. wertv.", "Durchdr.<br/>wertvoll",
                       "Marge<br/>n. wertv.", "Marge<br/>wertvoll"], zeilen,
                      [breite * 0.23] + [breite * 0.11] * 7, st))

    pm = erg["produkt_mix"].head(top_produkte)
    zeilen = [[str(i)[:30],
               _pct(r["anteil_vertraege_negativ"], 1), _pct(r["anteil_vertraege_positiv"], 1),
               _pct(r["delta_anteil_vertraege"], 1),
               _eur(r["beitrag_je_vertrag_negativ"]), _eur(r["beitrag_je_vertrag_positiv"]),
               _pct(r["marge_negativ"], 1)]
              for i, r in pm.iterrows()]
    s.append(Paragraph("Produkte (Top nach Anteil im Segment „nicht wertvoll“)", st["h2"]))
    s.append(_tabelle(["Produkt", "Anteil<br/>n. wertv.", "Anteil<br/>wertvoll", "Delta",
                       "Beitrag/Vertrag<br/>n. wertv.", "Beitrag/Vertrag<br/>wertvoll",
                       "Marge<br/>n. wertv."], zeilen,
                      [breite * 0.24, breite * 0.11, breite * 0.11, breite * 0.1,
                       breite * 0.16, breite * 0.16, breite * 0.12], st))
    s.append(PageBreak())

    # -------------------------------------------------- Verlusttreiber/Konzentration
    s.append(Paragraph("Verlusttreiber und Konzentration", st["h1"]))
    s.append(Paragraph(
        "Wo entsteht der negative Ergebnisbeitrag - und wie breit ist er im Segment "
        "gestreut? Die Antwort bestimmt, ob eine Maßnahme breit ansetzen muss oder "
        "auf wenige Kunden fokussiert werden kann.", st["lead"]))

    nebeneinander = Table(
        [[_bild(_chart_verlusttreiber(erg["verlusttreiber_produkt"]), breite / 2 - 6),
          _bild(_chart_konzentration(kunden), breite / 2 - 6)]],
        colWidths=[breite / 2, breite / 2], hAlign="LEFT",
        style=[("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
               ("VALIGN", (0, 0), (-1, -1), "TOP")])
    s.append(nebeneinander)
    s.append(Paragraph("Links: Produkte mit dem größten negativen Ergebnisbeitrag im "
                       "Segment. Rechts: Konzentration des Verlusts (Lorenzkurve).",
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
    s.append(PageBreak())

    # ---------------------------------------------------- Methodik/Sensitivitaet
    s.append(Paragraph("Methodik und Robustheit", st["h1"]))
    s.append(Paragraph("Definition der Segmentierung", st["h2"]))
    s.append(_hinweis(
        f"Je Kunde (vn_partner_id) werden Beitrag und Ergebnis über alle Verträge "
        f"summiert. Ein Kunde gilt als <b>nicht wertvoll</b>, wenn das aggregierte Ergebnis "
        f"negativ ist <b>und</b> betragsmäßig mindestens 20 Prozent des aggregierten "
        f"Bestandsjahresnettobeitrags erreicht. Formal: {regel}. Alle übrigen Kunden "
        f"gelten als wertvoll.", st))
    s.append(Spacer(1, 12))

    s.append(Paragraph("Sensitivität der 20-Prozent-Grenze", st["h2"]))
    s.append(Paragraph(
        "Die Grenze ist eine Setzung. Die Grafik zeigt, wie stark Segmentgröße und "
        "gebundenes Beitragsvolumen auf eine andere Wahl reagieren.", st["text"]))
    s.append(_bild(_chart_sensitivitaet(erg["sensitivitaet"]), breite * 0.62))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Lesehinweise und Einschränkungen", st["h2"]))
    for t in [
        "Die Betrachtung ist eine Momentaufnahme des Bestands ohne Kundenhistorie; "
        "junge Kunden mit erwartbar steigender Vertragsanzahl werden wie etablierte "
        "Kunden bewertet.",
        "Das Ergebnis je Vertrag beruht auf erwarteten Schadenaufwänden. Einzelne "
        "Großschäden können einen Kunden ins negative Segment schieben, ohne dass "
        "dies für die Zukunft aussagekräftig ist.",
        "Cross- und Up-Selling-Potenzial sowie Kundenbindungseffekte sind nicht "
        "eingerechnet. Ein heute defizitärer Kunde kann über den Lebenszyklus "
        "wertvoll sein.",
        "Die Segmentierung ist deskriptiv. Für die Frage, welche Maßnahme bei welchem "
        "Kunden wirkt, ist eine kausale Wirkungsanalyse erforderlich.",
    ]:
        s.append(Paragraph(t, st["bullet"], bulletText="\u25aa"))

    doc.build(s)
    return pfad
