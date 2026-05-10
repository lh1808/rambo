from __future__ import annotations

"""Evaluation mit EconML-DRTester und scikit-uplift Plots.

Hintergrund
-----------
EconML bringt mit dem `DRTester` eine Validierungsroutine mit, die u. a.
- BLP-Tests,
- Calibration,
- Qini-/TOC-basierte Uplift-Tests
bereitstellt.

rubin erzeugt CATEs als Cross-Predictions (Out-of-Fold), damit
Evaluationskennzahlen nicht optimistisch verzerrt sind. Der Standard-`DRTester`
erwartet jedoch ein CATE-Modell, aus dem er Vorhersagen selbst erzeugt.
Da die CATE-Vorhersagen in rubin bereits vorliegen, kapselt dieses Modul
eine angepasste DRTester-Variante (`CustomDRTester`), die vorberechnete
CATE-Werte direkt akzeptiert.

Zusätzlich werden Plots aus scikit-uplift (sklift) erzeugt, die für die
visuelle Beurteilung der Sortierung genutzt werden."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")  # Headless-Backend für Batch-/Server-Umgebungen
import matplotlib.pyplot as plt
import logging as _logging
_logging.getLogger("matplotlib.category").setLevel(_logging.WARNING + 1)
from pandas.plotting import table

from rubin.utils.plot_theme import (
    apply_rubin_theme, RUBIN_COLORS, RUBIN_PALETTE,
    COLOR_MODEL, COLOR_MODEL_FILL, COLOR_REFERENCE, COLOR_REFERENCE_FILL,
    COLOR_BASELINE, COLOR_DIFFERENCE, COLOR_HIGHLIGHT_BOX,
    recolor_figure,
)
apply_rubin_theme()

from econml.validate import EvaluationResults
from econml.validate.drtester import DRTester
from econml.validate.utils import calculate_dr_outcomes

# scikit-uplift (sklift) wird NICHT mehr verwendet.
# Alle 3 Visualisierungen (Qini, Uplift-by-Percentile, Treatment-Balance) sind
# nativ implementiert mit rubin-Farbpalette. sklift 0.5.1 hat einen bekannten
# numpy >=1.24 Kompatibilitäts-Bug (GitHub Issue #213) und wurde seit 2022
# nicht mehr aktualisiert.


def save_dataframe_as_png(df: pd.DataFrame, filename: str) -> str:
    """Speichert ein DataFrame als PNG-Tabelle.
In MLflow sind Tabellen als Bild oft schneller zu sichten als als CSV.
Die CSV wird in rubin an anderer Stelle ohnehin zusätzlich geloggt."""

    if df is None or len(df) == 0 or len(df.columns) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "Keine Daten verfügbar", ha="center", va="center",
                fontsize=12, color="#888")
        fig.savefig(filename, bbox_inches="tight", dpi=100)
        plt.close(fig)
        return filename

    # Adaptive Größe: Breite nach Spaltenanzahl, Höhe nach Zeilenanzahl
    n_cols = len(df.columns)
    n_rows = len(df)
    col_width = max(0.08, min(0.18, 0.9 / n_cols))
    fig_width = max(8, n_cols * 2.2)
    fig_height = max(2, 0.5 * (n_rows + 1) + 0.5)

    # DataFrame-Werte für Anzeige runden
    df_display = df.copy()
    for col in df_display.columns:
        if df_display[col].dtype in ('float64', 'float32'):
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.6g}" if pd.notna(x) else "")

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    tbl = table(ax, df_display, loc="center", cellLoc="center",
                colWidths=[col_width] * n_cols)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.4)
    # Rubin-Theme: Header-Zeile einfärben
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(RUBIN_COLORS["ruby"])
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor(RUBIN_COLORS["ruby_dark"])
        else:
            cell.set_facecolor(RUBIN_COLORS["ruby_pale"] if row % 2 == 0 else RUBIN_COLORS["white"])
            cell.set_edgecolor(RUBIN_COLORS["grid"])
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=120)
    plt.close(fig)
    return filename


class CustomEvaluationResults(EvaluationResults):
    """Erweiterung der EconML-Ergebnisse um Policy-Value-Extraktion.
Die EconML-Qini-Auswertung enthält Kurvenpunkte und Unsicherheiten.
Für den operativen Vergleich ist es praktisch, Policy Values für feste
Treat-Anteile als Tabelle auszugeben.

Unterstützt Partial-Failure: Wenn einzelne Sub-Evaluationen fehlschlagen,
werden die funktionierenden Teile trotzdem korrekt gespeichert."""

    def __init__(self, *, blp_res=None, cal_res=None, qini_res=None, toc_res=None):
        # Wenn alle Sub-Results vorhanden → Parent-__init__ nutzen (setzt ggf.
        # zusätzliche Attribute für plot_cal, plot_qini, plot_toc).
        # Nur bei Partial-Failure → direkte Zuweisung (Parent crasht bei None).
        if blp_res is not None and cal_res is not None and qini_res is not None and toc_res is not None:
            try:
                super().__init__(blp_res=blp_res, cal_res=cal_res, qini_res=qini_res, toc_res=toc_res)
                return
            except Exception:
                pass
        # Fallback: direkte Zuweisung für Partial-Failure
        self.blp = blp_res
        self.cal = cal_res
        self.qini = qini_res
        self.toc = toc_res

    def summary(self, *, tmt=0):
        """Erzeugt eine Summary-Tabelle.

        Versucht zuerst die EconML-Basisklasse (produziert die vollständige
        Tabelle mit BLP + CAL + Qini/TOC). Fällt nur bei Partial-Failure
        (None-Sub-Results) auf eine partielle Tabelle zurück."""
        # Normalfall: Parent-summary (wie in der funktionierenden Version)
        try:
            return super().summary()
        except Exception:
            pass
        # Fallback: Partielle Summary aus vorhandenen Sub-Results
        rows = []
        if self.blp is not None:
            try:
                blp_df = self.blp.summary_frame()
                for idx, row in blp_df.iterrows():
                    rows.append({"test": f"BLP ({idx})", "coef": row.get("coef", None),
                                 "std err": row.get("std err", None), "pvalue": row.get("P>|t|", row.get("pvalue", None))})
            except Exception:
                try:
                    rows.append({"test": "BLP", "coef": float(self.blp.params[1]) if len(self.blp.params) > 1 else None,
                                 "pvalue": float(self.blp.pvalues[1]) if len(self.blp.pvalues) > 1 else None})
                except Exception:
                    pass
        if self.cal is not None:
            try:
                cal_df = self.cal.summary_frame()
                for idx, row in cal_df.iterrows():
                    rows.append({"test": f"CAL ({idx})", "coef": row.get("coef", None),
                                 "std err": row.get("std err", None), "pvalue": row.get("P>|t|", row.get("pvalue", None))})
            except Exception:
                pass
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_policy_values(
        self,
        tmt: int,
        treated_percentages: List[float] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        conf_interval_type: str = "normal",  # "normal" | "ucb2" | "ucb1"
    ) -> pd.DataFrame:
        # Guard: Wenn evaluate_uplift(qini) fehlgeschlagen ist
        if self.qini is None:
            return pd.DataFrame()

        # Kurvendaten für das gewählte Treatment
        qini_curve = self.qini.curves[self.qini.treatments[tmt]]

        out: list[dict[str, float]] = []
        for p in treated_percentages:
            # Die Qini-Kurve arbeitet auf einer "Percentage treated" Skala.
            closest_idx = (qini_curve["Percentage treated"] - p).abs().idxmin()
            policy_value = float(qini_curve.loc[closest_idx, "value"])
            err = float(qini_curve.loc[closest_idx, "err"])
            ucb2 = float(qini_curve.loc[closest_idx, "uniform_critical_value"])
            ucb1 = float(qini_curve.loc[closest_idx, "uniform_one_side_critical_value"])

            if conf_interval_type == "normal":
                lo, hi = policy_value - 1.96 * err, policy_value + 1.96 * err
            elif conf_interval_type == "ucb2":
                lo, hi = policy_value - ucb2 * err, policy_value + ucb2 * err
            elif conf_interval_type == "ucb1":
                lo, hi = policy_value - ucb1 * err, policy_value
            else:
                raise ValueError(
                    f"Ungültiger conf_interval_type: {conf_interval_type}. Erlaubt: 'normal', 'ucb2', 'ucb1'."
                )

            out.append(
                {
                    "treated_percentage": float(p),
                    "policy_value": round(policy_value, 6),
                    "lower_bound": round(float(lo), 6),
                    "upper_bound": round(float(hi), 6),
                }
            )

        return pd.DataFrame(out)


class CustomDRTester(DRTester):
    """DRTester, der vorberechnete CATE-Vorhersagen akzeptiert.
Motivation
----------
In rubin werden CATEs in der Analyse typischerweise als Cross-Predictions
erzeugt. Damit DRTester die gleichen Werte nutzen kann, werden die
Vorhersagen an den Tester übergeben.
Hinweis
-------
`cate_preds_val` und `cate_preds_train` müssen zur jeweiligen X/T/Y-Menge
passen (gleiche Reihenfolge/Länge)."""

    def __init__(
        self,
        *,
        model_regression: Any,
        model_propensity: Any,
        cate: Any = None,
        cv: Union[int, List] = 5,
        cate_preds_val: Optional[np.ndarray] = None,
        cate_preds_train: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            model_regression=model_regression,
            model_propensity=model_propensity,
            cate=cate,
            cv=cv,
        )

        # EconML erwartet intern shape (n, 1)
        self.cate_preds_val_ = None
        self.cate_preds_train_ = None

        if cate_preds_val is not None:
            self.cate_preds_val_ = np.asarray(cate_preds_val).reshape(-1, 1)
        if cate_preds_train is not None:
            self.cate_preds_train_ = np.asarray(cate_preds_train).reshape(-1, 1)

        if self.cate_preds_val_ is None and cate is None:
            raise ValueError("Entweder cate oder cate_preds_val muss angegeben werden.")

    def fit_nuisance(
        self,
        Xval: np.ndarray,
        Dval: np.ndarray,
        yval: np.ndarray,
        Xtrain: Optional[np.ndarray] = None,
        Dtrain: Optional[np.ndarray] = None,
        ytrain: Optional[np.ndarray] = None,
    ):
        """Erzeugt Nuisance-Preds und DR-Outcomes über Cross-Fitting.

        Überschreibt die EconML-Methode, um sicherzustellen, dass Train-/Val-Daten
        konsistent verarbeitet werden, wenn beide Mengen übergeben werden."""

        self.Dval = Dval
        self.treatments = np.sort(np.unique(Dval))
        self.n_treat = len(self.treatments) - 1

        self.fit_on_train = (Xtrain is not None) and (Dtrain is not None) and (ytrain is not None)

        if self.fit_on_train:
            reg_preds_train, prop_preds_train = self.fit_nuisance_cv(Xtrain, Dtrain, ytrain)
            self.dr_train_ = self._sanitize_dr(calculate_dr_outcomes(Dtrain, ytrain, reg_preds_train, prop_preds_train))

            reg_preds_val, prop_preds_val = self.fit_nuisance_cv(Xval, Dval, yval)
            self.dr_val_ = self._sanitize_dr(calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val))
        else:
            reg_preds_val, prop_preds_val = self.fit_nuisance_cv(Xval, Dval, yval)
            self.dr_val_ = self._sanitize_dr(calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val))

        self.ate_val = self.dr_val_.mean(axis=0)
        return self

    @staticmethod
    def _sanitize_dr(arr: np.ndarray) -> np.ndarray:
        """Ersetzt NaN/Inf in DR-Outcomes durch Clip auf [1., 99.]-Perzentil.
        Propensity-Scores nahe 0/1 erzeugen extreme DR-Werte, die statsmodels OLS
        zum Absturz bringen ('exog contains inf or nans')."""
        mask = ~np.isfinite(arr)
        if not mask.any():
            return arr
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return np.zeros_like(arr)
        lo, hi = np.percentile(finite, [1, 99])
        out = arr.copy()
        out[mask] = np.clip(out[mask], lo, hi)
        # Additional: replace remaining NaN (if percentile itself was NaN)
        out = np.nan_to_num(out, nan=0.0, posinf=hi, neginf=lo)
        return out

    def get_cate_preds(self, Xval: np.ndarray, Xtrain: Optional[np.ndarray] = None) -> None:
        # Falls Val-Preds übergeben wurden, nichts weiter tun.
        if self.cate_preds_val_ is None:
            if self.cate is None:
                raise ValueError("CATE-Modell ist nicht gesetzt, und keine cate_preds_val angegeben.")
            base = self.treatments[0]
            vals = [self.cate.effect(X=Xval, T0=base, T1=t) for t in self.treatments[1:]]
            self.cate_preds_val_ = np.stack(vals).T

        if Xtrain is not None:
            if self.cate_preds_train_ is None:
                if self.cate is None:
                    raise ValueError("CATE-Modell ist nicht gesetzt, und keine cate_preds_train angegeben.")
                base = self.treatments[0]
                trains = [self.cate.effect(X=Xtrain, T0=base, T1=t) for t in self.treatments[1:]]
                self.cate_preds_train_ = np.stack(trains).T

    def evaluate_all(
        self,
        Xval: Optional[np.ndarray] = None,
        Xtrain: Optional[np.ndarray] = None,
        n_groups: int = 10,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
    ) -> CustomEvaluationResults:
        """Führt alle DRTester-Tests aus und liefert CustomEvaluationResults.

        n_bootstrap steuert die Anzahl der Bootstrap-Iterationen für Qini/TOC
        Konfidenzintervalle. Weniger Iterationen → schneller, breitere CIs.
        seed: Setzt np.random.seed vor Bootstrap-Aufrufen für Reproduzierbarkeit."""

        if (not hasattr(self, "cate_preds_val_")) or (self.cate_preds_val_ is None):
            if Xval is None:
                raise ValueError("CATE-Preds sind nicht gesetzt. Xval muss angegeben werden.")
            self.get_cate_preds(Xval, Xtrain)

        # Sanitisierung: EconML's evaluate_blp() ruft
        #   OLS(self.dr_val_, add_constant(self.cate_preds_val_))
        # Beide Arrays müssen NaN/Inf-frei sein, sonst crasht statsmodels.
        self.dr_val_ = self._sanitize_dr(self.dr_val_)
        if self.cate_preds_val_ is not None:
            self.cate_preds_val_ = self._sanitize_dr(self.cate_preds_val_)
        if hasattr(self, 'dr_train_') and self.dr_train_ is not None:
            self.dr_train_ = self._sanitize_dr(self.dr_train_)
        if hasattr(self, 'cate_preds_train_') and self.cate_preds_train_ is not None:
            self.cate_preds_train_ = self._sanitize_dr(self.cate_preds_train_)

        # Fallback: EconML's evaluate_cal() benötigt cate_preds_train_ und
        # dr_train_ für die Quantil-Berechnung. Wenn kein Train-Split
        # vorhanden ist (z.B. Ensemble-Modell, fehlende Train-CATEs),
        # verwende die Val-Daten als Fallback. Weniger rigoros, aber die
        # Calibration-Plots bleiben informativ.
        if not hasattr(self, 'cate_preds_train_') or self.cate_preds_train_ is None:
            self.cate_preds_train_ = self.cate_preds_val_
        if not hasattr(self, 'dr_train_') or self.dr_train_ is None:
            self.dr_train_ = self.dr_val_

        # DRTester-Berechnungen können bei Modellen mit geringer CATE-Varianz
        # (z.B. CausalForest) leere Quantil-Bins erzeugen → np.mean([]) gibt
        # NaN + RuntimeWarning. Das ist erwartetes Verhalten, keine Fehlfunktion.
        import warnings
        with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            warnings.filterwarnings("ignore", message="invalid value encountered")

            blp_res = self.evaluate_blp()
            cal_res = self.evaluate_cal(n_groups=n_groups)
            # Seed vor Bootstrap setzen — EconML's evaluate_uplift nutzt intern
            # np.random für Bootstrap-Resampling ohne eigenen seed-Parameter.
            if seed is not None:
                np.random.seed(seed)
            qini_res = self.evaluate_uplift(metric="qini", n_bootstrap=n_bootstrap)
            if seed is not None:
                np.random.seed(seed + 1)
            toc_res = self.evaluate_uplift(metric="toc", n_bootstrap=n_bootstrap)

        self.res = CustomEvaluationResults(blp_res=blp_res, cal_res=cal_res, qini_res=qini_res, toc_res=toc_res)
        return self.res


@dataclass
class DrTesterPlotBundle:
    """Sammelt alle Plots und Tabellen aus einer DRTester + Uplift Evaluation.

    Bietet Convenience-Methoden für MLflow-Logging, HTML-Report und Figure-Freigabe."""

    summary: pd.DataFrame
    cal_plot: plt.Figure
    qini_plot: plt.Figure
    toc_plot: plt.Figure
    policy_values: pd.DataFrame
    sklift_qini: Optional[plt.Figure]
    sklift_percentile: Optional[plt.Figure]
    treatment_balance: Optional[plt.Figure]

    _PLOT_FIELDS = ["cal_plot", "qini_plot", "toc_plot",
                    "sklift_qini", "sklift_percentile", "treatment_balance"]

    def all_figures(self) -> list:
        """Alle nicht-None Figures als (name, fig)-Paare."""
        return [(name, getattr(self, name)) for name in self._PLOT_FIELDS
                if getattr(self, name) is not None]

    def log_to_mlflow(self, mlflow, label: str, log_temp_artifact_fn=None) -> None:
        """Loggt Summary, Policy Values und alle Plots nach MLflow."""
        if log_temp_artifact_fn is not None:
            log_temp_artifact_fn(mlflow,
                                 lambda p: save_dataframe_as_png(self.summary, p),
                                 f"summary__{label}.png")
            log_temp_artifact_fn(mlflow,
                                 lambda p: save_dataframe_as_png(self.policy_values, p),
                                 f"policy_values__{label}.png")
        for name, fig in self.all_figures():
            # Render einmal + Cache für fig_to_base64 (vermeidet doppeltes savefig)
            import io as _io
            _buf = _io.BytesIO()
            fig.savefig(_buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
            _buf.seek(0)
            _png = _buf.read()
            _buf.close()
            fig._rubin_png_cache = _png
            if log_temp_artifact_fn is not None:
                log_temp_artifact_fn(mlflow,
                                     lambda p, _b=_png: open(p, "wb").write(_b),
                                     f"{name}__{label}.png")
            else:
                mlflow.log_figure(fig, f"{name}__{label}.png")

    def add_to_report(self, report, label: str) -> None:
        """Registriert alle Plots im HTML-Report."""
        for name, fig in self.all_figures():
            report.add_plot(label, name, fig)

    def close_figures(self) -> None:
        """Gibt alle Figure-Objekte frei (nach MLflow-Logging + Report)."""
        for _, fig in self.all_figures():
            plt.close(fig)


def filter_tester_for_mask(
    fitted_tester: CustomDRTester,
    eval_mask: np.ndarray,
    n_eval: int,
) -> Optional[CustomDRTester]:
    """Filtert einen Pre-Fit-DRTester auf ein Eval-Mask-Subset.

    Wird bei Train-Many-Evaluate-Some benötigt. Returns None bei Fehler."""
    try:
        ft = CustomDRTester(
            model_regression=getattr(fitted_tester, '_model_regression_ref', None),
            model_propensity=getattr(fitted_tester, '_model_propensity_ref', None),
            cate=None,
            cate_preds_val=np.zeros(n_eval),
        )
        if not hasattr(fitted_tester, 'dr_val_') or fitted_tester.dr_val_ is None:
            return None
        ft.dr_val_ = fitted_tester.dr_val_[eval_mask]
        ft.ate_val = ft.dr_val_.mean(axis=0)
        ft.Dval = fitted_tester.Dval[eval_mask] if hasattr(fitted_tester, 'Dval') else None
        ft.treatments = fitted_tester.treatments
        ft.n_treat = fitted_tester.n_treat
        ft.fit_on_train = fitted_tester.fit_on_train
        if hasattr(fitted_tester, 'dr_train_'):
            ft.dr_train_ = fitted_tester.dr_train_
        return ft
    except Exception:
        _logger.debug("filter_tester_for_mask fehlgeschlagen.")
        return None


def fit_drtester_nuisance(
    *,
    model_regression: Any,
    model_propensity: Any,
    X_val: pd.DataFrame,
    T_val: np.ndarray,
    Y_val: np.ndarray,
    X_train: Optional[pd.DataFrame] = None,
    T_train: Optional[np.ndarray] = None,
    Y_train: Optional[np.ndarray] = None,
    cv: int = 5,
    seed: int = 42,
) -> CustomDRTester:
    """Erstellt und fittet einen DRTester NUR für die Nuisance-Modelle (Outcome + Propensity).

    Die Nuisance-Ergebnisse (DR-Outcomes, ATE) sind für alle kausalen Modelle gleich
    und müssen nur einmal berechnet werden. Der zurückgegebene Tester kann dann
    mit evaluate_cate_with_plots(fitted_tester=...) wiederverwendet werden.

    cv: Anzahl Cross-Fitting-Folds für Nuisance-Predictions (Default: 5).
        Weniger Folds = schneller, leicht ungenauere DR-Outcomes."""
    from sklearn.model_selection import StratifiedKFold

    # StratifiedKFold auf T×Y für balancierte Treatment- und Outcome-Gruppen pro Fold
    _dr_strata = np.asarray(T_val).ravel().astype(int) * 10 + np.clip(np.asarray(Y_val).ravel(), 0, 1).astype(int)

    class _TxYSplitter:
        """Wrapper: DRTester ruft cv.split(X, D) mit nur T — wir erzwingen T×Y."""
        def __init__(self, skf, strata):
            self._skf, self._strata = skf, strata
        def split(self, X, y=None, groups=None):
            return self._skf.split(X, self._strata)
        def get_n_splits(self, *a, **kw):
            return self._skf.get_n_splits(*a, **kw)

    cv_splitter = _TxYSplitter(
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        _dr_strata,
    )

    # Dummy-CATE-Preds (werden bei der Nuisance-Fitphase nicht benötigt,
    # aber der Konstruktor verlangt entweder cate oder cate_preds_val)
    dummy_cate = np.zeros(len(X_val))

    tester = CustomDRTester(
        model_regression=model_regression,
        model_propensity=model_propensity,
        cate=None,
        cate_preds_val=dummy_cate,
        cv=cv_splitter,
    )
    # Referenzen speichern für spätere Nuisance-Wiederverwendung
    tester._model_regression_ref = model_regression
    tester._model_propensity_ref = model_propensity

    if X_train is not None and T_train is not None and Y_train is not None:
        tester.fit_nuisance(
            Xval=X_val.values,
            Dval=np.asarray(T_val).ravel(),
            yval=np.asarray(Y_val).ravel(),
            Xtrain=X_train.values,
            Dtrain=np.asarray(T_train).ravel(),
            ytrain=np.asarray(Y_train).ravel(),
        )
    else:
        tester.fit_nuisance(
            Xval=X_val.values,
            Dval=np.asarray(T_val).ravel(),
            yval=np.asarray(Y_val).ravel(),
        )

    return tester


def generate_cate_distribution_plot(
    cate_preds_val: np.ndarray,
    cate_preds_train: Optional[np.ndarray] = None,
    model_name: str = "",
    arm_label: str = "",
) -> Optional[Any]:
    """Erzeugt ein Histogramm der CATE-Predictions (Training + Cross-Validated).

    Zeigt die Verteilung der vorhergesagten Treatment-Effekte. Dient der
    visuellen Plausibilitätsprüfung: stark konzentrierte Verteilungen nahe
    Null deuten auf wenig Heterogenität hin, breite Verteilungen auf
    differenzierte Effektvorhersagen.

    Returns
    -------
    matplotlib.figure.Figure oder None bei Fehler."""
    from rubin.utils.plot_theme import apply_rubin_theme, RUBIN_COLORS
    apply_rubin_theme()

    val = np.asarray(cate_preds_val, dtype=float).ravel()
    val = val[~np.isnan(val)]
    if len(val) == 0:
        return None

    has_train = cate_preds_train is not None
    if has_train:
        train = np.asarray(cate_preds_train, dtype=float).ravel()
        train = train[~np.isnan(train)]
        has_train = len(train) > 0

    suffix = f" {arm_label}" if arm_label else ""
    n_cols = 2 if has_train else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    # Gemeinsamer Bin-Bereich für Vergleichbarkeit
    all_vals = np.concatenate([val, train]) if has_train else val
    q_lo, q_hi = np.percentile(all_vals, [0.5, 99.5])
    bins = np.linspace(q_lo, q_hi, 80)

    color = RUBIN_COLORS["ruby"]
    edge_color = RUBIN_COLORS["ruby_dark"]

    if has_train:
        ax_train = axes[0]
        ax_train.hist(train, bins=bins, color=color, edgecolor=edge_color,
                      linewidth=0.3, alpha=0.85)
        ax_train.set_title(f"Training Predictions{suffix}", fontsize=12, fontweight="bold")
        ax_train.set_xlabel(f"Train_{model_name}{suffix}", fontsize=10)
        ax_train.set_ylabel("Count", fontsize=10)
        ax_train.axvline(0, color=RUBIN_COLORS["slate"], linewidth=1, linestyle="--", alpha=0.6)

    ax_val = axes[-1]
    ax_val.hist(val, bins=bins, color=color, edgecolor=edge_color,
                linewidth=0.3, alpha=0.85)
    ax_val.set_title(f"Cross-Validated Predictions{suffix}", fontsize=12, fontweight="bold")
    ax_val.set_xlabel(f"Predictions_{model_name}{suffix}", fontsize=10)
    ax_val.set_ylabel("Count", fontsize=10)
    ax_val.axvline(0, color=RUBIN_COLORS["slate"], linewidth=1, linestyle="--", alpha=0.6)

    fig.suptitle(f"CATE Distribution — {model_name}{suffix}", fontsize=14,
                 fontweight="bold", color=RUBIN_COLORS["ruby_dark"], y=1.02)
    fig.tight_layout()
    return fig


def generate_ate_barplot(
    T: np.ndarray, Y: np.ndarray,
) -> Optional[plt.Figure]:
    """Barplot der Outcome-Rate pro Treatment-Gruppe mit ATE-Annotation.

    Zeigt Response Rates (E[Y|T=k]) pro Gruppe als Balken mit
    Standard-Error-Balken und ATE-Differenz als Annotation.
    Direkt in rubin-Farbpalette, kein recolor nötig."""
    try:
        from rubin.utils.plot_theme import RUBIN_COLORS
        from scipy import stats as _stats

        t_arr = np.asarray(T).ravel()
        y_arr = np.asarray(Y).ravel().astype(float)
        groups = sorted(np.unique(t_arr).tolist())
        n_groups = len(groups)

        rates = []
        ses = []
        ns = []
        labels = []
        for g in groups:
            mask = t_arr == g
            n_g = mask.sum()
            rate = float(y_arr[mask].mean()) if n_g > 0 else 0.0
            se = float(y_arr[mask].std(ddof=1)) / np.sqrt(n_g) if n_g > 1 else 0.0
            rates.append(rate)
            ses.append(se)
            ns.append(n_g)
            labels.append(f"T={int(g)}" if n_groups <= 5 else f"{int(g)}")

        # Farben: Control = Gold, Treatment(s) = Ruby-Palette
        if n_groups == 2:
            colors = [RUBIN_COLORS["gold"], RUBIN_COLORS["ruby"]]
        else:
            palette = [RUBIN_COLORS["gold"], RUBIN_COLORS["ruby"], RUBIN_COLORS["ruby_dark"],
                       RUBIN_COLORS["ruby_light"], "#8B6914", "#E07A5F"]
            colors = [palette[i % len(palette)] for i in range(n_groups)]

        fig, ax = plt.subplots(figsize=(max(4, n_groups * 1.5 + 1), 4.5))
        x = np.arange(n_groups)
        bars = ax.bar(x, rates, yerr=ses, width=0.5, color=colors,
                      edgecolor="black", linewidth=0.5, alpha=0.85,
                      capsize=5, error_kw={"linewidth": 1.2})

        # Werte über den Balken
        for i, (r, se, n_g) in enumerate(zip(rates, ses, ns)):
            ax.text(i, r + se + max(rates) * 0.02, f"{r:.4f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=RUBIN_COLORS["text"])
            ax.text(i, r / 2, f"n={n_g:,}",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="600")

        # ATE-Annotation
        if n_groups == 2:
            ate = rates[1] - rates[0]
            ax.annotate(
                "", xy=(1, rates[1]), xytext=(0, rates[0]),
                arrowprops=dict(arrowstyle="<->", color=RUBIN_COLORS["slate"], lw=1.5),
            )
            mid_y = (rates[0] + rates[1]) / 2
            ax.text(0.5, mid_y, f"ATE = {ate:+.4f}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=RUBIN_COLORS["ruby_dark"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=RUBIN_COLORS["ruby_dark"], alpha=0.9))
        elif n_groups > 2:
            # MT: ATE pro Arm vs. Control (T=0) als kleine Labels über den Balken
            for i in range(1, n_groups):
                ate_arm = rates[i] - rates[0]
                ax.text(i, rates[i] + ses[i] + max(rates) * 0.09,
                        f"Δ{ate_arm:+.4f}",
                        ha="center", va="bottom", fontsize=8.5,
                        color=RUBIN_COLORS["slate"],
                        fontstyle="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Outcome Rate  E[Y | T]", fontsize=11)
        ax.set_title("Outcome Rate by Treatment Group", fontsize=13, fontweight="bold",
                     color=RUBIN_COLORS["ruby_dark"])
        ax.set_ylim(0, max(rates) * (1.35 if n_groups > 2 else 1.25))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig
    except Exception as e:
        _logger.warning("ATE-Barplot fehlgeschlagen: %s", e)
        return None


def _native_uplift_by_percentile(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray,
    strategy: str = "overall", kind: str = "bar",
    n_bins: int = 10, string_percentiles: bool = True,
) -> Optional[plt.Figure]:
    """Uplift-by-Percentile Plot — feature-complete Ersatz für sklift.

    Repliziert ``sklift.viz.plot_uplift_by_percentile`` vollständig inkl. aller
    Parameter-Optionen (strategy, kind, bins, string_percentiles).

    Args:
        strategy: 'overall' (Default) — sortiert ALLE Daten nach Uplift, teilt in Bins.
                  'by_group' — sortiert Treatment/Control getrennt, nimmt Top-k% je Gruppe.
        kind: 'bar' (Default) — 2-Panel Barplot (Uplift oben, Response Rates unten).
              'line' — Einzelner Lineplot mit Fehlerbalken und Fill.
        n_bins: Anzahl Bins (Default 10).
        string_percentiles: True → X-Labels als Strings ('0-10', '10-20', ...).

    Hintergrund: sklift 0.5.1 hat bekannten numpy >=1.24 Bug (GitHub #213)."""
    try:
        from rubin.utils.plot_theme import RUBIN_COLORS, COLOR_MODEL, COLOR_REFERENCE, COLOR_BASELINE

        y_arr = np.asarray(y_true).ravel()
        u_arr = np.asarray(uplift).ravel()
        t_arr = np.asarray(treatment).ravel()

        # ── Berechnung der Perzentil-Metriken ──
        if strategy == "by_group":
            # Getrennte Sortierung: Top-k% Treatment vs Top-k% Control
            treat_idx = np.where(t_arr == 1)[0]
            ctrl_idx = np.where(t_arr == 0)[0]
            treat_order = treat_idx[np.argsort(u_arr[treat_idx])[::-1]]
            ctrl_order = ctrl_idx[np.argsort(u_arr[ctrl_idx])[::-1]]
            treat_edges = np.linspace(0, len(treat_order), n_bins + 1, dtype=int)
            ctrl_edges = np.linspace(0, len(ctrl_order), n_bins + 1, dtype=int)
            rr_treat, rr_ctrl, std_treat, std_ctrl = [], [], [], []
            n_treat_bins = []
            for i in range(n_bins):
                t_lo, t_hi = treat_edges[i], treat_edges[i + 1]
                c_lo, c_hi = ctrl_edges[i], ctrl_edges[i + 1]
                yt = y_arr[treat_order[t_lo:t_hi]]
                yc = y_arr[ctrl_order[c_lo:c_hi]]
                nt, nc = len(yt), len(yc)
                rt = float(yt.mean()) if nt > 0 else 0.0
                rc = float(yc.mean()) if nc > 0 else 0.0
                st = float(yt.std()) if nt > 1 else 0.0
                sc = float(yc.std()) if nc > 1 else 0.0
                rr_treat.append(rt); rr_ctrl.append(rc)
                std_treat.append(st); std_ctrl.append(sc)
                n_treat_bins.append(nt)
        else:  # strategy == "overall"
            order = np.argsort(u_arr)[::-1]
            y_s = y_arr[order]
            t_s = t_arr[order]
            n = len(y_s)
            bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
            rr_treat, rr_ctrl, std_treat, std_ctrl = [], [], [], []
            n_treat_bins = []
            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                y_bin, t_bin = y_s[lo:hi], t_s[lo:hi]
                treat_mask = t_bin == 1
                ctrl_mask = t_bin == 0
                nt = treat_mask.sum()
                nc = ctrl_mask.sum()
                rt = float(y_bin[treat_mask].mean()) if nt > 0 else 0.0
                rc = float(y_bin[ctrl_mask].mean()) if nc > 0 else 0.0
                st = float(y_bin[treat_mask].std()) if nt > 1 else 0.0
                sc = float(y_bin[ctrl_mask].std()) if nc > 1 else 0.0
                rr_treat.append(rt); rr_ctrl.append(rc)
                std_treat.append(st); std_ctrl.append(sc)
                n_treat_bins.append(nt)

        bin_uplifts = [rt - rc for rt, rc in zip(rr_treat, rr_ctrl)]
        std_uplifts = [np.sqrt(st**2 + sc**2) for st, sc in zip(std_treat, std_ctrl)]

        # Weighted average uplift
        total_treat = sum(n_treat_bins)
        weighted_avg = (sum(u * nt for u, nt in zip(bin_uplifts, n_treat_bins)) / total_treat
                        if total_treat > 0 else 0.0)

        # Perzentil-Werte (Mittelwerte der Bin-Ränder, wie sklift)
        pct_values = np.array([(i + 0.5) * 100 / n_bins for i in range(n_bins)])

        if string_percentiles:
            pct_labels = [f"0-{100 / n_bins:.0f}"] + \
                         [f"{i * 100 / n_bins:.0f}-{(i + 1) * 100 / n_bins:.0f}" for i in range(1, n_bins)]
        else:
            pct_labels = [f"{v:.0f}" for v in pct_values]

        # ── Plotting ──
        if kind == "line":
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(pct_values, rr_treat, linewidth=2,
                    color=RUBIN_COLORS["ruby_dark"], label="treatment\nresponse rate", marker="o", markersize=4)
            ax.plot(pct_values, rr_ctrl, linewidth=2,
                    color=COLOR_REFERENCE, label="control\nresponse rate", marker="o", markersize=4)
            ax.plot(pct_values, bin_uplifts, linewidth=2,
                    color=COLOR_MODEL, label="uplift", marker="s", markersize=4)
            ax.fill_between(pct_values, rr_treat, rr_ctrl, alpha=0.1, color=COLOR_MODEL)
            if min(bin_uplifts) < 0:
                ax.axhline(y=0, color="black", linewidth=1)
            ax.set_xticks(pct_values)
            ax.set_xticklabels(pct_labels, rotation=45)
            ax.legend(loc="upper right")
            ax.set_title(f"Uplift by percentile\nweighted average uplift = {weighted_avg:.4f}")
            ax.set_xlabel("Percentile")
            ax.set_ylabel("Uplift = treatment response rate - control response rate")
            fig.tight_layout()
            return fig

        else:  # kind == "bar"
            x = np.arange(n_bins)
            delta = pct_values[0] if len(pct_values) > 0 else 5
            w = delta / 3

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
            fig.text(0.04, 0.5, "Uplift = treatment response rate - control response rate",
                     va="center", ha="center", rotation="vertical", fontsize=9,
                     color=RUBIN_COLORS.get("slate", "#57606A"))

            # axes[0]: Uplift bars
            axes[0].bar(pct_values, bin_uplifts, width=delta / 1.5,
                        color=COLOR_MODEL, edgecolor="black", linewidth=0.5, alpha=0.7,
                        label="uplift")
            axes[0].axhline(0, color="black", linewidth=1)
            axes[0].legend(loc="upper right")
            axes[0].tick_params(axis="x", bottom=False)
            axes[0].set_title(f"Uplift by percentile\nweighted average uplift = {weighted_avg:.4f}")

            # axes[1]: Response rates
            axes[1].bar(pct_values - w / 2, rr_treat, width=w,
                        color=RUBIN_COLORS["ruby_dark"], edgecolor="black", linewidth=0.5,
                        alpha=0.7, label="treatment\nresponse rate")
            axes[1].bar(pct_values + w / 2, rr_ctrl, width=w,
                        color=COLOR_REFERENCE, edgecolor="black", linewidth=0.5,
                        alpha=0.7, label="control\nresponse rate")
            axes[1].axhline(0, color="black", linewidth=1)
            axes[1].set_xticks(pct_values)
            axes[1].set_xticklabels(pct_labels, rotation=45)
            axes[1].set_xlabel("Percentile")
            axes[1].legend(loc="upper right")
            axes[1].set_title("Response rate by percentile")

            fig.tight_layout()
            fig.subplots_adjust(left=0.12)  # Platz für rotiertes Y-Label
            return fig
    except Exception as e:
        _logger.warning("Native Uplift-by-Percentile fehlgeschlagen: %s", e)
        return None


def _native_treatment_balance(
    uplift: np.ndarray, treatment: np.ndarray,
    random: bool = True, winsize: float = 0.1,
) -> Optional[plt.Figure]:
    """Treatment-Balance-Kurve — feature-complete Ersatz für sklift.

    Repliziert ``sklift.viz.plot_treatment_balance_curve`` vollständig inkl.
    aller Parameter (random, winsize).

    Args:
        random: Zeichne Random-Baseline (durchschnittliche Treatment-Rate). Default True.
        winsize: Fenstergröße als Anteil (0-1, Extrema ausgeschlossen). Default 0.1."""
    try:
        from rubin.utils.plot_theme import RUBIN_COLORS, COLOR_MODEL, COLOR_BASELINE
        order = np.argsort(uplift)[::-1]
        t_s = treatment[order].astype(float)
        n = len(t_s)
        window = max(1, int(winsize * n))

        cumsum = np.cumsum(np.insert(t_s, 0, 0))
        window_sums = cumsum[window:] - cumsum[:-window]
        treatment_rates = window_sums / window

        x_axis = np.arange(len(treatment_rates)) / n
        overall_rate = float(treatment.mean())

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(x_axis, treatment_rates, linewidth=2, color=COLOR_MODEL, label="Model")

        if random:
            y_random = overall_rate * np.ones_like(x_axis)
            ax.plot(x_axis, y_random, color=COLOR_BASELINE, linewidth=1.0,
                    linestyle="--", label="Random")
            ax.fill_between(x_axis, treatment_rates, y_random, alpha=0.12, color=COLOR_MODEL)

        ax.legend()
        ax.set_title("Treatment balance curve")
        ax.set_xlabel("Percentage targeted")
        ax.set_ylabel("Balance: treatment / (treatment + control)")
        fig.tight_layout()
        return fig
    except Exception as e:
        _logger.warning("Native Treatment-Balance fehlgeschlagen: %s", e)
        return None


def _native_qini_curve(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray,
    random: bool = True, perfect: bool = False, negative_effect: bool = True,
    name: str = None,
) -> Optional[plt.Figure]:
    """Qini-Kurve — feature-complete Ersatz für sklift.

    Repliziert ``sklift.viz.plot_qini_curve`` vollständig inkl. aller
    Parameter (random, perfect, negative_effect, name).

    Args:
        random: Zeichne Random-Baseline. Default True.
        perfect: Zeichne perfekte Qini-Kurve (theoretisches Optimum). Default False.
        negative_effect: Enthält die perfekte Kurve negative Effekte? Default True.
        name: Modellname für die Legende. Default None."""
    try:
        from rubin.utils.plot_theme import COLOR_MODEL, COLOR_MODEL_FILL, COLOR_BASELINE
        from rubin.evaluation.uplift_metrics import uplift_curve as _uc, qini_coefficient as _qc

        curve = _uc(y=y_true, t=treatment, score=uplift)
        qini = _qc(curve)

        # Qini-Formel: Q(k) = Y_t(k) − Y_c(k) · N_t(k) / N_c(k)
        n_c_safe = np.maximum(curve.n_control, 1)
        y_qini = curve.y_treat - curve.y_control * curve.n_treat / n_c_safe
        n = len(np.asarray(y_true).ravel())
        x = np.r_[0, curve.fraction * n]  # Absolute Anzahl targeted (wie sklift)
        y_qini = np.r_[0, y_qini]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Model curve
        label = f"{name} (AUC = {qini:.4f})" if name else "Model"
        ax.plot(x, y_qini, linewidth=2, color=COLOR_MODEL, label=label)

        # Random baseline
        if random:
            y_random = x * y_qini[-1] / n if n > 0 else x * 0
            ax.plot(x, y_random, label="Random", color=COLOR_BASELINE, linewidth=1.2, linestyle="--")
            ax.fill_between(x, y_qini, y_random, alpha=0.12, color=COLOR_MODEL)

        # Perfect curve
        if perfect:
            y_arr = np.asarray(y_true).ravel()
            t_arr = np.asarray(treatment).ravel()
            n_t = (t_arr == 1).sum()
            n_c = max((t_arr == 0).sum(), 1)
            if negative_effect:
                # Perfekte Sortierung: true uplift = Y·T − Y·(1−T)
                true_uplift = y_arr * t_arr - y_arr * (1 - t_arr)
                perf_curve = _uc(y=y_arr, t=t_arr, score=true_uplift)
                n_c_perf = np.maximum(perf_curve.n_control, 1)
                y_perf = perf_curve.y_treat - perf_curve.y_control * perf_curve.n_treat / n_c_perf
                x_perf = np.r_[0, perf_curve.fraction * n]
                y_perf = np.r_[0, y_perf]
            else:
                ratio = y_arr[t_arr == 1].sum() - y_arr[t_arr == 0].sum() * n_t / n_c
                x_perf = np.array([0, n_t, n])
                y_perf = np.array([0, ratio, ratio])
            ax.plot(x_perf, y_perf, label="Perfect", color="#dc2626", linewidth=1.2, linestyle=":")

        ax.legend(loc="lower right")
        ax.set_title(f"Qini curve\nqini_auc_score={qini:.4f}")
        ax.set_xlabel("Number targeted")
        ax.set_ylabel("Number of incremental outcome")
        fig.tight_layout()
        return fig
    except Exception as e:
        _logger.warning("Native Qini-Curve fehlgeschlagen: %s", e)
        return None


def generate_sklift_plots(
    cate_preds_val: np.ndarray,
    T_val: np.ndarray,
    Y_val: np.ndarray,
) -> tuple:
    """Erzeugt Uplift-Plots: Qini-Kurve, Uplift-by-Percentile, Treatment-Balance.

    Alle Plots sind native rubin-Implementierungen mit rubin-Farbpalette.
    Ersetzt die sklearn-uplift-Abhängigkeit (sklift 0.5.1 hat bekannte
    numpy-Kompatibilitätsprobleme, GitHub Issue #213).

    Gibt (qini, percentile, treatment_balance) als Matplotlib-Figures zurück.
    Fehlende Plots sind None."""

    uplift = np.asarray(cate_preds_val).ravel()
    y_val_arr = np.asarray(Y_val).ravel()
    t_val_arr = np.asarray(T_val).ravel()

    sk_qini = _native_qini_curve(y_val_arr, uplift, t_val_arr)
    sk_pct = _native_uplift_by_percentile(y_val_arr, uplift, t_val_arr)
    sk_tb = _native_treatment_balance(uplift, t_val_arr)

    return sk_qini, sk_pct, sk_tb


def evaluate_cate_with_plots(
    *,
    model_regression: Any = None,
    model_propensity: Any = None,
    X_val: pd.DataFrame,
    T_val: np.ndarray,
    Y_val: np.ndarray,
    cate_preds_val: np.ndarray,
    X_train: Optional[pd.DataFrame] = None,
    T_train: Optional[np.ndarray] = None,
    Y_train: Optional[np.ndarray] = None,
    cate_preds_train: Optional[np.ndarray] = None,
    n_groups: int = 10,
    fitted_tester: Optional[CustomDRTester] = None,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> DrTesterPlotBundle:
    """Hauptfunktion: DRTester-Auswertung + sklift-Plots.

    Wenn fitted_tester übergeben wird, werden die vorberechneten Nuisance-Ergebnisse
    (DR-Outcomes) wiederverwendet — das spart das teure Nuisance-CV-Fitting.
    Nur die CATE-Predictions werden ausgetauscht."""

    if fitted_tester is not None:
        # Nuisance bereits berechnet → frischen Tester mit gleichen DR-Outcomes erstellen.
        # KEIN copy.copy (EconML-interner State ist nicht copy-sicher).
        # Stattdessen: neuer Tester, Nuisance-State manuell injizieren.
        tester = CustomDRTester(
            model_regression=getattr(fitted_tester, '_model_regression_ref', None),
            model_propensity=getattr(fitted_tester, '_model_propensity_ref', None),
            cate=None,
            cate_preds_val=cate_preds_val,
            cate_preds_train=cate_preds_train,
        )
        # DR-Outcomes + Nuisance-State aus dem Pre-Fit übernehmen (das teure Ergebnis)
        tester.dr_val_ = CustomDRTester._sanitize_dr(fitted_tester.dr_val_)
        tester.ate_val = tester.dr_val_.mean(axis=0)
        tester.Dval = fitted_tester.Dval
        tester.treatments = fitted_tester.treatments
        tester.n_treat = fitted_tester.n_treat
        tester.fit_on_train = fitted_tester.fit_on_train
        if hasattr(fitted_tester, 'dr_train_'):
            tester.dr_train_ = fitted_tester.dr_train_
        if hasattr(fitted_tester, 'cate_preds_train_') and fitted_tester.cate_preds_train_ is not None and cate_preds_train is not None:
            tester.cate_preds_train_ = np.asarray(cate_preds_train).reshape(-1, 1)
        _logger.debug(
            "DRTester Nuisance wiederverwendet: dr_val=%s, cate_preds_val=%s (min=%.4g, max=%.4g)",
            tester.dr_val_.shape, tester.cate_preds_val_.shape,
            float(np.nanmin(cate_preds_val)), float(np.nanmax(cate_preds_val)),
        )
    else:
        # Fallback: Nuisance komplett neu fitten (wenn kein Pre-Fit vorhanden)
        if model_regression is None or model_propensity is None:
            raise ValueError("Entweder fitted_tester oder model_regression + model_propensity muss angegeben werden.")
        tester = CustomDRTester(
            model_regression=model_regression,
            model_propensity=model_propensity,
            cate=None,
            cate_preds_val=cate_preds_val,
            cate_preds_train=cate_preds_train,
        )

        if X_train is not None and T_train is not None and Y_train is not None:
            tester.fit_nuisance(
                Xval=X_val.values,
                Dval=np.asarray(T_val).ravel(),
                yval=np.asarray(Y_val).ravel(),
                Xtrain=X_train.values,
                Dtrain=np.asarray(T_train).ravel(),
                ytrain=np.asarray(Y_train).ravel(),
            )
        else:
            tester.fit_nuisance(
                Xval=X_val.values,
                Dval=np.asarray(T_val).ravel(),
                yval=np.asarray(Y_val).ravel(),
            )

    # ── DRTester-Plots (Calibration, Qini, TOC) ──
    summary, cal_plot, qini_plot, toc_plot, policy_values = None, None, None, None, None
    try:
        res = tester.evaluate_all(X_val.values, X_train.values if X_train is not None else None, n_groups=n_groups, n_bootstrap=n_bootstrap, seed=seed)
        summary = res.summary()
        cal_plot = res.plot_cal(1).get_figure()
        recolor_figure(cal_plot)
        qini_plot = res.plot_qini(1).get_figure()
        recolor_figure(qini_plot)
        toc_plot = res.plot_toc(1).get_figure()
        recolor_figure(toc_plot)
        policy_values = res.get_policy_values(1)
    except Exception as e:
        _logger.warning("DRTester evaluate_all fehlgeschlagen: %s", e, exc_info=True)
        if summary is None:
            summary = pd.DataFrame()
        if policy_values is None:
            policy_values = pd.DataFrame()

    # ── Uplift-Plots: Qini, Uplift-by-Percentile, Treatment-Balance ──
    # Native rubin-Implementierungen (sklift 0.5.1 hat numpy-Kompatibilitäts-Bug)
    uplift = np.asarray(cate_preds_val).ravel()
    y_val_arr = np.asarray(Y_val).ravel()
    t_val_arr = np.asarray(T_val).ravel()

    sk_qini = _native_qini_curve(y_val_arr, uplift, t_val_arr)
    sk_pct = _native_uplift_by_percentile(y_val_arr, uplift, t_val_arr)
    sk_tb = _native_treatment_balance(uplift, t_val_arr)

    return DrTesterPlotBundle(
        summary=summary,
        cal_plot=cal_plot,
        qini_plot=qini_plot,
        toc_plot=toc_plot,
        policy_values=policy_values,
        sklift_qini=sk_qini,
        sklift_percentile=sk_pct,
        treatment_balance=sk_tb,
    )


def compute_qini_curve(outcomes: np.ndarray, score: np.ndarray, treatment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Berechnet Qini-Kurvenpunkte nativ.

    Q(k) = Y_t(k) − Y_c(k) · N_t(k) / N_c(k)
    Gibt (y_model, x, y_random) als interpolierte Arrays zurück."""
    from rubin.evaluation.uplift_metrics import uplift_curve as _uc

    curve = _uc(y=outcomes, t=treatment, score=score)
    n = len(outcomes)

    # Qini-Formel
    n_c_safe = np.maximum(curve.n_control, 1)
    y_qini = curve.y_treat - curve.y_control * curve.n_treat / n_c_safe

    # Auf ganzzahliges x-Raster interpolieren
    x = np.arange(n)
    x_frac = curve.fraction * n  # fraction → absolute Indizes
    y_score_i = np.interp(x, x_frac, y_qini)
    qini_total = y_qini[-1] if len(y_qini) > 0 else 0
    y_rand_i = x * qini_total / max(n, 1)
    return y_score_i, x, y_rand_i


def plot_custom_qini_curve(
    *,
    data: pd.DataFrame,
    causal_score_label: str,
    affinity_score_label: Optional[str] = None,
    ax: Optional[Any] = None,
    relative_axes: bool = True,
):
    """Custom-Qini-Plot zur direkten Gegenüberstellung zweier Scores.

    Stellt den kausalen Score gegen einen optionalen Referenzscore (z. B.
    historischer Affinity-Score) auf derselben Qini-Kurve dar. Ermöglicht
    einen schnellen visuellen Vergleich der Sortierqualität."""

    y_causal, x, y_random = compute_qini_curve(data["Y"].to_numpy(), data[causal_score_label].to_numpy(), data["T"].to_numpy())
    y_affinity = None
    if affinity_score_label is not None:
        y_affinity, _, _ = compute_qini_curve(data["Y"].to_numpy(), data[affinity_score_label].to_numpy(), data["T"].to_numpy())

    if relative_axes:
        x = x / np.max(x)
        inc_total = np.max(y_random)
        y_causal = y_causal / inc_total
        y_random = y_random / inc_total
        if y_affinity is not None:
            y_affinity = y_affinity / inc_total

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, y_random, color=COLOR_BASELINE, label="Random", linewidth=1.5, linestyle=":")

    ax.plot(x, y_causal, color=COLOR_MODEL, label=causal_score_label)
    ax.fill_between(x, y_causal, y_random, color=COLOR_MODEL, alpha=0.08)

    if y_affinity is not None and affinity_score_label is not None:
        ax.plot(x, y_affinity, color=COLOR_REFERENCE, label=affinity_score_label)
        ax.fill_between(x, y_affinity, y_random, color=COLOR_REFERENCE, alpha=0.08)

    if relative_axes:
        ax.set(xticks=np.linspace(0, 1, 5), yticks=np.linspace(0, 1, 5))
        ax.set_xlabel("Anteil Experimentalmenge")
        ax.set_ylabel("Anteil inkrementelles Ergebnis")
    else:
        ax.set_xlabel("Experimentalmenge")
        ax.set_ylabel("Inkrementelles Ergebnis")

    ax.set_title(f"Qini-Vergleich: {causal_score_label}")
    ax.legend(loc="upper left")

    return ax

def policy_value_comparison_plots(
    policy_values_dict: dict[str, pd.DataFrame],
    comparison_model_name: str,
) -> dict[str, plt.Figure]:
    """Erzeugt Vergleichsplots der Policy Values gegen ein Referenzmodell.

    Neben der Qini-Kurve ist der erwartete inkrementelle Nutzen (Policy Value)
    für feste Treat-Anteile (z. B. 10%, 20%, …) eine zentrale Entscheidungsgrundlage.
    Diese Funktion erstellt für jedes Modell (außer dem Referenzmodell) einen Plot
    mit drei Kurven:

    1) Policy Values des Modells inkl. Konfidenzintervall
    2) Policy Values des Referenzmodells inkl. Konfidenzintervall
    3) Differenzkurve (Modell − Referenz)

    Erwartetes DataFrame-Format je Modell:
    - treated_percentage
    - policy_value
    - lower_bound
    - upper_bound"""

    if comparison_model_name not in policy_values_dict:
        raise KeyError(
            f"Referenzmodell '{comparison_model_name}' nicht in policy_values_dict vorhanden."
        )

    plots: dict[str, plt.Figure] = {}
    ref = policy_values_dict[comparison_model_name]

    # Skip if reference has no data
    if ref is None or len(ref) == 0 or "policy_value" not in ref.columns:
        return plots

    for model_name, df in policy_values_dict.items():
        if model_name == comparison_model_name:
            continue
        if df is None or len(df) == 0 or "policy_value" not in df.columns:
            continue

        treated = df["treated_percentage"]
        model_values = df["policy_value"]
        model_lower = df["lower_bound"]
        model_upper = df["upper_bound"]

        ref_values = ref["policy_value"]
        ref_lower = ref["lower_bound"]
        ref_upper = ref["upper_bound"]

        difference = model_values.to_numpy() - ref_values.to_numpy()

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(treated, model_values, label=f"{model_name} Policy Value", marker="o", color=COLOR_MODEL)
        ax.fill_between(treated, model_lower, model_upper, alpha=0.12, color=COLOR_MODEL, label=f"{model_name} Konfidenzintervall")

        ax.plot(treated, ref_values, label=f"{comparison_model_name} Policy Value", linestyle="--", marker="o", color=COLOR_REFERENCE)
        ax.fill_between(treated, ref_lower, ref_upper, alpha=0.12, color=COLOR_REFERENCE, label=f"{comparison_model_name} Konfidenzintervall")

        ax.plot(treated, difference, label="Differenz (Modell - Referenz)", linestyle="-.", marker="x", color=COLOR_DIFFERENCE)
        ax.axhline(0, color=COLOR_BASELINE, linewidth=0.8, linestyle="--", label="Keine Differenz")

        ax.set_title(f"Policy Value Vergleich: {model_name} vs. {comparison_model_name}")
        ax.set_xlabel("Treated Percentage")
        ax.set_ylabel("Policy Value")
        ax.legend(loc="upper left")

        plots[model_name] = fig

    return plots



def plot_score_redistribution(
    cate_scores: np.ndarray,
    hist_scores: np.ndarray,
    n_bins: int = 10,
    model_name: str = "",
) -> "matplotlib.figure.Figure":
    """Score-Redistribution-Plot: Gestapeltes Balkendiagramm mit Colorbar.

    Zeigt pro Dezil des neuen CATE-Scores, welcher Anteil der Samples
    aus welchem Dezil des historischen Scores stammt.

    Parameters
    ----------
    cate_scores : (n,) array
        Neue CATE-Vorhersagen (höher = stärkerer Treatment-Effekt).
    hist_scores : (n,) array
        Historische Scores (höher = besser gemäß historical_score.higher_is_better).
    n_bins : int
        Anzahl der Perzentil-Bins (Default 10 = Dezile).
    model_name : str
        Modellname für den Titel.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.colorbar import ColorbarBase
    from scipy.stats import spearmanr
    try:
        from rubin.utils.plot_theme import RUBIN_COLORS, apply_rubin_theme
        apply_rubin_theme()
    except ImportError:
        RUBIN_COLORS = {"ruby_dark": "#6B0D15", "ruby": "#9B111E",
                        "ruby_pale": "#FDF2F3", "slate": "#57606A",
                        "text": "#24292F", "grid": "#EDE6E7"}

    cate_scores = np.asarray(cate_scores, dtype=float).ravel()
    hist_scores = np.asarray(hist_scores, dtype=float).ravel()
    assert len(cate_scores) == len(hist_scores), (
        "CATE und historische Scores müssen gleich lang sein."
    )

    # ── Rangkorrelation ──
    rho, _ = spearmanr(cate_scores, hist_scores)

    # ── Dezil-Zuordnung (0=niedrigste 10%, n_bins-1=höchste 10%) ──
    def _assign_bins(scores, n):
        edges = np.percentile(scores, np.linspace(0, 100, n + 1))
        edges[-1] += 1e-10
        bins = np.digitize(scores, edges[1:], right=False)
        return np.clip(bins, 0, n - 1)

    cate_bins = _assign_bins(cate_scores, n_bins)
    hist_bins = _assign_bins(hist_scores, n_bins)

    # ── Transitionsmatrix: (n_bins × n_bins) — row=hist, col=cate ──
    matrix = np.zeros((n_bins, n_bins), dtype=float)
    for cb, hb in zip(cate_bins, hist_bins):
        matrix[hb, cb] += 1.0

    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    matrix_pct = matrix / col_sums * 100.0

    # ── Spalten umkehren: D10 (höchstes CATE) links, D1 rechts ──
    # Konsistent mit Qini-Kurve und Uplift-by-Percentile (Top-Kunden links)
    matrix_pct = matrix_pct[:, ::-1]

    # ── Farbverlauf: D1 (niedrigstes hist.) = hell, Dn (höchstes) = dunkel ──
    cmap = LinearSegmentedColormap.from_list(
        "rubin_redistribution",
        [RUBIN_COLORS["ruby_pale"], RUBIN_COLORS["ruby"], RUBIN_COLORS["ruby_dark"]],
        N=256,
    )
    colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]

    # ── Figure ──
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[35, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    x = np.arange(1, n_bins + 1)
    bottom = np.zeros(n_bins)

    for hist_d in range(n_bins):
        heights = matrix_pct[hist_d, :]
        ax.bar(x, heights, bottom=bottom, width=0.82,
               color=colors[hist_d], edgecolor="white", linewidth=0.5)
        bottom += heights

    ax.set_xlabel("Dezil des Kausal-Scores (CATE)")
    ax.set_ylabel("Anteil der Samples (%)")

    # Kompakte Kennzahl statt Titel (wie native Plots)
    ax.set_title(f"Score-Redistribution\nSpearman \u03c1 = {rho:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"D{n_bins + 1 - i}" for i in x])
    ax.set_ylim(0, 100)
    ax.set_xlim(0.35, n_bins + 0.65)

    # ── Colorbar ──
    norm = Normalize(vmin=1, vmax=n_bins)
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("Dezil Hist. Score", fontsize=10, labelpad=8)
    tick_labels = [f"D{i}" for i in range(1, n_bins + 1)]
    tick_labels[0] = "D1\n(niedrig)"
    tick_labels[-1] = f"D{n_bins}\n(hoch)"
    cb.set_ticks(list(range(1, n_bins + 1)))
    cb.set_ticklabels(tick_labels, fontsize=8)
    cb.ax.tick_params(size=0)

    fig.subplots_adjust(bottom=0.10, top=0.90, left=0.10, right=0.91)
    return fig
