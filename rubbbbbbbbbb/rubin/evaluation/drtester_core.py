from __future__ import annotations

"""DRTester-Kernklassen: CustomDRTester, CustomEvaluationResults, DrTesterPlotBundle.

Kapselt die EconML DRTester-Validierung für vorberechnete CATE-Werte
(Cross-Predictions). Enthält Nuisance-Fitting und Filter-Hilfsfunktionen.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
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
        """Erzeugt Nuisance-Preds und DR-Outcomes über Cross-Fitting auf der Val-Menge.

        Überschreibt die EconML-Methode. Die Train-Argumente werden aus
        Kompatibilitätsgründen weiterhin akzeptiert, aber NICHT für eine eigene
        Train-Nuisance-Berechnung verwendet (siehe Kommentar unten)."""

        self.Dval = Dval
        self.treatments = np.sort(np.unique(Dval))
        self.n_treat = len(self.treatments) - 1

        self.fit_on_train = (Xtrain is not None) and (Dtrain is not None) and (ytrain is not None)

        # dr_train_ wird von KEINER evaluate_*-Methode gelesen: EconMLs
        # evaluate_blp/cal/uplift nutzen cate_preds_train_ (separat aus den
        # Train-Cross-Predictions gesetzt), nicht dr_train_. Die frühere
        # Train-Nuisance-Berechnung war daher toter Code — und crashte im
        # External-/TMES-Modus, weil der vorberechnete T×Y-Splitter mit
        # Val-Längen-Strata gebaut ist und nicht auf die (anders langen)
        # Train-Daten passt (StratifiedKFold: inconsistent number of samples).
        # Wir berechnen daher nur die tatsächlich genutzten Val-DR-Outcomes.
        self.dr_train_ = None
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

        # Fallback: EconML's evaluate_cal() bildet die Quantil-Cuts aus
        # cate_preds_train_ (nicht aus dr_train_!) und nimmt die Gate-Werte aus
        # dr_val_; dr_train_ wird dort nur über einen hasattr-Guard auf Existenz
        # geprüft — sein Wert geht NICHT in die Berechnung ein (deshalb ist das
        # Nullsetzen oben unkritisch). Fehlen cate_preds_train_ (z.B. Ensemble-
        # Modell) bzw. existiert dr_train_ nicht (hier bewusst None gesetzt),
        # werden die Val-Daten als Fallback verwendet, damit der Guard passt.
        # Weniger rigoros bei den Cuts, aber die Calibration-Plots bleiben informativ.
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
    uplift_qini: Optional[plt.Figure]
    uplift_percentile: Optional[plt.Figure]
    treatment_balance: Optional[plt.Figure]

    _PLOT_FIELDS = ["cal_plot", "qini_plot", "toc_plot",
                    "uplift_qini", "uplift_percentile", "treatment_balance"]

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


