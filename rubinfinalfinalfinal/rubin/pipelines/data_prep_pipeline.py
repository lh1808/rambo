from __future__ import annotations

"""DataPrepPipeline.
Dieses Modul implementiert die Datenaufbereitung als eigenständige, objektorientierte
Pipeline. Alle Parameter werden über die zentrale Projekt-Konfiguration (`config.yml`)
gesteuert (Sektion `data_prep`).
Ziel der DataPrepPipeline ist es, aus Rohdaten reproduzierbar die drei Kernobjekte
für kausale Verfahren zu erzeugen:
- **X**: Feature-Matrix
- **T**: Treatment (diskret: 0/1 bei Binary Treatment; 0, 1, …, K-1 bei Multi-Treatment)
- **Y**: Outcome (0/1)
Zusätzlich wird ein Preprocessing-Artefakt erzeugt, das in der Analyse- und später
in der Produktionspipeline wiederverwendet werden kann."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import json
import logging
import pickle

import numpy as np
import pandas as pd
from rubin.preprocessing import FittedPreprocessor, fit_preprocessor
from rubin.settings import AnalysisConfig, DataPrepConfig, load_config
from rubin.utils.data_utils import reduce_mem_usage, is_object_like_dtype


@dataclass
class DataPrepOutputs:
    """Outputs der Datenaufbereitung."""

    X: pd.DataFrame
    T: np.ndarray
    Y: np.ndarray
    S: Optional[np.ndarray]
    preprocessor: FittedPreprocessor


class DataPrepPipeline:
    """Objektorientierte Datenaufbereitung."""

    _logger = logging.getLogger("rubin.dataprep")

    def __init__(self, cfg: AnalysisConfig, dataprep_cfg: DataPrepConfig) -> None:
        self.cfg = cfg
        self.dp = dataprep_cfg


    def _target_cols(self) -> list:
        """Normalisiert dp.target zu einer Liste von Spaltennamen (uppercase)."""
        t = self.dp.target
        if isinstance(t, str):
            return [t.upper()]
        return [str(c).upper() for c in t]

        # Sanity-Checks früh, damit Fehler nicht erst nach teuren I/O-Schritten auffallen.
        if not self.dp.data_path:
            raise ValueError("data_prep.data_path ist leer. Bitte mindestens eine Eingabedatei konfigurieren.")
        if not self.dp.target or not self.dp.treatment:
            raise ValueError("data_prep.target und data_prep.treatment müssen gesetzt sein.")

    @classmethod
    def from_config_path(cls, config_path: str) -> "DataPrepPipeline":
        """Erstellt die Pipeline aus der zentralen Projekt-Konfiguration."""

        cfg = load_config(config_path)
        if cfg.data_prep is None:
            raise ValueError(
                "In der Konfiguration fehlt die Sektion 'data_prep'. "
                "Für DataPrep-Läufe muss diese Sektion vorhanden sein."
            )
        return cls(cfg, cfg.data_prep)

    def _read_files(self, file_paths: Optional[List[str]] = None, merge_only: bool = False) -> pd.DataFrame:
        dp = self.dp
        paths = file_paths if file_paths is not None else dp.data_path
        df_list: List[pd.DataFrame] = []

        for file_path in paths:
            p = str(file_path)
            p_lower = p.lower()
            if p_lower.endswith(".csv"):
                obj = pd.read_csv(
                    file_path,
                    delimiter=dp.delimiter,
                    low_memory=False,
                    chunksize=dp.chunksize,
                )
            elif p_lower.endswith((".parquet", ".pq")):
                # Parquet liefert immer direkt ein DataFrame; eine Chunk-Verarbeitung ist hier
                # bewusst nicht vorgesehen.
                try:
                    obj = pd.read_parquet(file_path)
                except ImportError as e:
                    raise ImportError(
                        "Für Parquet-Dateien wird 'pyarrow' oder 'fastparquet' benötigt. "
                        "Bitte die Abhängigkeiten aus requirements.txt installieren."
                    ) from e
            elif p_lower.endswith(".sas7bdat"):
                obj = pd.read_sas(file_path, chunksize=dp.chunksize, encoding=dp.sas_encoding)
            else:
                raise ValueError(f"Nicht unterstützter Dateityp in data_prep.data_path: {file_path}")

            # pandas liefert bei gesetztem chunksize einen Iterator, sonst direkt ein DataFrame.
            if isinstance(obj, pd.DataFrame):
                iterator = iter([obj])
            else:
                iterator = iter(obj)

            chunk_list: List[pd.DataFrame] = []
            try:
                first = next(iterator)
            except StopIteration as e:
                raise ValueError(f"Die Eingabedatei '{file_path}' enthält keine Daten.") from e

            # Spaltennamen vereinheitlichen, damit Feature-Dictionary, CSV und SAS identisch behandelt werden.
            first.columns = [str(c).upper() for c in first.columns]
            target_cols = self._target_cols()
            treat_col = str(dp.treatment).upper()

            required_columns = target_cols + [treat_col]
            missing_required = [c for c in required_columns if c not in first.columns]
            if missing_required:
                raise ValueError(
                    f"Pflichtspalten fehlen in '{file_path}': {missing_required}. "
                    f"Verfügbare Spalten (erste Datei/erster Chunk): {list(first.columns)[:20]}"
                )

            # Replacement-Mappings (z. B. 'J'/'N' -> 0/1)
            if dp.treatment_replacement:
                first[treat_col] = first[treat_col].replace(dp.treatment_replacement)
            if dp.target_replacement:
                for _tc in target_cols:
                    first[_tc] = first[_tc].replace(dp.target_replacement)

            dtypes = first.dtypes.to_dict()
            chunk_list.append(first)

            for chunk in iterator:
                chunk.columns = [str(c).upper() for c in chunk.columns]
                if dp.treatment_replacement:
                    chunk[treat_col] = chunk[treat_col].replace(dp.treatment_replacement)
                if dp.target_replacement:
                    for _tc in target_cols:
                        chunk[_tc] = chunk[_tc].replace(dp.target_replacement)
                for col, dt in dtypes.items():
                    if col in chunk.columns:
                        try:
                            chunk[col] = chunk[col].astype(dt)
                        except Exception:
                            # Typ-Downcasts sind best-effort; im Zweifel bleibt der Chunk wie er ist.
                            pass
                chunk_list.append(chunk)

            df_list.append(pd.concat(chunk_list, ignore_index=True))

        if len(df_list) == 1:
            return df_list[0]

        # Datei-Herkunft markieren (für Treatment-Balance-Prüfung)
        for i, df_part in enumerate(df_list):
            df_part["__file_source__"] = i

        # Eval-Daten werden immer zusammengeführt (kein treatment_only).
        opt = "merge" if merge_only else dp.multiple_files_option
        if opt == "merge":
            return pd.concat(df_list, ignore_index=True)

        if opt == "treatment_only":
            treat_col = str(dp.treatment).upper()
            treatments = [df[df[treat_col] == 1] for df in df_list]
            ctrl_like = treatments[dp.control_file_index].copy()
            ctrl_like[treat_col] = 0
            del treatments[dp.control_file_index]
            return pd.concat([ctrl_like] + treatments, ignore_index=True)

        raise ValueError(f"Unbekannte data_prep.multiple_files_option={opt}")

    def _check_and_balance_treatments(self, df: pd.DataFrame, treat_col: str, dp, _progress) -> pd.DataFrame:
        """Prüft Treatment-Verteilung pro Datei und gleicht per Downsampling auf eine
        gemeinsame Ziel-Rate aus.

        Wenn die Treatment-Rate zwischen Dateien um mehr als 5 Prozentpunkte abweicht,
        wird gewarnt. Bei balance_treatments=True werden mehrere Ziel-Raten-Kandidaten
        simuliert und diejenige gewählt, die die EFFEKTIVE STICHPROBE maximiert — nicht
        die, die am wenigsten Zeilen verliert.

        Auswahlkriterium: effektive N statt Zeilenverlust
        ─────────────────────────────────────────────────
        effektive N(p) = N_keep(p) · p · (1−p),  mit p = Ziel-Treatment-Rate.

        Begründung (Heterogenitätsanalyse): Die Varianz jedes Effekt-/Uplift-Schätzers
        skaliert mit 1/(N · p · (1−p)); der kleinere Arm limitiert die Präzision. Reines
        „wenigste Zeilen verlieren" maximiert die Gesamtzahl N, ignoriert aber die Balance
        p·(1−p) und kann einen Arm stark schrumpfen lassen (schlechter Overlap, instabile
        CATE/Qini). Effektive N gewichtet den behaltenen Umfang mit der Balance und trifft
        damit den für die Heterogenitätsschätzung tatsächlich nutzbaren Stichprobenumfang.

        Beispiel (gleich große Dateien, Raten 61/62/67/50%):
        - Ziel=50%: 20% Verlust → effektive N ≈ 800 (0.200·N)  ← gewählt (von {min,max})
        - Ziel=67%: 10,5% Verlust → effektive N ≈ 792 (0.198·N)
        Trotz höheren Verlusts gewinnt 50%, weil die bessere Balance überwiegt.
        Mit Gitter (balance_target_grid_step) gewinnt sogar 60%: effektive N ≈ 859 (0.215·N)
        bei ~10,5% Verlust — der eigentliche Sweet-Spot zwischen Balance und Umfang.

        Kandidaten & Schranken (alle optional):
        - Standard-Kandidaten: {min(Raten), max(Raten)} (nur eine Seite wird gedownsamplet).
        - balance_target_grid_step > 0: zusätzliches Gitter Richtung 0.5 (beide Seiten).
        - balance_min_arm_abs / _frac: harte Mindestschranke je Arm (disqualifiziert Kandidaten).
        - balance_max_loss_frac: Deckel auf den Zeilenverlust-Anteil.
        Erfüllt kein Kandidat die Schranken, wird das beste effektive N gewählt und gewarnt.

        Warum überhaupt balancieren?
        ────────────────────────────
        DML/DR handhaben Treatment-Imbalance theoretisch via Propensity-Residualisierung
        (Chernozhukov et al. 2018). Aber bei Multi-File-Daten mit unterschiedlichen Raten
        können CV-Folds systematisch andere Treatment-Verteilungen haben, was Uplift-Metriken
        (Qini, AUUC, Policy Value) verzerrt. Alternativ lösen treatment-/datei-stratifizierte
        CV-Folds dasselbe Problem ohne Datenverlust (separates Thema).
        """
        file_ids = sorted(df["__file_source__"].unique())
        if len(file_ids) <= 1:
            # Einzelne Datei → keine Balance nötig. __file_source__ NICHT entfernen,
            # der Aufrufer braucht sie ggf. für eval_file_index.
            return df

        # Treatment-Rate pro Datei berechnen
        stats = []
        for fid in file_ids:
            mask = df["__file_source__"] == fid
            n = int(mask.sum())
            t_vals = df.loc[mask, treat_col]
            treat_rate = float((t_vals == 1).mean()) if n > 0 else 0.0
            stats.append({"file": fid, "n": n, "treat_rate": treat_rate})
            self._logger.info(
                "Datei %d: %d Zeilen, Treatment-Rate: %.1f%%",
                fid, n, treat_rate * 100,
            )

        rates = [s["treat_rate"] for s in stats]
        max_diff = max(rates) - min(rates)

        if max_diff > 0.05:  # > 5 Prozentpunkte Unterschied
            self._logger.warning(
                "Treatment-Imbalance erkannt! Treatment-Raten pro Datei: %s "
                "(Differenz: %.1f Prozentpunkte). Dies kann zu verzerrten "
                "Uplift-Metriken führen.",
                [f"{s['treat_rate']*100:.1f}%" for s in stats],
                max_diff * 100,
            )
            rate_strs = [f"{s['treat_rate']*100:.1f}%" for s in stats]
            balance_msg = "Downsampling wird angewendet." if dp.balance_treatments else "Aktiviere balance_treatments für Ausgleich."
            print(
                f"[rubin] WARNUNG: Treatment-Imbalance erkannt! "
                f"Raten pro Datei: {rate_strs} "
                f"(Diff: {max_diff*100:.1f}pp). {balance_msg}",
                flush=True,
            )

            if dp.balance_treatments:
                n_before = len(df)
                _seed = self.cfg.constants.random_seed

                # ── Auswahl-Schranken (optional, Default aus) ──
                _min_arm_abs = int(getattr(dp, "balance_min_arm_abs", 0) or 0)
                _min_arm_frac = float(getattr(dp, "balance_min_arm_frac", 0.0) or 0.0)
                # None explizit behandeln statt `or 1.0`: ein legitimes 0.0 (strengster
                # Verlust-Deckel) ist falsy und würde sonst stillschweigend zu 1.0 (Deckel aus).
                _v_max_loss = getattr(dp, "balance_max_loss_frac", 1.0)
                _max_loss_frac = float(_v_max_loss if _v_max_loss is not None else 1.0)
                _grid_step = float(getattr(dp, "balance_target_grid_step", 0.0) or 0.0)

                n_t_orig = int((df[treat_col] == 1).sum())
                n_c_orig = int((df[treat_col] == 0).sum())
                _arm_floor = max(_min_arm_abs, int(_min_arm_frac * min(n_t_orig, n_c_orig)))

                def _simulate(target_r: float):
                    """Behaltene (Treatment, Control, Gesamt) bei gegebener Ziel-Rate.

                    Nutzt exakt dieselben per-Datei-Formeln wie der Downsampling-Loop unten,
                    damit Simulation und Ausführung konsistent sind.
                    """
                    kept_t = kept_c = 0
                    for s in stats:
                        part = df[df["__file_source__"] == s["file"]]
                        n_t = int((part[treat_col] == 1).sum())
                        n_c = int((part[treat_col] == 0).sum())
                        if abs(s["treat_rate"] - target_r) < 0.01:
                            kept_t += n_t
                            kept_c += n_c
                        elif s["treat_rate"] > target_r and n_c > 0:
                            target_n_t = max(int(target_r / (1 - target_r) * n_c), 1)
                            kept_t += min(n_t, target_n_t)
                            kept_c += n_c
                        elif s["treat_rate"] < target_r and n_t > 0:
                            target_n_c = max(int((1 - target_r) / target_r * n_t), 1)
                            kept_t += n_t
                            kept_c += min(n_c, target_n_c)
                        else:
                            kept_t += n_t
                            kept_c += n_c
                    return kept_t, kept_c, kept_t + kept_c

                # ── Kandidaten-Zielraten: {min, max} (+ optionales Gitter Richtung 0.5) ──
                _cands = {min(rates), max(rates)}
                if _grid_step > 0:
                    _lo, _hi = min(rates), max(rates)
                    _g = _lo
                    while _g <= _hi + 1e-9:
                        _cands.add(round(_g, 4))
                        _g += _grid_step
                    if _lo <= 0.5 <= _hi:
                        _cands.add(0.5)
                _candidates = [p for p in sorted(_cands) if 0.0 < p < 1.0]
                # Degenerierte Zielraten (0 oder 1, z. B. reine Control-/Treatment-Dateien)
                # ausschließen: effektive N wäre 0 und das Downsampling dividierte durch 0.
                if not _candidates:
                    _candidates = [0.5]

                # ── Bewerten: effektive Stichprobe N_keep · p·(1−p) statt nur Zeilenverlust ──
                # Die Varianz eines Effekt-/Uplift-Schätzers skaliert mit 1/(N·p·(1−p));
                # der kleinere Arm limitiert die Präzision. „Effektive N" ist der für die
                # Heterogenitätsschätzung tatsächlich nutzbare Stichprobenumfang. p_actual
                # = kept_t/N_keep (tatsächlich erreichte Balance) → eff_n = kept_t·kept_c/N_keep.
                _evals = []
                for _p in _candidates:
                    _kt, _kc, _nk = _simulate(_p)
                    _e = {
                        "p": _p, "kept_t": _kt, "kept_c": _kc, "n_keep": _nk,
                        "eff_n": (_kt * _kc / _nk) if _nk > 0 else 0.0, "min_arm": min(_kt, _kc),
                        "loss_frac": 1.0 - _nk / max(n_before, 1),
                    }
                    _e["feasible"] = (_e["min_arm"] >= _arm_floor) and (_e["loss_frac"] <= _max_loss_frac)
                    _evals.append(_e)

                _feasible = [e for e in _evals if e["feasible"]]
                if _feasible:
                    _best = max(_feasible, key=lambda e: e["eff_n"])
                else:
                    _best = max(_evals, key=lambda e: e["eff_n"])
                    self._logger.warning(
                        "Treatment-Balance: kein Kandidat erfüllt Arm-Floor (%d) bzw. "
                        "Verlust-Deckel (%.0f%%). Wähle bestes effektive N (Ziel=%.1f%%, "
                        "kleinster Arm=%d). DML/DR tragen Rate-Imbalance ohnehin via "
                        "Propensity-Residualisierung — ggf. balance_treatments deaktivieren.",
                        _arm_floor, _max_loss_frac * 100, _best["p"] * 100, _best["min_arm"],
                    )

                target_rate = _best["p"]
                _lost_t = n_t_orig - _best["kept_t"]
                _lost_c = n_c_orig - _best["kept_c"]
                if _lost_t > 0 and _lost_c > 0:
                    _direction = "gemischtes Downsampling"
                elif _lost_c >= _lost_t:
                    _direction = "Control-Downsampling"
                else:
                    _direction = "Treatment-Downsampling"

                # Kandidaten-Vergleich transparent loggen (effektive N entscheidet).
                self._logger.info(
                    "Balance-Auswahl nach effektiver N (N_keep·p·(1−p)). Kandidaten: %s. "
                    "Gewählt: Ziel=%.1f%% | effektive N=%.0f | kleinster Arm=%d | Verlust=%.1f%% (%s).",
                    "; ".join(
                        f"{e['p']*100:.1f}%→effN={e['eff_n']:.0f},Arm={e['min_arm']},"
                        f"Verlust={e['loss_frac']*100:.1f}%{'' if e['feasible'] else '(✗)'}"
                        for e in _evals
                    ),
                    target_rate * 100, _best["eff_n"], _best["min_arm"],
                    _best["loss_frac"] * 100, _direction,
                )

                balanced_parts = []
                for s in stats:
                    part = df[df["__file_source__"] == s["file"]].copy()
                    if abs(s["treat_rate"] - target_rate) < 0.01:
                        balanced_parts.append(part)
                        continue

                    t_mask = part[treat_col] == 1
                    c_mask = part[treat_col] == 0
                    n_treat = int(t_mask.sum())
                    n_control = int(c_mask.sum())

                    if s["treat_rate"] > target_rate and n_control > 0:
                        # Treatment downsamplen
                        target_n_treat = int(target_rate / (1 - target_rate) * n_control)
                        target_n_treat = max(target_n_treat, 1)
                        if target_n_treat < n_treat:
                            treat_sample = part[t_mask].sample(n=target_n_treat, random_state=_seed)
                            part = pd.concat([part[c_mask], treat_sample], ignore_index=True)
                    elif s["treat_rate"] < target_rate and n_treat > 0:
                        # Control downsamplen
                        target_n_control = int((1 - target_rate) / target_rate * n_treat)
                        target_n_control = max(target_n_control, 1)
                        if target_n_control < n_control:
                            control_sample = part[c_mask].sample(n=target_n_control, random_state=_seed)
                            part = pd.concat([part[t_mask], control_sample], ignore_index=True)

                    balanced_parts.append(part)

                df = pd.concat(balanced_parts, ignore_index=True)
                n_after = len(df)
                self._logger.info(
                    "Treatment-Balance Downsampling: %d → %d Zeilen (%.0f%% entfernt). "
                    "Ziel-Treatment-Rate: %.1f%% (%s).",
                    n_before, n_after, (1 - n_after / max(n_before, 1)) * 100,
                    target_rate * 100, _direction,
                )
                if hasattr(dp, 'log_to_mlflow') and dp.log_to_mlflow:
                    try:
                        import mlflow
                        mlflow.log_param("treatment_balance_applied", True)
                        mlflow.log_metric("treatment_balance_rows_before", n_before)
                        mlflow.log_metric("treatment_balance_rows_after", n_after)
                        mlflow.log_param("treatment_balance_target_rate", round(target_rate, 4))
                        mlflow.log_metric("treatment_balance_effective_n", round(_best["eff_n"], 1))
                        mlflow.log_metric("treatment_balance_min_arm", _best["min_arm"])
                    except Exception:
                        pass
        else:
            self._logger.info(
                "Treatment-Balance OK: Raten pro Datei %s (Diff: %.1f pp).",
                [f"{s['treat_rate']*100:.1f}%" for s in stats],
                max_diff * 100,
            )

        # Hinweis: __file_source__ wird NICHT hier entfernt — der Aufrufer
        # braucht sie ggf. noch für eval_file_index (Train Many, Evaluate Some).
        return df

    def run(self) -> DataPrepOutputs:
        dp = self.dp
        out_dir = Path(dp.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        mlflow_run = None
        if dp.log_to_mlflow:
            try:
                import mlflow
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "MLflow ist nicht installiert. Für DataPrep-Läufe mit Logging wird MLflow benötigt (pip install mlflow)."
                ) from e

            # MLflow Tracking: Konsolidiert unter work_dir.
            # MLFLOW_TRACKING_URI hat Vorrang (z.B. Remote-Server).
            import os
            work_dir = self.cfg.constants.resolved_work_dir
            os.makedirs(work_dir, exist_ok=True)
            if not os.environ.get("MLFLOW_TRACKING_URI"):
                # Lokaler File-Store ist hier der gewollte Betriebsmodus (Single-User-
                # Tracking, Artefakte konsolidiert unter work_dir). MLflow >=3.14 verlangt
                # dafuer ein explizites Opt-in, da der File-Store nur noch gewartet, aber
                # nicht weiterentwickelt wird — fuer lokales Tracking ohne Server-Features
                # ist er vollstaendig ausreichend.
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
                mlflow_dir = os.path.join(work_dir, "mlruns")
                mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")

            exp_name = dp.mlflow_experiment_name or self.cfg.mlflow.experiment_name
            mlflow.set_experiment(exp_name)

            # Run-Name: User-Eingabe oder auto-generiert
            from rubin.utils.run_names import generate_run_name
            run_name = dp.mlflow_run_name if dp.mlflow_run_name else generate_run_name("Datenaufbereitung")
            mlflow.start_run(run_name=run_name)
            mlflow_run = mlflow.active_run()
            mlflow.log_param("pipeline", "data_prep")
            # Snapshot der zentralen Konfiguration ist in DataPrep-Läufen sehr hilfreich,
            # weil Datenaufbereitung und Analyse damit reproduzierbar bleiben.
            if self.cfg.source_config_path and Path(str(self.cfg.source_config_path)).is_file():
                mlflow.log_artifact(self.cfg.source_config_path)

            # Experiment-Name im Output-Verzeichnis speichern, damit die
            # Analyse-Pipeline das gleiche Experiment verwenden kann.
            try:
                (out_dir / ".mlflow_experiment").write_text(exp_name, encoding="utf-8")
            except Exception:
                pass

            # Run-Name im Output-Verzeichnis speichern, damit die UI ihn
            # auf der Konfigurationsseite automatisch übernehmen kann.
            try:
                (out_dir / ".mlflow_run_name").write_text(run_name, encoding="utf-8")
            except Exception:
                pass

        try:
            return self._run_inner(dp, out_dir)
        finally:
            if dp.log_to_mlflow and mlflow_run is not None:
                import mlflow
                mlflow.end_run()

    def _run_inner(self, dp, out_dir) -> DataPrepOutputs:
        """Innere Logik der Datenaufbereitung (von run() aufgerufen)."""
        has_eval = dp.eval_data_path and len(dp.eval_data_path) > 0
        total = 7 if has_eval else 6
        step = [0]
        def _progress(label):
            step[0] += 1
            print(f"[rubin] Step {step[0]}/{total}: {label}", flush=True)

        _progress("Dateien einlesen")
        df = self._read_files()

        _progress("Deduplizierung")
        # Deduplizierung: auf einen Eintrag pro Kunde reduzieren.
        # Geschieht VOR der Treatment-Balance-Prüfung und Feature-Reduktion,
        # damit die Balance-Raten auf deduplizierten Daten berechnet werden
        # und die ID-Spalte anschließend entfernt werden kann.
        treat_col = str(dp.treatment).upper()
        eval_mask = None
        if dp.deduplicate:
            if not dp.deduplicate_id_column:
                raise ValueError(
                    "data_prep.deduplicate ist aktiviert, aber data_prep.deduplicate_id_column ist nicht gesetzt. "
                    "Bitte den Spaltennamen angeben, der die Kunden-ID enthält (z. B. 'PARTNER_ID')."
                )
            id_col = str(dp.deduplicate_id_column).upper()
            if id_col not in df.columns:
                raise ValueError(
                    f"Deduplizierungsspalte '{id_col}' ist nicht im Datensatz vorhanden. "
                    f"Verfügbare Spalten: {list(df.columns)[:20]}"
                )
            n_before = len(df)
            df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
            n_after = len(df)
            n_removed = n_before - n_after
            if n_removed > 0:
                self._logger.info(
                    "Deduplizierung auf '%s': %d → %d Zeilen (%d Duplikate entfernt).",
                    id_col, n_before, n_after, n_removed,
                )
            if dp.log_to_mlflow:
                import mlflow
                mlflow.log_param("deduplicate_column", id_col)
                mlflow.log_metric("deduplicate_rows_before", n_before)
                mlflow.log_metric("deduplicate_rows_after", n_after)
                mlflow.log_metric("deduplicate_rows_removed", n_removed)

        # ── Treatment-Balance-Prüfung bei mehreren Dateien (nach Dedup!) ──
        if "__file_source__" in df.columns and treat_col in df.columns:
            df = self._check_and_balance_treatments(df, treat_col, dp, _progress)

        # ── "Train Many, Evaluate Some": Eval-Maske extrahieren (nach Dedup!) ──
        if "__file_source__" in df.columns and dp.eval_file_index is not None:
            # Normalisiere zu Liste — die Config erlaubt Skalar ODER Liste (Union-Typ
            # in settings.py; die UI emittiert bei einer Datei die Skalar-Form).
            _eval_indices = [dp.eval_file_index] if isinstance(dp.eval_file_index, int) else list(dp.eval_file_index)
            valid_indices = sorted(df["__file_source__"].unique())
            _invalid = [i for i in _eval_indices if i not in valid_indices]
            if _invalid:
                self._logger.warning(
                    "eval_file_index %s nicht in Datei-Indizes %s — übersprungen.",
                    _invalid, valid_indices,
                )
            _valid_eval = [i for i in _eval_indices if i in valid_indices]
            if _valid_eval:
                eval_mask = df["__file_source__"].isin(_valid_eval).to_numpy()
                n_eval = int(eval_mask.sum())
                n_total = len(df)
                if n_eval == 0:
                    self._logger.warning("eval_file_index %s: Keine Zeilen nach Balance/Dedup. Eval-Maske wird nicht erstellt.", _valid_eval)
                    eval_mask = None
                else:
                    _idx_str = ", ".join(str(i) for i in _valid_eval)
                    self._logger.info(
                        "TMES: Dateien [%s] als Eval-Set (%d von %d Zeilen = %.1f%%).",
                        _idx_str, n_eval, n_total, n_eval / max(n_total, 1) * 100,
                    )

        # __file_source__ aufräumen
        if "__file_source__" in df.columns:
            df = df.drop(columns=["__file_source__"])

        _progress("Feature-Extraktion")

        score_col = str(dp.score_name).upper() if dp.score_name else None
        target_cols = self._target_cols()
        treat_col = str(dp.treatment).upper()

        if dp.features:
            # Explizite Feature-Liste aus UI oder YAML → höchste Priorität
            # Uppercase-Abgleich: dp.features kann Mixed-Case enthalten, df.columns sind UPPER
            features_upper = [str(f).upper() for f in dp.features]
            available_features = [f for f in features_upper if f in df.columns]
            X = df[available_features].copy()
            if dp.categorical_columns:
                cat_upper = [str(c).upper() for c in dp.categorical_columns]
                categorical_columns = [c for c in cat_upper if c in X.columns]
            else:
                categorical_columns = []
            self._logger.info("Explizite Feature-Liste: %d Features, %d kategorisch.", len(available_features), len(categorical_columns))
        elif dp.feature_path:
            # Feature-Dictionary vorhanden → verwende ROLE/NAME/LEVEL
            _fp = str(dp.feature_path)
            if _fp.lower().endswith(".csv"):
                feature_dictionary = pd.read_csv(_fp)
            else:
                # .xlsx / .xls → openpyxl, dann xlrd als Fallback
                try:
                    feature_dictionary = pd.read_excel(_fp, engine="openpyxl")
                except Exception:
                    feature_dictionary = pd.read_excel(_fp, engine="xlrd")
            # Dictionary-Spaltenheader case-insensitiv machen
            feature_dictionary.columns = [str(c).upper() for c in feature_dictionary.columns]
            required_feature_dict_columns = {"ROLE", "NAME", "LEVEL"}
            missing_feature_dict_columns = sorted(required_feature_dict_columns - set(feature_dictionary.columns))
            if missing_feature_dict_columns:
                raise ValueError(
                    "Im Feature-Dictionary fehlen Pflichtspalten: "
                    f"{missing_feature_dict_columns}. Erwartet werden mindestens ROLE, NAME und LEVEL."
                )
            if dp.log_to_mlflow:
                import mlflow
                mlflow.log_param("feature_path", dp.feature_path)

            input_list = feature_dictionary.loc[feature_dictionary["ROLE"].astype(str).str.upper() == "INPUT", "NAME"].astype(str).str.upper().tolist()
            available_features = [f for f in input_list if f in df.columns]
            X = df[available_features].copy()

            nominal_list = feature_dictionary.loc[feature_dictionary["LEVEL"].astype(str).str.upper() == "NOMINAL", "NAME"].astype(str).str.upper().tolist()
            categorical_columns = [c for c in nominal_list if c in X.columns]
            self._logger.info("Feature-Dictionary: %d INPUT-Features, %d NOMINAL.", len(available_features), len(categorical_columns))
        else:
            # Kein Feature-Dictionary → alle Spalten außer Target/Treatment/Score als Features
            exclude_cols = set(target_cols) | {treat_col}
            if score_col:
                exclude_cols.add(score_col)
            # Die explizit als ID bekannte Spalte (deduplicate_id_column) ist kein
            # Feature: IDs tragen kein kausales Signal, korrelieren aber häufig mit
            # Datei-/Zeitreihenfolge (z.B. TMES-Eval-Datei) und öffnen damit einen
            # Leakage-Kanal. Keine Namensheuristik — nur die konfigurierte Spalte.
            if dp.deduplicate_id_column:
                exclude_cols.add(str(dp.deduplicate_id_column).upper())
                exclude_cols.add(str(dp.deduplicate_id_column))
            available_features = [c for c in df.columns if c not in exclude_cols]
            X = df[available_features].copy()
            categorical_columns = []
            self._logger.info("Kein Feature-Dictionary: %d Features (alle außer Target/Treatment%s).",
                              len(available_features),
                              "/ID" if dp.deduplicate_id_column else "")

        if dp.score_as_feature and score_col and score_col in df.columns:
            X[score_col] = df[score_col]

        # Explizite categorical_columns aus YAML/UI überschreiben die Auto-Erkennung
        if dp.categorical_columns and not dp.features:
            cat_upper = [str(c).upper() for c in dp.categorical_columns]
            categorical_columns = [c for c in cat_upper if c in X.columns]


        else:
            # Auto-Modus: object/category-Spalten als kategorisch, numerische Spalten retten
            # CSV/SAS-Dateien laden numerische Spalten als 'object', wenn sie Werte wie
            # ".", "", "N/A" oder Leerzeichen enthalten. Solche Spalten werden hier
            # per pd.to_numeric() zurückkonvertiert, statt sie als kategorisch zu behandeln.
            #
            # WICHTIG: Explizit als kategorisch gelistete Spalten (Feature-Dictionary,
            # Config categorical_columns) werden IMMER als kategorisch übernommen —
            # auch wenn sie einen numerischen Dtype haben (z.B. label-encoded 0/1/2).
            # Die Reklassifizierung greift NUR für auto-erkannte object-Spalten.
            _coerced_count = 0
            _converted_numeric_cat = []
            validated_categorical = []
            for col in X.columns:
                if col in categorical_columns:
                    # Explizit als kategorisch konfiguriert → IMMER übernehmen.
                    # Numerische Spalten (int/float) werden zu category konvertiert —
                    # der User/das Feature-Dictionary hat entschieden.
                    if is_object_like_dtype(X[col].dtype):
                        # Object-Spalte: Prüfen ob sie wirklich Text ist oder
                        # nur wegen Parsing-Artefakten als object geladen wurde
                        coerced = pd.to_numeric(X[col], errors="coerce")
                        n_orig_valid = X[col].notna().sum()
                        n_coerced_valid = coerced.notna().sum()
                        if n_orig_valid > 0 and n_coerced_valid / n_orig_valid >= 0.95:
                            # Fast alle Werte numerisch → als int laden, dann category
                            X[col] = coerced.fillna(-1).astype("int32")
                            _coerced_count += 1
                    if X[col].dtype.kind in ("i", "u", "f"):
                        _converted_numeric_cat.append(col)
                    validated_categorical.append(col)
                elif is_object_like_dtype(X[col].dtype):
                    coerced = pd.to_numeric(X[col], errors="coerce")
                    n_orig_valid = X[col].notna().sum()
                    n_coerced_valid = coerced.notna().sum()
                    if n_orig_valid > 0 and n_coerced_valid / n_orig_valid >= 0.5:
                        X[col] = coerced
                        _coerced_count += 1
                    else:
                        validated_categorical.append(col)
                elif X[col].dtype.name == "category":
                    validated_categorical.append(col)

            categorical_columns = validated_categorical

            if _converted_numeric_cat:
                self._logger.info(
                    "Kategorische Spalten mit numerischem Dtype: %d Spalten "
                    "(explizit als kategorisch konfiguriert, werden als category behandelt): %s",
                    len(_converted_numeric_cat),
                    _converted_numeric_cat[:10] if len(_converted_numeric_cat) <= 10
                    else f"{_converted_numeric_cat[:5]}... (+{len(_converted_numeric_cat)-5})",
                )
            if _coerced_count:
                self._logger.info(
                    "Auto-Typkonversion: %d object-Spalten als numerisch erkannt und konvertiert.",
                    _coerced_count,
                )

        # Y, T, optional S
        if len(target_cols) == 1:
            Y = df[target_cols[0]].to_numpy().ravel()
        else:
            Y = df[target_cols].sum(axis=1).to_numpy().ravel()
            self._logger.info("Multi-Target: %d Spalten (%s) aufsummiert.", len(target_cols), ", ".join(target_cols))
        T = df[treat_col].to_numpy().ravel()
        S = None
        if score_col and score_col in df.columns:
            S = df[score_col].to_numpy().ravel()

        if dp.binary_target:
            Y = (Y > 0).astype(int)
            if dp.log_to_mlflow:
                import mlflow

                mlflow.log_param("binary_target_conversion", "Target converted to binary (0/1)")

        _progress("Preprocessing (Encoding, NaN)")
        preproc = fit_preprocessor(X, categorical_columns, fill_na_method=dp.fill_na_method)
        Xp = preproc.transform(X)

        # Artefakte persistieren
        with open(out_dir / "encoding.obj", "wb") as f:
            pickle.dump(preproc.encoding_maps, f)
        with open(out_dir / "missing_values.json", "w", encoding="utf-8") as f:
            json.dump(preproc.fillna_values, f, indent=2, ensure_ascii=False)

        _progress("Memory-Reduktion")
        Xp = reduce_mem_usage(Xp)
        _progress("Artefakte speichern")
        # ── Stale-Dateien aus vorherigen Runs entfernen ──
        # Optionale Dateien (S, eval_mask, eval_*) werden nur geschrieben wenn
        # sie in diesem Run erzeugt wurden. Ohne Cleanup würden veraltete Dateien
        # aus früheren Runs persistieren und Length-Mismatches verursachen.
        for _stale in ["S.parquet", "eval_mask.npy",
                       "X_eval.parquet", "T_eval.parquet", "Y_eval.parquet", "S_eval.parquet"]:
            _stale_path = out_dir / _stale
            if _stale_path.exists():
                _stale_path.unlink()
                self._logger.debug("Stale-Datei entfernt: %s", _stale)
        # Nach der Speicherreduktion muss auch das serialisierte Preprocessing den finalen
        # Zielzustand kennen. Sonst würden spätere Transforms wieder auf die alten Typen casten.
        preproc.dtypes_after = Xp.dtypes.apply(lambda x: x.name).to_dict()
        try:
            Xp.to_parquet(out_dir / "X.parquet")
        except ImportError as e:
            raise ImportError(
                "X.parquet konnte nicht geschrieben werden, weil kein Parquet-Engine verfügbar ist. "
                "Bitte 'pyarrow' oder 'fastparquet' installieren."
            ) from e

        dtypes_dict = dict(preproc.dtypes_after)
        with open(out_dir / "dtypes.json", "w", encoding="utf-8") as f:
            json.dump(dtypes_dict, f, indent=2, ensure_ascii=False)

        try:
            pd.DataFrame({"T": T}, index=Xp.index).to_parquet(out_dir / "T.parquet")
            pd.DataFrame({"Y": Y}, index=Xp.index).to_parquet(out_dir / "Y.parquet")
        except ImportError as e:
            raise ImportError(
                "T.parquet/Y.parquet konnten nicht geschrieben werden, weil kein Parquet-Engine verfügbar ist. "
                "Bitte 'pyarrow' oder 'fastparquet' installieren."
            ) from e
        if S is not None:
            try:
                pd.DataFrame({"S": S}, index=Xp.index).to_parquet(out_dir / "S.parquet")
            except ImportError as e:
                raise ImportError(
                    "S.parquet konnte nicht geschrieben werden, weil kein Parquet-Engine verfügbar ist. "
                    "Bitte 'pyarrow' oder 'fastparquet' installieren."
                ) from e

        with open(out_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(preproc, f)

        # ── Eval-Maske speichern (Train Many, Evaluate Some) ──
        if eval_mask is not None:
            np.save(out_dir / "eval_mask.npy", eval_mask)
            self._logger.info("eval_mask.npy gespeichert: %d Eval-Zeilen von %d.", int(eval_mask.sum()), len(eval_mask))
            if dp.log_to_mlflow:
                try:
                    import mlflow
                    mlflow.log_param("eval_file_index", dp.eval_file_index)
                    mlflow.log_metric("eval_mask_rows", int(eval_mask.sum()))
                except Exception:
                    pass

        # ── Eval-Daten: fit-on-train, transform-on-eval ──
        if has_eval:
            _progress("Eval-Daten transformieren")
            try:
                df_eval = self._read_files(file_paths=dp.eval_data_path, merge_only=True)
                if dp.deduplicate and dp.deduplicate_id_column:
                    id_col = str(dp.deduplicate_id_column).upper()
                    if id_col in df_eval.columns:
                        df_eval = df_eval.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)

                # Gleiche Feature-Extraktion wie Train (gleiche feature_dictionary)
                X_eval = df_eval[[c for c in available_features if c in df_eval.columns]].copy()
                if dp.score_as_feature and score_col and score_col in df_eval.columns:
                    X_eval[score_col] = df_eval[score_col]
                # Fehlende Spalten auffüllen (preproc.transform kümmert sich)
                if len(target_cols) == 1:
                    Y_eval = df_eval[target_cols[0]].to_numpy().ravel()
                else:
                    Y_eval = df_eval[target_cols].sum(axis=1).to_numpy().ravel()
                T_eval = df_eval[treat_col].to_numpy().ravel()
                S_eval = None
                if score_col and score_col in df_eval.columns:
                    S_eval = df_eval[score_col].to_numpy().ravel()

                if dp.binary_target:
                    Y_eval = (Y_eval > 0).astype(int)

                # Transform mit dem auf Train gefitteten Preprocessor
                Xp_eval = preproc.transform(X_eval)
                Xp_eval = reduce_mem_usage(Xp_eval)

                # Eval-Artefakte speichern
                Xp_eval.to_parquet(out_dir / "X_eval.parquet")
                pd.DataFrame({"T": T_eval}, index=Xp_eval.index).to_parquet(out_dir / "T_eval.parquet")
                pd.DataFrame({"Y": Y_eval}, index=Xp_eval.index).to_parquet(out_dir / "Y_eval.parquet")
                if S_eval is not None:
                    pd.DataFrame({"S": S_eval}, index=Xp_eval.index).to_parquet(out_dir / "S_eval.parquet")

                self._logger.info(
                    "Eval-Daten transformiert: %d Zeilen, %d Features (Train-Preprocessor angewendet).",
                    len(Xp_eval), Xp_eval.shape[1],
                )
            except Exception as e:
                self._logger.error("Eval-Daten-Transformation fehlgeschlagen: %s", e, exc_info=True)
                raise

        # Schema zusätzlich separat ablegen, damit Datenänderungen schon vor dem Laden des Pickles
        # sichtbar gemacht werden können.
        try:
            from rubin.utils.schema_utils import save_schema, Schema
            save_schema(Schema.from_dataframe(Xp, categorical_columns=categorical_columns), str(out_dir / "schema.json"))
        except Exception:
            pass

        # DataPrep-Konfiguration im Output-Verzeichnis speichern,
        # damit die Analyse-Pipeline sie bei Bedarf nach MLflow loggen kann.
        # categorical_columns wird mit der TATSÄCHLICHEN Liste nach Typ-Erkennung
        # überschrieben, nicht mit der ursprünglichen Config-Liste.
        try:
            import yaml as _yaml
            dp_cfg_dict = {"data_prep": dp.model_dump() if hasattr(dp, "model_dump") else dp.dict()}
            dp_cfg_dict["data_prep"]["categorical_columns"] = categorical_columns
            (out_dir / "dataprep_config.yml").write_text(
                _yaml.dump(dp_cfg_dict, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        # Optional: MLflow-Logging der Artefakte
        if dp.log_to_mlflow:
            import mlflow

            artifact_files = [
                "encoding.obj",
                "missing_values.json",
                "X.parquet",
                "dtypes.json",
                "T.parquet",
                "Y.parquet",
                "preprocessor.pkl",
                "schema.json",
                "dataprep_config.yml",
            ]
            if S is not None:
                artifact_files.append("S.parquet")
            if has_eval:
                artifact_files.extend(["X_eval.parquet", "T_eval.parquet", "Y_eval.parquet"])
                if (out_dir / "S_eval.parquet").exists():
                    artifact_files.append("S_eval.parquet")

            for fn in artifact_files:
                path = out_dir / fn
                if path.exists():
                    mlflow.log_artifact(str(path))

        print(f"[rubin] Step {total}/{total}: Fertig", flush=True)
        return DataPrepOutputs(X=Xp, T=T, Y=Y, S=S, preprocessor=preproc)
