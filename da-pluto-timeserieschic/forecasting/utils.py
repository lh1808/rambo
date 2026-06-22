"""
Hilfsfunktionen für den Rolling-Block-Backtest und das Verketten von
Vorhersageblöcken.

Die zentrale Funktion ist :func:`rolling_block_forecast`. Sie trainiert in
einem gleitenden Fenster je Block ein frisches Modell und erzeugt
Vorhersageblöcke fester Länge. Über :func:`_merge_blocks` werden die
Einzelblöcke zu einer konsistenten Zeitreihe zusammengeführt.
"""

from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from darts import TimeSeries


def slice_from(ts: TimeSeries, start) -> TimeSeries:
    """
    Gibt eine Teilserie ab ``start`` (inklusive) bis zum Ende von ``ts`` zurück.

    Parameters
    ----------
    ts
        Quellserie.
    start
        Startzeitpunkt als ``pd.Timestamp`` (oder Integer bei
        RangeIndex-basierten Serien). Muss zum Zeitindex von ``ts`` passen.

    Returns
    -------
    TimeSeries
        Zeitscheibe ``[start, ts.end_time()]`` (beidseitig inklusive).
    """
    return ts.slice(start, ts.end_time())


def _merge_blocks(
    blocks: List[TimeSeries],
    mode: str = "keep_last",
) -> Optional[TimeSeries]:
    """
    Führt mehrere (ggf. überlappende) Vorhersageblöcke zu einer Serie
    zusammen.

    Parameters
    ----------
    blocks
        Liste der zu verbindenden Zeitreihen. Alle Blöcke müssen dieselben
        Komponenten und dieselbe Frequenz haben.
    mode
        Strategie bei Überlappung:

        - ``"keep_last"``  – bei doppeltem Zeitstempel gewinnt der Block,
          der in ``blocks`` später steht.
        - ``"keep_first"`` – der früher gereihte Block gewinnt.
        - ``"stack"``      – Blöcke werden als weitere Komponenten gestackt.

    Returns
    -------
    TimeSeries | None
        Zusammengeführte Zeitreihe. ``None`` wenn ``blocks`` leer ist.
    """
    if not blocks:
        return None

    if mode == "stack":
        merged = blocks[0]
        for b in blocks[1:]:
            merged = merged.stack(b)
        return merged

    if mode not in ("keep_last", "keep_first"):
        raise ValueError(f"Unbekannter merge-Modus: {mode}")

    df_all = pd.concat([b.to_dataframe() for b in blocks])
    keep = "last" if mode == "keep_last" else "first"
    df_all = df_all[~df_all.index.duplicated(keep=keep)].sort_index()

    return TimeSeries.from_dataframe(
        df_all,
        freq=blocks[0].freq,
        static_covariates=blocks[0].static_covariates,
    )


def rolling_block_forecast(
    model_builder: Callable[[], Any],
    y: TimeSeries,
    past_cov: Optional[TimeSeries],
    future_cov: Optional[TimeSeries],
    test_start,
    train_length: int,
    horizon: int,
    stride: int,
    fit_kwargs: Optional[Dict] = None,
    predict_kwargs: Optional[Dict] = None,
    merge: str = "keep_last",
    required_input_chunk_length: Optional[int] = None,
    verbose: bool = False,
    transforms_builder: Optional[Callable[[], Any]] = None,
    window_mode: str = "fixed",
) -> Dict[str, Any]:
    """
    Rolling-Block-Backtest: trainiert pro Fenster ein frisches Modell und
    erzeugt Vorhersageblöcke der Länge ``horizon``.

    Parameters
    ----------
    model_builder
        Callable, das pro Block ein frisches (untrainiertes) Modell liefert.
    y
        Zielzeitreihe. Ist ``transforms_builder`` gesetzt, in Originaleinheit;
        sonst bereits vom Caller transformiert.
    past_cov, future_cov
        Past- bzw. Future-Kovariaten. ``future_cov`` muss bis mindestens
        ``test_start + horizon - 1`` reichen.
    test_start
        Zeitpunkt (oder Positionsindex), ab dem die ersten Predict-Werte
        erzeugt werden sollen.
    train_length
        Größe des Trainingsfensters in Zeitschritten. Im Modus
        ``"fixed"`` wird exakt dieses Fenster verwendet; im Modus
        ``"expanding"`` ist es die Obergrenze.
    horizon
        Vorhersagelänge pro Block.
    stride
        Schrittweite, mit der ``test_start`` zwischen den Blöcken verschoben
        wird.
    fit_kwargs, predict_kwargs
        Zusätzliche Keyword-Arguments für ``model.fit`` / ``model.predict``.
    merge
        Merge-Strategie für die Blöcke; siehe :func:`_merge_blocks`.
    required_input_chunk_length
        Minimale Länge des Trainingsfensters, unterhalb derer der Block
        übersprungen (``"fixed"``) bzw. zum nächsten weitergesprungen
        (``"expanding"``) wird. Sollte auf
        ``input_chunk_length + output_chunk_length`` gesetzt werden,
        da darts diese Gesamtlänge für mindestens ein Trainings-Sample
        benötigt.
    verbose
        Gibt pro Block die Zeiträume von Training und Prognose aus.
    transforms_builder
        Optionaler Factory-Callable, der pro Block frische
        ``TransformArtifacts`` liefert. Ist der Parameter gesetzt, werden
        ``y`` und ``past_cov`` pro Trainingsfenster lokal gefittet – die
        Skalierer sehen ausschließlich das aktuelle Trainingsfenster und
        niemals zukünftige Validierungsdaten. Die zurückgegebenen Blöcke
        sind dann bereits invers-transformiert, also in Originaleinheit.
        ``future_cov`` wird in keinem Fall skaliert (typischerweise binäre
        oder kleine Zählvariablen).
    window_mode
        Steuerung des Trainingsfensters:

        - ``"fixed"`` – jeder Block trainiert auf exakt
          ``train_length`` Zeitschritten. Blöcke, für die nicht genug
          Historie verfügbar ist, werden übersprungen.
        - ``"expanding"`` – das Fenster beginnt am Serienstart und
          wächst mit jedem Block. ``train_length`` wirkt als Obergrenze.

    Returns
    -------
    dict
        ``{"blocks": [TimeSeries, ...], "merged": TimeSeries | None}``.
    """
    if window_mode not in ("fixed", "expanding"):
        raise ValueError(f"Unbekannter window_mode: {window_mode!r}. Erlaubt: 'fixed', 'expanding'.")

    fit_kwargs = fit_kwargs or {}
    predict_kwargs = predict_kwargs or {}

    def _idx(ts: TimeSeries, pos: int):
        return ts.time_index[pos]

    last_valid_start = _idx(y, len(y) - horizon)

    if isinstance(test_start, int):
        cur_start = _idx(y, test_start)
    else:
        cur_start = test_start

    blocks: List[TimeSeries] = []

    while True:
        cur_pos = y.get_index_at_point(cur_start)
        train_end_pos = cur_pos - 1

        if window_mode == "fixed":
            train_start_pos = cur_pos - train_length
            if train_start_pos < 0:
                # Nicht genug Historie für ein volles Fenster → Block
                # überspringen und mit dem nächsten Stride-Schritt
                # weitermachen, falls dort genug Daten vorhanden sind.
                if verbose:
                    print(
                        f"Überspringe Block bei {cur_start.date()}: "
                        f"nur {cur_pos} Wochen verfügbar, "
                        f"benötigt {train_length}."
                    )
                next_pos = cur_pos + stride
                if next_pos >= len(y) or _idx(y, next_pos) > last_valid_start:
                    break
                cur_start = _idx(y, next_pos)
                continue
        else:
            # expanding: nimm alles ab Serienstart, aber höchstens train_length
            train_start_pos = max(0, cur_pos - train_length)

        if train_end_pos <= train_start_pos:
            break

        train_start = _idx(y, train_start_pos)
        train_end = _idx(y, train_end_pos)

        y_train_raw = y.slice(train_start, train_end)

        if past_cov is not None:
            past_train_raw = past_cov.slice(train_start, train_end)
            past_full_raw = slice_from(past_cov, train_start)
        else:
            past_train_raw = None
            past_full_raw = None

        if future_cov is not None:
            fut_train = future_cov.slice(train_start, train_end)
            fut_full = slice_from(future_cov, train_start)
        else:
            fut_train = None
            fut_full = None

        if required_input_chunk_length is not None and len(y_train_raw) < required_input_chunk_length:
            if verbose:
                print(
                    f"Trainingsfenster zu kurz ({len(y_train_raw)}) für "
                    f"required_input_chunk_length={required_input_chunk_length}."
                )
            if window_mode == "fixed":
                # Im Fixed-Modus haben alle Blöcke dieselbe Länge →
                # alle folgenden wären genauso kurz.
                break
            else:
                # Im Expanding-Modus werden spätere Blöcke länger →
                # diesen überspringen, nächsten versuchen.
                next_pos = cur_pos + stride
                if next_pos >= len(y) or _idx(y, next_pos) > last_valid_start:
                    break
                cur_start = _idx(y, next_pos)
                continue

        # Pro-Block-Transformation: die Skalierer sehen nur das aktuelle
        # Trainingsfenster, nie die späteren Validierungsdaten.
        if transforms_builder is not None:
            transforms = transforms_builder()
            y_train = transforms.y_pipeline.fit_transform(y_train_raw)
            if past_train_raw is not None:
                # Scaler nur auf dem Trainingsfenster fitten, damit keine
                # zukünftigen Werte in die Skalierungsparameter einfließen.
                past_train = transforms.x_scaler.fit_transform(past_train_raw)
                past_full = transforms.x_scaler.transform(past_full_raw)
            else:
                past_full = None
                past_train = None
        else:
            transforms = None
            y_train = y_train_raw
            past_train = past_train_raw
            past_full = past_full_raw

        model = model_builder()
        model.fit(
            series=y_train,
            past_covariates=past_train,
            future_covariates=fut_train,
            **fit_kwargs,
        )

        y_pred = model.predict(
            n=horizon,
            past_covariates=past_full,
            future_covariates=fut_full,
            **predict_kwargs,
        )

        if transforms is not None:
            y_pred = transforms.y_pipeline.inverse_transform(y_pred)

        # Auf den Test-Start ausrichten, falls das Modell mit vorgelagertem
        # Kontext predictet.
        if y_pred.start_time() != cur_start:
            y_pred = slice_from(y_pred, cur_start)

        blocks.append(y_pred)

        if verbose:
            print(
                f"[Block] train [{train_start.date()} .. {train_end.date()}] "
                f"-> pred [{y_pred.start_time().date()} .. {y_pred.end_time().date()}] "
                f"(len={len(y_pred)})"
            )

        next_pos = cur_pos + stride
        if next_pos >= len(y):
            break
        next_start = _idx(y, next_pos)

        if next_start > last_valid_start:
            if verbose:
                print(
                    "Stop: nächster Blockstart läge hinter dem letzten gültigen "
                    "Start für vollen Horizont."
                )
            break

        cur_start = next_start

    if not blocks:
        return {"blocks": [], "merged": None}

    return {"blocks": blocks, "merged": _merge_blocks(blocks, mode=merge)}
