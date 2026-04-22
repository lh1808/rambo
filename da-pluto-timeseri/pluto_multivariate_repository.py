"""
DB2-Repository für die multivariate Termineingangs-Prognose in PLUTO.

Lesen
-----
:meth:`PlutoMultivariateRepository.read_timeseries` liest die Fakttabelle
und liefert ein breites Pandas-DataFrame mit einer Spalte je
``KENNZAHL__PRODUKT__STATUS``-Kombination.

Schreiben
---------
:meth:`PlutoMultivariateRepository.write_forecast` schreibt die Prognose
per ``DELETE + executemany INSERT`` in einer atomaren Transaktion in die
Zieltabelle. Bei Fehler wird ein ``ROLLBACK`` ausgeführt, sodass die
Zieltabelle auf dem bisherigen Stand verbleibt.
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
from typing import List, Tuple

import ibm_db
import ibm_db_dbi
import pandas as pd


class PlutoMultivariateRepository:
    """
    Repository für die multivariate Prognose der Termineingänge in PLUTO.

    Erwartete Tabellenstruktur laut Vorgabe:

    Lesetabelle (z. B. t7.TA_DA_PLUTO_SP_2025)
    ------------------------------------------------
    DIM_ZEIT        INTEGER (YYYYMMDD)
    DIM_KENNZAHL    VARCHAR
    DIM_PRODUKT     VARCHAR(50)
    DIM_SCHADENSTATUS VARCHAR(50)
    KENNZAHLWERT    DECIMAL(20,9)

    Schreibtabelle (z. B. t7.TA_DA_PLUTO_SP_2025_PROGNOSE)
    ------------------------------------------------------
    DIM_ZEIT        INTEGER (YYYYMMDD)
    DIM_KENNZAHL    VARCHAR
    DIM_PRODUKT     VARCHAR(50)
    DIM_SCHADENSTATUS VARCHAR(50)
    KENNZAHLWERT    DECIMAL(20,9)
    LOAD_DATE       DATE
    """

    def __init__(
        self,
        kpi_list: List[str] | None = None,
        product_list: List[str] | None = None,
        status_list: List[str] | None = None,
        source_table: str | None = None,
        target_table: str | None = None,
    ) -> None:
        """
        Baut eine DB2-Verbindung über ``ibm_db`` auf.

        Die Zugangsdaten werden aus Umgebungsvariablen gelesen:
        ``DB2_USERNAME``, ``DB2_PASSWORT``, ``DB2_HOST``, ``DB2_DB_NAME``,
        ``DB2_PORT``, ``DB2_SCHEMA``. Fehlt eine davon, wird beim
        ``os.environ``-Zugriff ein :class:`KeyError` geworfen.

        Parameters
        ----------
        kpi_list, product_list, status_list
            Zu ladende Fachdimensionen. ``None`` setzt die Defaults.
        source_table, target_table
            Voll qualifizierte Tabellennamen (``schema.name``). ``None``
            erzeugt die Standardnamen aus ``DB2_SCHEMA``.

        Raises
        ------
        Exception
            Wenn der DB2-Connect fehlschlägt – wird geloggt und
            re-raised.
        """
        self._logger = logging.getLogger(__name__)
        self.username = os.environ["DB2_USERNAME"]
        self.password = os.environ["DB2_PASSWORT"]
        self.host = os.environ["DB2_HOST"]
        self.db_name = os.environ["DB2_DB_NAME"]
        self.port = os.environ["DB2_PORT"]
        self.schema = os.environ["DB2_SCHEMA"]

        self.kpi_list = kpi_list or [
            "TERM_EINGANG_SCHRIFTST",
            "TERM_EINGANG_SONST",
        ]
        self.product_list = product_list or [
            "KFZ_Vollkasko",
            "KFZ_Teilkasko",
            "KFZ_Haftpflicht",
            "KFZ_Rest",
            "HUS_Haftpflicht",
            "HUS_Wohngebäude",
            "HUS_Hausrat",
            "HUS_Rest",
        ]
        self.status_list = status_list or [
            "Neuschaden",
            "Folgebearbeitung",
        ]

        if source_table is None:
            self.source_table = f"{self.schema}.TA_DA_PLUTO_SP_2025"
        else:
            self.source_table = source_table

        if target_table is None:
            self.target_table = f"{self.schema}.TA_DA_PLUTO_SP_2025_PROGNOSE"
        else:
            self.target_table = target_table

        dsn = [
            f"DATABASE={self.db_name};",
            f"HOSTNAME={self.host};",
            f"PORT={self.port};",
            "PROTOCOL=TCPIP;",
            f"UID={self.username};",
            f"PWD={self.password};",
        ]
        try:
            self._connection: ibm_db.IBM_DBConnection | None = ibm_db.connect("".join(dsn), "", "")
            self._conn: ibm_db_dbi.Connection = ibm_db_dbi.Connection(self._connection)
            self._logger.info("Connection to DB2 database was successful.")
            self._logger.info(f"Using source table: {self.source_table}")
            self._logger.info(f"Using target table: {self.target_table}")
        except Exception as e:
            self._connection = None
            self._logger.error(f"Error while connecting to DB2 database: {e}")
            raise

    @staticmethod
    @staticmethod
    def _dim_zeit_to_datetime(dim_zeit: int) -> datetime:
        """Wandelt ``YYYYMMDD``-Integer in einen ``datetime``."""
        return datetime.strptime(str(dim_zeit), "%Y%m%d")

    def read_timeseries(self) -> pd.DataFrame:
        """
        Liest die multivariaten Termineingänge aus der Quelltabelle und
        liefert ein breites DataFrame mit einer Spalte je
        ``KENNZAHL__PRODUKT__STATUS``-Kombination.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (aus ``DIM_ZEIT``), Spaltenformat
            ``KENNZAHL__PRODUKT__STATUS``. Leeres DataFrame, wenn keine
            Zeilen zurückkommen.

        Raises
        ------
        Exception
            SQL- oder Treiber-Fehler werden geloggt und re-raised.
        """
        placeholders = ", ".join(["?"] * len(self.kpi_list))
        query = f"""
            SELECT
                DIM_ZEIT,
                DIM_KENNZAHL,
                DIM_PRODUKT,
                DIM_SCHADENSTATUS,
                KENNZAHLWERT
            FROM {self.source_table}
            WHERE DIM_KENNZAHL IN ({placeholders})
        """

        try:
            # ibm_db_dbi ist ein DBAPI2-Connector; pandas bevorzugt zwar
            # SQLAlchemy, aber ibm_db_sa ist im Ziel-Environment nicht
            # verfügbar. Die Warnung ist rein kosmetisch – das Verhalten
            # ist korrekt und stabil.
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*pandas only supports SQLAlchemy.*",
                    category=UserWarning,
                )
                df_long = pd.read_sql(
                    sql=query,
                    con=self._conn,
                    params=self.kpi_list,
                )
        except Exception as e:
            self._logger.error(f"Error while reading time series from DB2: {e}")
            raise

        if df_long.empty:
            self._logger.warning("No data returned from source table.")
            return pd.DataFrame()

        df_long["date"] = df_long["DIM_ZEIT"].astype(int).apply(self._dim_zeit_to_datetime)
        df_long["date"] = pd.to_datetime(df_long["date"])

        df_pivot = df_long.pivot_table(
            index="date",
            columns=["DIM_KENNZAHL", "DIM_PRODUKT", "DIM_SCHADENSTATUS"],
            values="KENNZAHLWERT",
            aggfunc="sum",
        )

        df_pivot = df_pivot.sort_index()

        df_pivot.columns = [
            f"{kpi}__{produkt}__{status}"
            for (kpi, produkt, status) in df_pivot.columns.to_list()
        ]

        self._logger.info(
            "Loaded multivariate time series from DB2. "
            f"Rows: {len(df_pivot)}, Columns (components): {len(df_pivot.columns)}"
        )

        return df_pivot

    @staticmethod
    def _parse_component_name(col_name: str) -> Tuple[str, str, str]:
        """
        Zerlegt einen Spaltennamen im Format ``KENNZAHL__PRODUKT__STATUS``
        in seine drei Bestandteile.

        Raises
        ------
        ValueError
            Wenn der Name nicht exakt drei durch ``__`` getrennte Teile hat.
        """
        parts = col_name.split("__")
        if len(parts) != 3:
            raise ValueError(f"Unexpected component name format: {col_name}")
        return parts[0], parts[1], parts[2]

    def write_forecast(self, forecast: pd.DataFrame) -> None:
        """
        Schreibt die multivariate Prognose atomar in die Zieltabelle.

        Ablauf: ``DELETE FROM <target>`` + ``executemany INSERT`` in einer
        einzigen Transaktion. Bei Fehler wird ein ``ROLLBACK`` ausgeführt
        und die Zieltabelle bleibt auf dem bisherigen Stand.

        ``DELETE`` ist in DB2 — anders als ``TRUNCATE TABLE IMMEDIATE`` —
        vollständig rollback-fähig. Dadurch ist sichergestellt, dass die
        Zieltabelle niemals in einem leeren Zustand verbleibt.
        """
        if forecast.empty:
            self._logger.warning("Forecast DataFrame is empty. Nothing to write.")
            return

        # Prognose in Tupel-Liste überführen (überspringe NaN-Werte).
        rows: List[Tuple] = []
        for idx, row in forecast.iterrows():
            ts = pd.to_datetime(idx)
            dim_zeit = int(ts.strftime("%Y%m%d"))
            for col_name, value in row.items():
                if pd.isna(value):
                    continue
                dim_kennzahl, dim_produkt, dim_status = self._parse_component_name(col_name)
                rows.append(
                    (dim_zeit, dim_kennzahl, dim_produkt, dim_status, float(value))
                )

        if not rows:
            self._logger.warning(
                "Forecast enthält ausschließlich NaN-Werte. Nothing to write."
            )
            return

        cursor: ibm_db_dbi.Cursor = self._conn.cursor()

        try:
            # Atomarer Austausch: DELETE + INSERT in einer Transaktion.
            # Bei Fehler → ROLLBACK → alter Stand bleibt erhalten.
            cursor.execute(f"DELETE FROM {self.target_table}")

            insert_sql = f"""
                INSERT INTO {self.target_table} (
                    DIM_ZEIT,
                    DIM_KENNZAHL,
                    DIM_PRODUKT,
                    DIM_SCHADENSTATUS,
                    KENNZAHLWERT,
                    LOAD_DATE
                )
                VALUES (?, ?, ?, ?, ?, CURRENT DATE)
            """
            cursor.executemany(insert_sql, rows)
            self._conn.commit()

            self._logger.info(
                "Forecast in %s geschrieben (%d Zeilen).",
                self.target_table,
                len(rows),
            )
        except Exception as e:
            self._logger.error("Fehler beim Schreiben der Prognose: %s", e)
            try:
                self._conn.rollback()
            except Exception as rb_err:
                self._logger.error("Rollback ebenfalls fehlgeschlagen: %s", rb_err)
            raise

    def close_connection(self) -> None:
        """
        Schließt die DB2-Verbindung. Ist idempotent – mehrfaches Aufrufen
        ist harmlos, ebenso der Aufruf auf einer nie erfolgreich
        aufgebauten Verbindung.
        """
        if getattr(self, "_connection", None):
            ibm_db.close(self._connection)
            self._logger.info("DB2 connection successfully closed.")
