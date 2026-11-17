
from pathlib import Path

import jinja2
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm
from loguru import logger
from sqlalchemy import text

delta_table_create_sql = """
CREATE TEMPORARY TABLE SESSION.vertrag_delta (
    VERTRAG_ID BIGINT,
    VORGANGSNUMMER BIGINT,
    VERTRAGSAKTE_ID BIGINT
)
"""

delta_table_upload_sql = """
INSERT INTO SESSION.vertrag_delta VALUES (?, ?, ?)
"""


class DWHDeltaPullChunked:
    """
    Contains the logic to extract the ids of all contracts of KFZ/HUS Bestand
    that have changed since the last delta table pull.

    It also includes a method to remove all information of contracts
    that have changed from the KW database so that they can be replaced
    by newly computed client values.

    Parameters:

    session_dwh:     SQLAlchemy session connected to the DWH DB2 database.
    session_dakw:    SQLAlchemy session connected to the KW DB2 database.
    stichtag:        Date you want to draw the information of (normally today).
    last_delta_date: Date you have performed the last delta pull.
    debug:           Flag to pull all data or only a sample.
    """

    def __init__(
        self,
        product: str,
        session_dwh: sqlalchemy.orm.session.Session,
        session_dakw: sqlalchemy.orm.session.Session,
        stichtag: str,
        last_delta_date: str,
        debug: bool,
        dwh_db_schema: str,
        kw_db_schema: str,
        kw_hc_hca_table: str,
        kw_h24_table: str,
    ):
        if product in ["KFZ", "HUS"]:
            self._product = product
        else:
            raise Exception("Invalid product type - must by KFZ or HUS.")
        self._session_dwh = session_dwh
        self._connection_dwh = self._session_dwh.connection()
        self._session_dakw = session_dakw
        self._stichtag = stichtag
        self._last_delta_date = last_delta_date
        self._debug = debug
        self._dwh_db_schema = dwh_db_schema
        self._kw_db_schema = kw_db_schema
        self._kw_hc_hca_table = kw_hc_hca_table
        self._kw_h24_table = kw_h24_table

        sql_delta = (
            Path(__file__).parent
            / f"{self._product.lower()}_delta_pull_sql/delta_{self._product.lower()}_incremental_pure.sql"
        )
        query_template = jinja2.Template(sql_delta.read_text())
        context = {
            "last_delta_date": self._last_delta_date,
            "dwh_schema": self._dwh_db_schema,
            "debug": self._debug,
        }
        query = query_template.render(**context)
        self._session_dwh.execute(text(query))
        deltas = session_dwh.execute(
            text(f"SELECT * FROM SESSION.VERTRAG_DELTA_{self._product}_PURE")
        ).fetchall()
        self.ids = pd.DataFrame(deltas, columns=["delta_contract_id"])
        logger.info(f"# of Delta_IDs {len(self.ids)} for {self._product}")
        # logger.info(f"{','.join(map(str, self.ids['delta_contract_id']))}")

    def pull_delta_2(self, engine: sa.engine.Engine):
        if len(self.ids) == 0:
            logger.info(f"No contracts have changed since {self._last_delta_date}.")
            return "No changes"

        sql_file = (
            Path(__file__).parent
            / f"{self._product.lower()}_delta_pull_sql/delta_{self._product.lower()}_inc_blank.sql"
        )
        query_template = jinja2.Template(sql_file.read_text())

        batch_size = 50_000
        # with engine.connect() as conn:
        logger.info(f"Create temporary table VERTRAG_DELTA_{self._product}")
        metadata = sa.MetaData()
        delta_table = sa.Table(
            f"VERTRAG_DELTA_{self._product}",
            metadata,
            sa.Column("VERTRAG_ID", sa.BigInteger),
            sa.Column("VORGANGSNUMMER", sa.BigInteger),
            sa.Column("VERTRAGSAKTE_ID", sa.BigInteger),
            prefixes=["TEMPORARY"],
            schema="SESSION",
        )
        delta_table.create(self._connection_dwh)
        counter = 1
        logger.info(
            f"Needs {int(len(self.ids)/batch_size)+1} iterations for {len(self.ids)} and Batch/Chunk size {batch_size}"
        )
        try:
            for batch_start in range(0, len(self.ids), batch_size):
                logger.info(f"Start iteration {counter} with batch_size {batch_size}")
                batch_ids = self.ids.iloc[batch_start : batch_start + batch_size]
                batch_ids_str = ", ".join(map(str, batch_ids["delta_contract_id"]))
                # batch_ids = self.ids.iloc[batch_start: batch_start + batch_size]
                # logger.info(f"{batch_start}: {batch_start + batch_size} BatchIDs: {batch_ids_str}")

                context = {
                    "stichtag": self._stichtag,
                    "dwh_schema": self._dwh_db_schema,
                    "delta_ids": batch_ids_str,
                }
                query = query_template.render(**context)

                batch_selected = self._connection_dwh.execute(text(query)).fetchall()
                # logger.info(f"{batch_selected}")
                df_batch_selected = pd.DataFrame(
                    batch_selected,
                    columns=["VERTRAG_ID", "VORGANGSNUMMER", "VERTRAGSAKTE_ID"],
                )
                df_batch_selected.columns = df_batch_selected.columns.str.upper()
                # logger.info(f"{df_batch_selected.head()}")

                # with conn.begin():
                stmt = sa.insert(delta_table)
                self._connection_dwh.execute(
                    stmt, df_batch_selected.to_dict(orient="records")
                )
                # break

                counter += 1

                if self._debug:
                    break  # only use 1 iterations for debugging due to set sql limit for delta ids
        except Exception as e:
            logger.error("Error in Chunking")

        try:
            result = self._connection_dwh.execute(
                text(f"SELECT count(*) FROM SESSION.VERTRAG_DELTA_{self._product}")
            ).scalar()
            # result = self._connection_dwh.execute(
            #     text(f"SELECT * FROM SESSION.VERTRAG_DELTA_{self._product}")
            # ).fetchall()
            logger.info(f"{result} in SESSION.VERTRAG_DELTA_{self._product}")
        except Exception as e:
            logger.error("Error in Counting")
        return None

    def remove_changed_contracts(self) -> None:
        """
        This method removes all contracts that have changed from the KW
        database.

        Returns:
            None: Removes changed contracts.
        """
        batch_size = 50_000
        if len(self.ids) == 0:
            return None
        try:
            delta_info = pd.DataFrame(
                self._session_dwh.execute(
                    text(
                        f"SELECT VERTRAGSAKTE_ID FROM SESSION.VERTRAG_DELTA_{self._product}"
                    )
                ).fetchall(),
                columns=["vertragsakte_id"],
            )
            logger.info(
                f"Needs {int(len(delta_info)/batch_size)+1} iterations for {len(delta_info)} and Batch/Chunk size {batch_size}"
            )

            # ids_str = ", ".join([f"'{id}'" for id in delta_info["vertragsakte_id"]])
            query_hchca_raw = "DELETE FROM {{ db_schema }}.{{ hc_hca_table }} WHERE VE_ID IN ({{ delta_ids }})"
            query_template_hchca = jinja2.Template(query_hchca_raw)
            query_h24_raw = "DELETE FROM {{ db_schema }}.{{ h24_table }} WHERE VE_ID IN ({{ delta_ids }})"
            query_template_h24 = jinja2.Template(query_h24_raw)

            counter = 1
            for batch_start in range(0, len(delta_info), batch_size):
                logger.info(f"Start iteration {counter} with batch_size {batch_size}")
                delta_ids = delta_info.iloc[batch_start : batch_start + batch_size]
                delta_ids_str = ", ".join(map(str, delta_ids["vertragsakte_id"]))

                context_hc_hca = {
                    "delta_ids": delta_ids_str, "db_schema": self._kw_db_schema, "hc_hca_table": self._kw_hc_hca_table
                }
                context_h24 = {
                    "delta_ids": delta_ids_str, "db_schema": self._kw_db_schema, "h24_table": self._kw_h24_table
                }
                query_hchca = query_template_hchca.render(**context_hc_hca)
                query_h24 = query_template_h24.render(**context_h24)

                self._session_dakw.execute(text(query_hchca))
                self._session_dakw.execute(text(query_h24))
                self._session_dakw.commit()

                counter += 1

                if self._debug:
                    break  # only use 1 iterations for debugging due to set sql limit for delta ids

            logger.info("Changed contracts have been removed from the KW database.")
        except Exception as e:
            logger.error("Error in Deleting")

    def pull_delta(self) -> None | str:
        """
        Pulls identifying information about the changed contracts.

        Returns:
            None | str: Creates a temporary table within session_dwh or returns "No changes".
        """
        if len(self.ids) == 0:
            logger.info(f"No contracts have changed since {self._last_delta_date}.")
            return "No changes"
        ids_str = ", ".join(map(str, self.ids["delta_contract_id"]))
        sql_file = (
            Path(__file__).parent
            / f"{self._product.lower()}_delta_pull_sql/delta_{self._product.lower()}_incremental.sql"
        )
        query_template = jinja2.Template(sql_file.read_text())
        context = {
            "stichtag": self._stichtag,
            "dwh_schema": self._dwh_db_schema,
            "delta_ids": ids_str,
        }
        query = query_template.render(**context)
        self._session_dwh.execute(text(query))
        logger.info(
            f"Delta contract ids are stored in SESSION.VERTRAG_DELTA_{self._product}."
        )
        result = self._session_dwh.execute(
            text(f"SELECT * FROM SESSION.VERTRAG_DELTA_{self._product}")
        ).fetchall()
        logger.info(
            f"{len(pd.DataFrame(result, columns=['VERTRAG_ID', 'VORGANGSNUMMER', 'VERTRAGSAKTE_ID']))}"
        )
        return None


################################
class DWHDeltaPull:
    """
    Contains the logic to extract the ids of all contracts of KFZ/HUS Bestand
    that have changed since the last delta table pull.

    It also includes a method to remove all information of contracts
    that have changed from the KW database so that they can be replaced
    by newly computed client values.

    Parameters:

    session_dwh:     SQLAlchemy session connected to the DWH DB2 database.
    session_dakw:    SQLAlchemy session connected to the KW DB2 database.
    stichtag:        Date you want to draw the information of (normally today).
    last_delta_date: Date you have performed the last delta pull.
    debug:           Flag to pull all data or only a sample.
    """

    def __init__(
        self,
        product: str,
        session_dwh: sqlalchemy.orm.session.Session,
        session_dakw: sqlalchemy.orm.session.Session,
        stichtag: str,
        last_delta_date: str,
        debug: bool,
        dwh_db_schema: str,
        kw_db_schema: str,
        kw_hc_hca_table: str,
        kw_h24_table: str,
    ):
        if product in ["KFZ", "HUS"]:
            self._product = product
        else:
            raise Exception("Invalid product type - must by KFZ or HUS.")
        self._session_dwh = session_dwh
        self._session_dakw = session_dakw
        self._stichtag = stichtag
        self._last_delta_date = last_delta_date
        self._debug = debug
        self._dwh_db_schema = dwh_db_schema
        self._kw_db_schema = kw_db_schema
        self._kw_hc_hca_table = kw_hc_hca_table
        self._kw_h24_table = kw_h24_table

        sql_delta = (
            Path(__file__).parent
            / f"{self._product.lower()}_delta_pull_sql/delta_{self._product.lower()}_incremental_pure.sql"
        )
        query_template = jinja2.Template(sql_delta.read_text())
        context = {
            "last_delta_date": self._last_delta_date,
            "dwh_schema": self._dwh_db_schema,
            "debug": self._debug,
        }
        query = query_template.render(**context)
        self._session_dwh.execute(text(query))
        deltas = session_dwh.execute(
            text(f"SELECT * FROM SESSION.VERTRAG_DELTA_{self._product}_PURE")
        ).fetchall()
        self.ids = pd.DataFrame(deltas, columns=["delta_contract_id"])

    def pull_delta(self) -> None | str:
        """
        Pulls identifying information about the changed contracts.

        Returns:
            None | str: Creates a temporary table within session_dwh or returns "No changes".
        """
        if len(self.ids) == 0:
            logger.info(f"No contracts have changed since {self._last_delta_date}.")
            return "No changes"
        ids_str = ", ".join(map(str, self.ids["delta_contract_id"]))
        sql_file = (
            Path(__file__).parent
            / f"{self._product.lower()}_delta_pull_sql/delta_{self._product.lower()}_incremental.sql"
        )
        query_template = jinja2.Template(sql_file.read_text())
        context = {
            "stichtag": self._stichtag,
            "dwh_schema": self._dwh_db_schema,
            "delta_ids": ids_str,
        }
        query = query_template.render(**context)
        self._session_dwh.execute(text(query))
        logger.info(
            f"Delta contract ids are stored in SESSION.VERTRAG_DELTA_{self._product}."
        )
        return None

    def remove_changed_contracts(self) -> None:
        """
        This method removes all contracts that have changed from the KW
        database.

        Returns:
            None: Removes changed contracts.
        """
        if len(self.ids) == 0:
            return None

        delta_info = pd.DataFrame(
            self._session_dwh.execute(
                text(
                    f"SELECT VERTRAGSAKTE_ID FROM SESSION.VERTRAG_DELTA_{self._product}"
                )
            ).fetchall(),
            columns=["vertragsakte_id"],
        )

        ids_str = ", ".join([f"'{id}'" for id in delta_info["vertragsakte_id"]])
        query_hchca_raw = "DELETE FROM {{ db_schema }}.{{ hc_hca_table }} WHERE VE_ID IN ({{ delta_ids }})"
        query_template_hchca = jinja2.Template(query_hchca_raw)
        query_h24_raw = "DELETE FROM {{ db_schema }}.{{ h24_table }} WHERE VE_ID IN ({{ delta_ids }})"
        query_template_h24 = jinja2.Template(query_h24_raw)

        context_hc_hca = {
            "delta_ids": ids_str, "db_schema": self._kw_db_schema, "hc_hca_table": self._kw_hc_hca_table
        }
        context_h24 = {
            "delta_ids": ids_str, "db_schema": self._kw_db_schema, "h24_table": self._kw_h24_table
        }
        query_hchca = query_template_hchca.render(**context_hc_hca)
        query_h24 = query_template_h24.render(**context_h24)

        self._session_dakw.execute(text(query_hchca))
        self._session_dakw.execute(text(query_h24))
        self._session_dakw.commit()
        logger.info("Changed contracts have been removed from the KW database.")


class DWHDeltaPullAll:
    """
    Contains the logic to extract the ids of all contracts of KFZ/HUS Bestand.

    It also includes a method to remove all data from the KW database.

    Parameters:

    session_dwh:     SQLAlchemy session connected to the DWH DB2 database.
    session_dakw:    SQLAlchemy session connected to the KW DB2 database.
    stichtag:        Date you want to draw the information of (normally today).
    debug:           Flag to pull all data or only a sample.
    """

    def __init__(
        self,
        product: str,
        session_dwh: sqlalchemy.orm.session.Session,
        session_dakw: sqlalchemy.orm.session.Session,
        stichtag: str,
        debug: bool,
        dwh_db_schema: str,
        kw_db_schema: str,
        kw_hc_hca_table: str,
        kw_h24_table: str,
        gesellschaft: str = "HCHCA", # "H24"
    ):
        if product in ["KFZ", "HUS"]:
            self._product = product
        else:
            raise Exception("Invalid product type - must by KFZ or HUS.")
        self._session_dwh = session_dwh
        self._session_dakw = session_dakw
        self._stichtag = stichtag
        self._debug = debug
        self._dwh_db_schema = dwh_db_schema
        self._kw_db_schema = kw_db_schema
        self._gesellschaft = gesellschaft
        self._kw_hc_hca_table = kw_hc_hca_table
        self._kw_h24_table = kw_h24_table

    def pull_delta(self) -> None:
        """
        Pulls identifying information about all PKW contracts.

        Returns:
            None: Creates a temporary table within session_dwh.
        """
        sql_file = (
            Path(__file__).parent
            / f"{self._product.lower()}_delta_pull_sql/delta_{self._product.lower()}_full_{self._gesellschaft.lower()}.sql"
        )
        logger.info(f"Use SQL Delta File {sql_file}")
        query_template = jinja2.Template(sql_file.read_text())
        context = {
            "dwh_schema": self._dwh_db_schema,
            "stichtag": self._stichtag,
            "unlimited": "9999-12-31-23.59.59.999999",
            "debug": self._debug,
        }
        query = query_template.render(**context)
        self._session_dwh.execute(text(query))
        logger.info(
            f"Delta contract ids are stored in SESSION.VERTRAG_DELTA_{self._product}."
        )

    def remove_changed_contracts(self) -> None:
        """Removes all contracts from the KW database."""
        query_hchca_raw = "DELETE FROM {{ db_schema }}.{{ hc_hca_table }}"
        query_template_hchca = jinja2.Template(query_hchca_raw)
        query_h24_raw = "DELETE FROM {{ db_schema }}.{{ h24_table }}"
        query_template_h24 = jinja2.Template(query_h24_raw)

        context_hc_hca = {
            "db_schema": self._kw_db_schema, "hc_hca_table": self._kw_hc_hca_table
        }
        context_h24 = {
            "db_schema": self._kw_db_schema, "h24_table": self._kw_h24_table
        }
        query_hchca = query_template_hchca.render(**context_hc_hca)
        query_h24 = query_template_h24.render(**context_h24)

        self._session_dakw.execute(text(query_hchca))
        self._session_dakw.execute(text(query_h24))
        self._session_dakw.commit()
        logger.info("All contracts are removed from the KW database.")


# %%
