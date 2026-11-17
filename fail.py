Kundenwert Delta Load Problem

da_kundenwert_airflow_blocks/pipeline_components/dwh_load/_delta_component


da_kundenwert/data/dwh_delta_pull

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


da_kundenwert/data/hus_delta_pull_sql

CREATE TEMPORARY TABLE SESSION.VERTRAG_DELTA_HUS AS (
    SELECT
        v.VERTRAG_ID,
        v.VORGANGSNUMMER,
        v.VERTRAGSAKTE_ID
    FROM {{ dwh_schema }}.VW_MF_AK_HUS_VERTRAG as v
    WHERE
        v.available_to_date        >= TIMESTAMP('{{ unlimited }}')
        AND v.effective_from_date	<=	TIMESTAMP('{{ stichtag }}')
        AND v.effective_to_date	>	TIMESTAMP('{{ stichtag }}')
        AND v.valid_from_date	    <=	TIMESTAMP('{{ stichtag }}')
        AND v.valid_to_date	    >	TIMESTAMP('{{ stichtag }}')
        AND v.anz_vertraege     > 0
        AND v.dim_gesellschaft_id IN ('23', '24') --- HC/HCA ---
    {% if debug %}LIMIT 1000{% endif %}
)


delta_hus_incremental_pure.sql

CREATE TEMPORARY TABLE SESSION.VERTRAG_DELTA_HUS_PURE AS(
     SELECT DISTINCT VERTRAG_ID
          FROM {{ dwh_schema }}.VW_MF_AK_DA_DELTA_HUS as delta_hus
          WHERE delta_hus.operating_time >= TIMESTAMP('{{ last_delta_date }}')
          {% if debug %}LIMIT 5000{% endif %}
)
