
SELECT DISTINCT
        VERTRAG_ID,
        VORGANGSNUMMER,
        VERTRAGSAKTE_ID
    FROM (
        SELECT v.VERTRAG_ID, v.VORGANGSNUMMER, v.VERTRAGSAKTE_ID
        FROM {{ dwh_schema }}.VW_MF_AK_HUS_VERTRAG as v
        WHERE
            v.VERTRAG_ID IN ({{ delta_ids }})
            AND v.effective_from_date	 <=	TIMESTAMP('{{ stichtag }}')
            AND v.effective_to_date      >	TIMESTAMP('{{ stichtag }}')
            AND v.valid_from_date	     <=	TIMESTAMP('{{ stichtag }}')
            AND v.valid_to_date	         >	TIMESTAMP('{{ stichtag }}')
            AND v.anz_vertraege       >  0
            AND v.dim_gesellschaft_id IN ('23', '24', '133', '134')
    )

