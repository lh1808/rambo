import numpy as np
import pandas as pd
from hf3_tariffs.ratebook import Ratebook_20250801
from quantcore.learn.utils import case_when


class ProfitAdjuster:
    def __init__(
        self,
        beitrag_col: str,
        mlevr_cost: float = 50,
        cr_neutarif_original: float = 0.93,
        use_onnx = False,
    ):
        self._beitrag_col = beitrag_col
        self._mlevr_cost = mlevr_cost
        self._cr_neutarif_original = cr_neutarif_original
        self._use_onnx = use_onnx

    def _compute_contract_duration_profit(self, df: pd.DataFrame):
        rb = Ratebook_20250801(use_onnx=self._use_onnx)
        df["ve_neutarif_beitrag"] = rb.predict(df)

        # ve_neutarif_beitrag * cr_neutarif_original -> "hypothetical" neutarif premium with CR_100
        return df[self._beitrag_col] - (
            df["ve_neutarif_beitrag"] * self._cr_neutarif_original
        )

    def _compute_adjusted_profit_pkw(self, df: pd.DataFrame) -> pd.Series:
        df["contract_duration_profit"] = self._compute_contract_duration_profit(df)

        return pd.Series(
            case_when(
                [
                    (df["ve_anzahl_mahnverfahren"] > 0)
                    | (df["ve_anzahl_levr_letzte_3_jahre"] > 0),
                    df["ve_profit"] - self._mlevr_cost,
                ],
                [
                    True,
                    np.maximum(
                        df.ve_profit,
                        df.contract_duration_profit,
                    ),
                ],
            ),
            index=df.index,
        )

    def compute_adjusted_profit(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df.ve_wkz == "112") & (~df["ve_sparte"].isnull())
        df_pkw = df[mask]
        df_non_pkw = df[~mask]

        df_pkw["ve_profit"] = self._compute_adjusted_profit_pkw(df_pkw)
        return pd.concat([df_pkw, df_non_pkw], axis=0, ignore_index=True)
