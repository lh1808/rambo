"""Build the flatfile: one file per partner_id % 100, one row per partner x year."""

from tqdm import tqdm

from clv_analysis.config import settings
from clv_analysis.flatfile._features_dwh import features_dwh
from clv_analysis.flatfile._features_hf3 import features_hf3, scan_hf3_agg


def build_flatfile(remainders: list[int], overwrite: bool = False) -> None:
    """Write .data/flat/<modulo>/<remainder>.parquet.

    Parameters
    ----------
    remainders : list[int]
        Build the partners with ``partner_id % 100`` in this list, one file each.
    overwrite : bool, default=False
        Build again if the file already exists; otherwise skip it.
    """
    out_dir = settings.flat_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    key = ["partner_id", "year"]
    todo = [
        r for r in remainders if overwrite or not (out_dir / f"{r}.parquet").exists()
    ]
    for remainder in tqdm(todo, desc="flatfile buckets"):
        hf3 = features_hf3(scan_hf3_agg(remainder))
        years = set(hf3["year"].to_list())
        out = (
            hf3.lazy()
            .join(features_dwh(years, [remainder]), on=key, how="left", validate="1:1")
            .sort(key)
            .collect(engine="streaming")
        )
        path = out_dir / f"{remainder}.parquet"
        out.write_parquet(path)
