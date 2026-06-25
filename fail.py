import numpy as np
from econml.score import EnsembleCateEstimator
from rubin.training import _predict_effect

pipe = ProductionPipeline(bundle_path)
members = [m for n, m in pipe.models.items() if n not in ("Ensemble", "SurrogateTree")]
ens = EnsembleCateEstimator(cate_models=members, weights=np.ones(len(members)) / len(members))
