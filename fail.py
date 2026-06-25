from econml.score import EnsembleCateEstimator

members = [m for n, m in pipe.models.items() if n not in ("Ensemble", "SurrogateTree")]
ens = EnsembleCateEstimator(cate_models=members, weights=np.ones(len(members)) / len(members))
