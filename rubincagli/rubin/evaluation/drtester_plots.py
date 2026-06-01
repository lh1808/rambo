from __future__ import annotations

"""Backward-Kompatibilitäts-Shim für drtester_plots.

Dieses Modul wurde in 3 Module aufgeteilt:
  - drtester_core.py:     CustomDRTester, CustomEvaluationResults, DrTesterPlotBundle, ...
  - evaluation_plots.py:  generate_cate_distribution_plot, evaluate_cate_with_plots, ...
  - score_plots.py:       compute_qini_curve, plot_score_redistribution, ...

Bestehende Imports funktionieren weiterhin über diesen Re-Export.
"""

# Re-Exports: drtester_core
from rubin.evaluation.drtester_core import (  # noqa: F401
    save_dataframe_as_png,
    CustomEvaluationResults,
    CustomDRTester,
    DrTesterPlotBundle,
    filter_tester_for_mask,
    fit_drtester_nuisance,
)

# Re-Exports: evaluation_plots
from rubin.evaluation.evaluation_plots import (  # noqa: F401
    generate_cate_distribution_plot,
    generate_ate_barplot,
    generate_uplift_plots,
    evaluate_cate_with_plots,
)

# Re-Exports: score_plots
from rubin.evaluation.score_plots import (  # noqa: F401
    compute_qini_curve,
    plot_custom_qini_curve,
    policy_value_comparison_plots,
    plot_score_redistribution,
)
