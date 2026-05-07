Analyse fehlgeschlagen: Fehlgeschlagen (Exit 1)

Details:
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 132, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 126, in main
    cfg = load_config(args.config)
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/settings.py", line 600, in load_config
    return AnalysisConfig.model_validate(raw)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/pydantic/main.py", line 732, in model_validate
    return cls.__pydantic_validator__.validate_python(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for AnalysisConfig
causal_forest.econml_tune_params
  Extra inputs are not permitted [type=extra_forbidden, input_value='auto', input_type=str]
    For further information visit https://errors.pydantic.dev/2.13/v/extra_forbidden
