09:40:21 INFO [rubin.analysis] Daten geladen: X=(299988, 203), T=(299988,) (unique=[0, 1]), Y=(299988,) (unique=[0, 1]), S=(389988,)
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 132, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 2749, in run
    X, T, Y, S, eval_mask = self._load_inputs()
                            ^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 250, in _load_inputs
    assert len(S) == n, f"S-Länge ({len(S)}) ≠ X-Länge ({n}). Prüfe ob x_file und s_file zusammenpassen."
           ^^^^^^^^^^^
AssertionError: S-Länge (389988) ≠ X-Länge (299988). Prüfe ob x_file und s_file zusammenpassen.
