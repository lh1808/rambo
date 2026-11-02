multi = df[df.duplicated("PARTNER_ID", keep=False)]
feat_var = (multi.groupby("PARTNER_ID")[feature_cols].nunique() > 1).any(axis=1).mean()
print(f"Anteil Mehrfach-Partner mit variierenden Features: {feat_var:.1%}")
