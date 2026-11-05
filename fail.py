df.corrwith(df["meine_variable"], numeric_only=True).drop("meine_variable").sort_values(key=abs, ascending=False).head(10)
