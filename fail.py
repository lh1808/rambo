n = int(len(df) * 0.6)

oben = df.iloc[:n].sample(frac=1, random_state=42)
unten = df.iloc[n:].sort_values("SPALTE", ascending=True)

df_neu = pd.concat([oben, unten], ignore_index=True)
