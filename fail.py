ordner = Path(r"C:/pfad/zum/ordner")
ausgabe = ordner / "gesamt.csv"
sep = ";"

csv_files = sorted(
    f for f in ordner.glob("*.csv")
    if not f.name.lower().startswith(("gesamt", "merged"))
)
print("Lese:", [f.name for f in csv_files])

frames = [pd.read_csv(f, sep=sep, header=None, encoding="utf-8-sig") for f in csv_files]

# Sicherheitscheck: alle Dateien gleich viele Spalten?
for f, df in zip(csv_files, frames):
    if df.shape[1] != frames[0].shape[1]:
        print(f"ACHTUNG: {f.name} hat {df.shape[1]} Spalten statt {frames[0].shape[1]}")

gesamt = pd.concat(frames, ignore_index=True)
gesamt.to_csv(ausgabe, index=False, header=False, sep=sep, float_format="%.6g")
print(f"{len(csv_files)} Dateien -> {ausgabe.name}: {gesamt.shape[0]} Zeilen, {gesamt.shape[1]} Spalten")
