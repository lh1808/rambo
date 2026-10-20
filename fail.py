from pathlib import Path
import pandas as pd

# --- anpassen ---
ordner = Path(r"C:/pfad/zum/ordner")
ausgabe_name = "gesamt.csv"
sep = ";"
# -----------------

ausgabe = ordner / ausgabe_name
csv_files = sorted(f for f in ordner.glob("*.csv") if f.name != ausgabe_name)

frames = []
for f in csv_files:
    df = pd.read_csv(f, sep=sep)
    df["quelldatei"] = f.name
    frames.append(df)
    print(f"{f.name}: {len(df):>8} Zeilen")

gesamt = pd.concat(frames, ignore_index=True)
gesamt.to_csv(ausgabe, index=False, sep=sep)
print(f"\n{len(csv_files)} Dateien -> {ausgabe} ({len(gesamt)} Zeilen)")
