import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# path ke data mentah
base_dir = Path(__file__).parent.parent
data_dir = base_dir / "dataset" / "play_tennis_dataset.csv"

df = pd.read_csv(data_dir)

# buang kolom Day
df = df.drop(columns=["Day"])

# Label encoding setiap kolom
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# path folder untuk data bersih
clean_dir = base_dir / "clean_data"
clean_dir.mkdir(exist_ok=True)  # pastikan folder ada
clean_path = clean_dir / "clean_play_tennis_dataset.csv"

df.to_csv(clean_path, index=False, encoding="utf-8")

print("Preprocessing Berhasil Loh Yaa")
