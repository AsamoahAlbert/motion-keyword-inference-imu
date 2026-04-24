import pandas as pd
from pathlib import Path

INPUT_CSV = r'./metadata/keyword_labeled_samples_full.csv'
OUTPUT_CSV = r'./metadata/keyword_labeled_samples_imu_only.csv'

TARGET_KEYWORDS = [
    "degrees",
    "reminded",
    "tomorrow",
    "today",
    "high",
    "low",
    "expect",
    "predicted",
    "continue",
    "remind",
    "temperature",
    "united",
    "forecast",
    "closed",
    "cloudy",
    "sunny",
    "rain",
    "mile"
]

def find_imu_files(sample_folder, candidate_ids):
    for sid in candidate_ids:
        acc_sync = sample_folder / f"{sid}_sync.acc"
        gyro_sync = sample_folder / f"{sid}_sync.gyro"
        if acc_sync.exists() and gyro_sync.exists():
            return str(acc_sync), str(gyro_sync), sid

        acc = sample_folder / f"{sid}.acc"
        gyro = sample_folder / f"{sid}.gyro"
        if acc.exists() and gyro.exists():
            return str(acc), str(gyro), sid

    return None, None, None

df = pd.read_csv(INPUT_CSV)

# Normalize labels just in case
df["label"] = df["label"].astype(str).str.strip().str.lower()


data_root = Path("./data")

# Build folder lookup once
folder_lookup = {}
for p in data_root.rglob("*"):
    if p.is_dir():
        folder_lookup.setdefault(p.name, []).append(p)

rows = []
missing = 0

for _, row in df.iterrows():
    row_id = str(row["id"]).strip()
    label = str(row["label"]).strip().lower()
    transcript = str(row["transcript"]).strip()

    wav_path = Path(str(row["wav_path"]))
    wav_stem = wav_path.stem.strip()
    wav_folder = wav_path.parent
    wav_folder_name = wav_folder.name.strip()

    candidate_ids = []
    for sid in [row_id, wav_stem, wav_folder_name]:
        if sid and sid not in candidate_ids:
            candidate_ids.append(sid)

    candidate_folders = []
    if wav_folder.exists() and wav_folder.is_dir():
        candidate_folders.append(wav_folder)

    for sid in candidate_ids:
        for p in folder_lookup.get(sid, []):
            if p not in candidate_folders:
                candidate_folders.append(p)

    acc_path = None
    gyro_path = None
    matched_id = None

    for folder in candidate_folders:
        acc_path, gyro_path, matched_id = find_imu_files(folder, candidate_ids)
        if acc_path and gyro_path:
            break

    if acc_path is None or gyro_path is None:
        missing += 1
        continue

    rows.append({
        "id": row_id,
        "matched_id": matched_id,
        "label": label,
        "transcript": transcript,
        "acc_path": acc_path,
        "gyro_path": gyro_path,
    })

imu_df = pd.DataFrame(rows)
imu_df.to_csv(OUTPUT_CSV, index=False)

print("Input rows:", len(df))
print("IMU-only rows saved:", len(imu_df))
print("Missing rows:", missing)

print("\nClass counts:")
if len(imu_df) > 0:
    print(imu_df["label"].value_counts().to_string())
else:
    print("No IMU-aligned rows found.")

print(f"\nSaved: {OUTPUT_CSV}")