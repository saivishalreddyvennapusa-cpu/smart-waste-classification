
import zipfile
from pathlib import Path

zip_path = Path("dataset2.zip")
output_dir = Path("dataset")

if not zip_path.exists():
    raise FileNotFoundError("dataset2.zip not found in the project folder.")

print("Checking dataset ZIP...")
with zipfile.ZipFile(zip_path, "r") as z:
    members = [name for name in z.namelist() if name.startswith("wastes/")]
    print(f"Found {len(members)} files/folders.")
    print("Extracting dataset. Please wait...")
    z.extractall(output_dir, members=members)

print("Extraction completed successfully!")
print(f"Dataset path: {output_dir.resolve()}")
