"""Script to download FAISS database and chunks from Google Drive."""

import os
import gdown
from pathlib import Path

# Google Drive folder ID
DRIVE_FOLDER_ID = "1dyrnFUngeg-koqz8M1G21zZMmR5D_Ig8"

# Files to download
FILES_TO_DOWNLOAD = {
    "faiss_db.zip": "https://drive.google.com/uc?id=1dyrnFUngeg-koqz8M1G21zZMmR5D_Ig8",
    "all_chunks_with_meta_all.pickle": None,  # Will be constructed
    "all_chunks_with_meta22.pickle": None,
    "all_chunks_with_meta11.pickle": None,
}

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def download_file_from_drive(file_id: str, output_path: Path):
    """Download a file from Google Drive."""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(output_path), quiet=False)
        print(f"✅ Downloaded: {output_path.name}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {output_path.name}: {e}")
        return False


def download_folder_from_drive(folder_id: str, output_dir: Path):
    """Download a folder from Google Drive."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        gdown.download_folder(url, output=str(output_dir), quiet=False, use_cookies=False)
        print(f"✅ Downloaded folder to: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Error downloading folder: {e}")
        return False


def main():
    """Download FAISS data from Google Drive."""
    print("📥 Downloading FAISS database and chunks from Google Drive...")
    print(f"Folder ID: {DRIVE_FOLDER_ID}")
    print()
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Download faiss_db.zip
    faiss_db_zip = DATA_DIR / "faiss_db.zip"
    print("Downloading faiss_db.zip...")
    if download_file_from_drive(DRIVE_FOLDER_ID, faiss_db_zip):
        # Unzip to data/faiss_db
        import zipfile
        faiss_db_dir = DATA_DIR / "faiss_db"
        faiss_db_dir.mkdir(exist_ok=True)
        print(f"Extracting {faiss_db_zip.name} to {faiss_db_dir}...")
        with zipfile.ZipFile(faiss_db_zip, 'r') as zip_ref:
            zip_ref.extractall(faiss_db_dir)
        print(f"✅ Extracted to {faiss_db_dir}")
    
    # Download pickle files
    print("\nDownloading pickle files...")
    # Note: You'll need to get direct file IDs from Google Drive
    # For now, we'll try to download from the folder
    
    print("\n✅ Download complete!")
    print(f"\nFiles should be in:")
    print(f"  - {DATA_DIR / 'faiss_db'}")
    print(f"  - {DATA_DIR}")


if __name__ == "__main__":
    main()

