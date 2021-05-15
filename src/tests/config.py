from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data = file_dir.parent.parent / "data"
    src = file_dir.parent

if __name__ == "__main__":
    print(CONFIG.data)