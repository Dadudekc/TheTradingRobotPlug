import os
import shutil

PROJECT_ROOT = r"D:\TradingRobotPlug2"

# Cache files to remove
CACHE_DIRS = [
    "__pycache__", ".pytest_cache", ".coverage"
]

def delete_cache():
    """ Deletes unnecessary cache directories and files. """
    print("\n🗑️  Cleaning up cache and temporary files...\n")

    for root, dirs, files in os.walk(PROJECT_ROOT, topdown=False):
        for cache_dir in CACHE_DIRS:
            full_path = os.path.join(root, cache_dir)
            if os.path.exists(full_path):
                print(f"🗑️ Deleting {full_path}")
                shutil.rmtree(full_path, ignore_errors=True)

    print("\n✅ Cleanup complete!\n")


if __name__ == "__main__":
    delete_cache()
