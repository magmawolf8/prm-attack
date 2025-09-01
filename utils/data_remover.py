import argparse
import sqlite3
import sys

def delete_by_commit(db_path, commit_hash):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM attacks WHERE commit_hash = ?", (commit_hash,))
    conn.commit()
    
    print(f"Deleted {cursor.rowcount} rows with commit_hash='{commit_hash}'")
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove rows from attacks table by commit hash.")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file")
    parser.add_argument("--commit-hash", required=True, help="Commit hash to delete rows for")
    args = parser.parse_args()

    try:
        delete_by_commit(args.db_path, args.commit_hash)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)