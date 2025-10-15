""""""




import sqlite3
from prm_attack.config import Attack
import pandas as pd




class DataReader:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_entry(self, id: str):
        row = self.cursor.execute("SELECT * FROM attacks WHERE original_id=? AND description=?", (id, "catattack iteration 11 (appended step)")).fetchone()
        if row is None:
            raise IndexError("Table index out of range")
        return row

    def generate_entries(self):
        return self.cursor.execute("SELECT * FROM attacks")

    def __del__(self):
        self.conn.close()