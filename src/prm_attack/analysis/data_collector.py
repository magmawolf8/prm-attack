"""Collects data about adversarial methods to transform question-answer trajectories.

Let's write a specialized one for CatAttack experiments first.
It should collect some simple data: the original dataset, and a relational database
of attacks. Then for visualizations we can go through it and create graphs etc.

Suppose we started with a table of questions and answers (just from the processbench dataset).
Then, we link attacks on particular questions and answers using a modified trajectory, the ID of the question-answer pair, 
the git commit hash of the current attack, and a single-word description of the attack."""




import sqlite3
import threading
import queue
import time
import pandas as pd




class DataCollector(threading.Thread):
    def __init__(self, db_path: str, q, commit_hash: str):
        super().__init__(daemon=True)
        self.db_path = db_path
        
        self.q = q

        self.commit_hash = commit_hash
        
        self.stop_flag = threading.Event()

        self.MAX_BUF_LEN = 30
        self.BUF_TIMEOUT = 10

    def run(self):
        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()

        cursor.execute("""
CREATE TABLE IF NOT EXISTS attacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_id INTEGER NOT NULL,
    mod_idx INTEGER NOT NULL,
    mod_len INTEGER NOT NULL,
    modification TEXT NOT NULL,                       
    mod_reward TEXT NOT NULL,
    description TEXT NOT NULL,
    commit_hash TEXT NOT NULL
)
                       """)
        conn.commit()

        try:
            buf = list()
            last_flush = time.time()

            while not (self.stop_flag.is_set() and self.q.empty()):
                try:
                    entry = self.q.get(timeout=0.1)
                    buf.append(entry)
                except queue.Empty:
                    pass

                should_flush = (
                    len(buf) >= self.MAX_BUF_LEN or
                    (buf and (time.time() - last_flush) >= self.BUF_TIMEOUT)
                )
                if should_flush:
                    self._flush(conn, cursor, buf)
                    buf.clear()
                    last_flush = time.time()

            if buf:
                self._flush(conn, cursor, buf)
        finally:
            conn.close()
        
    def _flush(self, conn, cursor, buf):
        rows = [
            (*entry.get_as_tuple(), self.commit_hash)
            for entry in buf
        ]
        cursor.executemany(
            "INSERT INTO attacks (original_id, mod_idx, mod_len, modification, mod_reward, description, commit_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows
        )

        conn.commit()
 
    def stop(self):
        self.stop_flag.set()