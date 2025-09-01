from prm_attack.analysis.data_collector import DataCollector
from prm_attack.analysis.data_reader import DataReader
import queue
import time
import os
from prm_attack.config import Attack

print("imports done")

li = [1, 2, 3, 4]

q = queue.Queue()
dc = DataCollector("hi.db", q, "deadbeef")
dc.start()
print("init done")

print("testing size-based flush")
for i in range(100):
    q.put(Attack(i, 0, 1, "change", repr(li), "iteration 0"))

time.sleep(1)

print("testing time-based flush")
for i in range(10):
    q.put(Attack(i + 1000, 0, 1, "change (time based)", repr(li), "iteration 1"))
time.sleep(31)

dc.stop()

dr = DataReader("hi.db")
for row in dr.generate_entries():
    print(str(row))

print(dr.get_entry(58))

os.remove("hi.db")