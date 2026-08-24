#!/usr/bin/env python3
"""Usage: python submit.py COUNTRY VARIANT START_IDX END_IDX"""
import subprocess, sys, os, time

COUNTRY   = sys.argv[1]
VARIANT   = sys.argv[2]
START_IDX = int(sys.argv[3])
END_IDX   = int(sys.argv[4])

script_dir = os.path.dirname(os.path.abspath(__file__))
MAX_RETRY, RETRY_WAIT = 30, 60

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting submission for {COUNTRY} {VARIANT}, chunk {START_IDX}-{END_IDX}")

for i in range(START_IDX, END_IDX + 1):
    sh_path = os.path.join(script_dir, f"run_{COUNTRY}_{VARIANT}_{i}.sh")
    if not os.path.exists(sh_path):
        print(f"[SKIP] Missing {sh_path}")
        continue

    for attempt in range(1, MAX_RETRY + 1):
        ret = subprocess.run(["sbatch", sh_path], capture_output=True, text=True)
        if ret.returncode == 0:
            print(f"[{i}] {ret.stdout.strip()}")
            break
        else:
            err = ret.stderr.strip()
            print(f"[{i}] Attempt {attempt}/{MAX_RETRY} failed: {err}")
            if attempt == MAX_RETRY:
                print(f"[{i}] Maximum retries exceeded; stopping.")
                sys.exit(1)
            time.sleep(RETRY_WAIT)

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed {COUNTRY} {VARIANT}, chunk {START_IDX}-{END_IDX}")
