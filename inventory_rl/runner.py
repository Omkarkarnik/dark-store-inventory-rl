"""Run many independent jobs, four at a time, one CPU thread each."""
import datetime
import multiprocessing as mp
import sys
import time


def log(msg, logfile=None):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        with open(logfile, "a") as f:
            f.write(line + "\n")


def run_jobs(fn, jobs, workers, logfile=None, label="job"):
    """Run fn(job) for every job in a pool of `workers` fresh processes and log progress with an ETA."""
    if not jobs:
        log(f"Nothing to do: all {label}s already finished.", logfile)
        return []
    ctx = mp.get_context("spawn")
    t0 = time.time()
    results = []
    log(f"Starting {len(jobs)} {label}s, {workers} at a time.", logfile)
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, jobs), 1):
            results.append(res)
            elapsed = time.time() - t0
            eta = elapsed / i * (len(jobs) - i)
            log(f"{label} {i}/{len(jobs)} done: {res['name']} ({res['summary']}) | "
                f"elapsed {elapsed / 60:.1f} min, about {eta / 60:.0f} min left", logfile)
    return results


def check_python():
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required (the project was run with Python 3.12).")
