# run_all.py
from pathlib import Path
import argparse
import concurrent.futures as futures
import subprocess
import sys
import time
import psutil


def find_experiment_scripts(root: Path, pattern: str):
    """Yield scripts under experiments_* dirs matching pattern, sorted."""
    for sub in sorted(root.glob("experiments_*")):
        if sub.is_dir():
            for f in sorted(sub.glob(pattern)):
                if f.is_file():
                    yield f

def find_plot_scripts(root: Path, plots_root: str, pattern: str, recursive: bool = False):
    """Yield scripts under create_plots (or custom) matching pattern, sorted."""
    plot_dir = (root / plots_root).resolve()
    if not plot_dir.exists():
        return []
    globber = plot_dir.rglob if recursive else plot_dir.glob
    return [p for p in sorted(globber(pattern)) if p.is_file()]

def run_one(script: Path, logs_dir: Path):
    """Run a single script and stream output to both console and log file (UTF-8)."""
    import os

    log_path = logs_dir / f"{script.parent.name}__{script.stem}.log"
    start = time.time()

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }

    proc = subprocess.Popen(
        [sys.executable, "-u", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)

    proc.wait()
    return script, proc.returncode, time.time() - start, log_path

def run_phase(scripts, logs_dir, parallel, stop_on_error, phase_name):
    """Run a list of scripts with given parallelism; return list of (script, rc, secs, logp)."""
    scripts = list(scripts)
    if not scripts:
        print(f"\n[Phase: {phase_name}] No scripts found.")
        return []

    print(f"\n[Phase: {phase_name}] {len(scripts)} script(s):")
    for s in scripts:
        print("  -", s)

    results = []
    if parallel == 1:
        for s in scripts:
            print(f"\n▶ [{phase_name}] Running {s} …")
            script, rc, secs, logp = run_one(s, logs_dir)
            print(f"   ↳ exit={rc} time={secs:.1f}s log={logp}")
            results.append((script, rc, secs, logp))
            if stop_on_error and rc != 0:
                print(f"[Phase: {phase_name}] Stopping due to error.")
                break
    else:
        print(f"\n[Phase: {phase_name}] Running with parallelism={parallel}")
        with futures.ThreadPoolExecutor(max_workers=parallel) as ex:
            future_map = {ex.submit(run_one, s, logs_dir): s for s in scripts}
            for fut in futures.as_completed(future_map):
                script, rc, secs, logp = fut.result()
                print(f"▶ [{phase_name}] {script} ↳ exit={rc} time={secs:.1f}s log={logp}")
                results.append((script, rc, secs, logp))
                if stop_on_error and rc != 0:
                    print(f"[Phase: {phase_name}] One script failed; others may still complete.")
                    break
    return results

# WARNING: The following experiments require 80GB of peak memory 
def main():
    parser = argparse.ArgumentParser(description="Run experiments then plots.")
    parser.add_argument("--root", default=".", help="Project root (default: current dir)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="How many scripts to run at once per phase (default: 1)")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop immediately if a script fails in a phase")
    parser.add_argument("--run-pattern", default="run_*.py",
                        help="Glob for experiment scripts inside experiments_* (default: run_*.py)")
    parser.add_argument("--plots-root", default="create_plots",
                        help="Directory containing plotting scripts (default: create_plots)")
    parser.add_argument("--plot-pattern", default="*.py",
                        help="Glob for plotting scripts (default: *.py)")
    parser.add_argument("--plots-recursive", action="store_true",
                        help="Recurse into subfolders of the plots directory")
    args = parser.parse_args()

    # Minimum recommended memory (in GB)
    min_memory_gb = 80

    # Get available memory
    available_gb = psutil.virtual_memory().available / (1024 ** 3)

    if available_gb < min_memory_gb:
        print(f"⚠️ WARNING: Low memory detected! Only {available_gb:.2f} GB available.")
    else:
        print(f"✅ Memory check OK: {available_gb:.2f} GB available.")

    root = Path(args.root).resolve() if args.root != "." else Path(__file__).parent.resolve()

    logs_dir = root / "logs"
    (logs_dir / "experiments").mkdir(parents=True, exist_ok=True)
    (logs_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Phase 1: experiments
    # WARNING: The following experiments require 80GB of peak memory 
    run_scripts = list(find_experiment_scripts(root, args.run_pattern))
    results_run = run_phase(run_scripts, logs_dir / "experiments",
                            args.parallel, args.stop_on_error, "RUN")

    # If any failures and stop-on-error, abort before plots
    if args.stop_on_error and any(rc != 0 for _, rc, _, _ in results_run):
        print("\nAborting before plot phase due to errors in run phase.")
        return sys.exit(1)

    # Phase 2: plots
    plot_scripts = find_plot_scripts(root, args.plots_root, args.plot_pattern, args.plots_recursive)
    results_plot = run_phase(plot_scripts, logs_dir / "plots",
                             args.parallel, args.stop_on_error, "PLOT")

    # Summary
    results = results_run + results_plot
    failed = [r for r in results if r[1] != 0]
    print("\n=== Summary ===")
    for script, rc, secs, logp in results:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"{status:9} {secs:7.1f}s  {script}  [log: {logp.name}]")
    sys.exit(0 if not failed else 1)

if __name__ == "__main__":
    main()
