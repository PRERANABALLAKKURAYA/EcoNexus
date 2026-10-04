"""Minimal runner (used when pytest is unavailable): python scripts/run_tests.py"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib  # noqa: E402

mods = [importlib.import_module(m) for m in ("tests.test_core", "tests.test_dashboard")]

fails = 0
for mod, name in [(m, n) for m in mods for n in sorted(dir(m)) if n.startswith("test_")]:
    t0 = time.time()
    try:
        getattr(mod, name)()
        print(f"PASS {name} ({time.time() - t0:.1f}s)")
    except Exception:
        fails += 1
        print(f"FAIL {name}")
        traceback.print_exc()
print(f"\n{fails} failed")
sys.exit(1 if fails else 0)
