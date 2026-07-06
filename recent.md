reouvenbenchetrit@macbook-air2 neurotrade % pwd source /Users/reouvenb
enchetrit/Desktop/neurotrade/venv/bin/activate
pwd: too many arguments
reouvenbenchetrit@macbook-air2 neurotrade % 
reouvenbenchetrit@macbook-air2 neurotrade % pwd
/Users/reouvenbenchetrit/Desktop/neurotrade
reouvenbenchetrit@macbook-air2 neurotrade % uv sync
Resolved 96 packages in 8ms



^[[A^[[A^[[A^[[A



^C
reouvenbenchetrit@macbook-air2 neurotrade % uv sync
uv run neurotrade backtest --config configs/btc_1min.yaml
Resolved 96 packages in 20ms




      Built neurotrade @ file:///Users/reouvenbenchetrit/Desktop/neurotrade
Prepared 1 package in 378ms
Uninstalled 1 package in 1.09s
Installed 1 package in 3ms
 - neurotrade==0.1.0 (from file:///Users/reouvenbenchetrit/Desktop/pro-perso/projets/neurotrade)
 + neurotrade==0.1.0 (from file:///Users/reouvenbenchetrit/Desktop/neurotrade)
^CTraceback (most recent call last):
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/bin/neurotrade", line 4, in <module>
    from neurotrade.cli.main import app
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/cli/main.py", line 11, in <module>
    import typer
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/lib/python3.12/site-packages/typer/__init__.py", line 22, in <module>
    from .main import Typer as Typer
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/lib/python3.12/site-packages/typer/main.py", line 18, in <module>
    from annotated_doc import Doc
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1132, in get_code
  File "<frozen importlib._bootstrap_external>", line 1191, in get_data
KeyboardInterrupt
reouvenbenchetrit@macbook-air2 neurotrade % 
reouvenbenchetrit@macbook-air2 neurotrade % cd ~/Desktop/neurotrade
reouvenbenchetrit@macbook-air2 neurotrade % rm -rf .venv
rm -rf venv
reouvenbenchetrit@macbook-air2 neurotrade % uv sync --reinstall
Using CPython 3.12.13
Creating virtual environment at: .venv
Resolved 96 packages in 24ms
      Built neurotrade @ file:///Users/reouvenbenchetrit/Desktop/neurotrade
Prepared 74 packages in 330ms
Installed 74 packages in 684ms
 + aiodns==4.0.4
 + aiohappyeyeballs==2.6.2
 + aiohttp==3.14.0
 + aiosignal==1.4.0
 + annotated-doc==0.0.4
 + annotated-types==0.7.0
 + ast-serialize==0.5.0
 + attrs==26.1.0
 + black==26.5.1
 + ccxt==4.5.56
 + certifi==2026.5.20
 + cffi==2.0.0
 + charset-normalizer==3.4.7
 + click==8.4.1
 + coincurve==21.0.0
 + contourpy==1.3.3
 + coverage==7.14.1
 + cryptography==48.0.0
 + cycler==0.12.1
 + filelock==3.29.1
 + fonttools==4.63.0
 + frozenlist==1.8.0
 + fsspec==2026.4.0
 + idna==3.18
 + iniconfig==2.3.0
 + jinja2==3.1.6
 + kiwisolver==1.5.0
 + librt==0.11.0
 + markdown-it-py==4.2.0
 + markupsafe==3.0.3
 + matplotlib==3.10.9
 + mdurl==0.1.2
 + mpmath==1.3.0
 + multidict==6.7.1
 + mypy==2.1.0
 + mypy-extensions==1.1.0
 + networkx==3.6.1
 + neurotrade==0.1.0 (from file:///Users/reouvenbenchetrit/Desktop/neurotrade)
 + numpy==2.4.6
 + packaging==26.2
 + pandas==3.0.3
 + pandas-stubs==3.0.3.260530
 + pathspec==1.1.1
 + pillow==12.2.0
 + platformdirs==4.10.0
 + pluggy==1.6.0
 + propcache==0.5.2
 + pyarrow==24.0.0
 + pycares==5.0.1
 + pycparser==3.0
 + pydantic==2.13.4
 + pydantic-core==2.46.4
 + pygments==2.20.0
 + pyparsing==3.3.2
 + pytest==9.0.3
 + pytest-cov==7.1.0
 + python-dateutil==2.9.0.post0
 + pytokens==0.4.1
 + pyyaml==6.0.3
 + requests==2.34.2
 + rich==15.0.0
 + ruff==0.15.16
 + scipy==1.17.1
 + setuptools==81.0.0
 + shellingham==1.5.4
 + six==1.17.0
 + sympy==1.14.0
 + torch==2.12.0
 + typer==0.26.7
 + types-pyyaml==6.0.12.20260518
 + typing-extensions==4.15.0
 + typing-inspection==0.4.2
 + urllib3==2.7.0
 + yarl==1.24.2
reouvenbenchetrit@macbook-air2 neurotrade % uv pip show neurotrade
Name: neurotrade
Version: 0.1.0
Location: /Users/reouvenbenchetrit/Desktop/neurotrade/.venv/lib/python3.12/site-packages
Editable project location: /Users/reouvenbenchetrit/Desktop/neurotrade
Requires: ccxt, matplotlib, numpy, pandas, pyarrow, pydantic, pyyaml, rich, scipy, torch, typer
Required-by:
reouvenbenchetrit@macbook-air2 neurotrade % uv run python -c "import neurotrade; print(neurotrade.__file__)"
/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/__init__.py
reouvenbenchetrit@macbook-air2 neurotrade % uv pip show neurotrade
ls -la
cat pyproject.toml
Name: neurotrade
Version: 0.1.0
Location: /Users/reouvenbenchetrit/Desktop/neurotrade/.venv/lib/python3.12/site-packages
Editable project location: /Users/reouvenbenchetrit/Desktop/neurotrade
Requires: ccxt, matplotlib, numpy, pandas, pyarrow, pydantic, pyyaml, rich, scipy, torch, typer
Required-by:
total 1424
drwxr-xr-x@ 22 reouvenbenchetrit  staff     704 Jul  6 13:18 .
drwx------@ 20 reouvenbenchetrit  staff     640 Jul  6 13:16 ..
-rw-r--r--@  1 reouvenbenchetrit  staff    8196 Jul  6 13:06 .DS_Store
drwxr-xr-x  13 reouvenbenchetrit  staff     416 Jul  6 13:15 .git
-rw-r--r--@  1 reouvenbenchetrit  staff     426 Jun  7 10:42 .gitignore
drwxr-xr-x@  5 reouvenbenchetrit  staff     160 Jul  6 13:05 .mypy_cache
drwxr-xr-x@  6 reouvenbenchetrit  staff     192 Jul  6 13:05 .pytest_cache
drwxr-xr-x@  5 reouvenbenchetrit  staff     160 Jul  6 13:05 .ruff_cache
drwxr-xr-x@  9 reouvenbenchetrit  staff     288 Jul  6 13:18 .venv
-rw-r--r--@  1 reouvenbenchetrit  staff    7766 Jun 16 19:07 01_NEUROTRADE.md
-rw-r--r--@  1 reouvenbenchetrit  staff    2297 Jun  7 10:43 CLAUDE.md
-rw-r--r--@  1 reouvenbenchetrit  staff     499 Jun  7 10:42 Makefile
-rw-r--r--@  1 reouvenbenchetrit  staff    2553 Jun  7 11:43 PROGRESS.md
-rw-r--r--@  1 reouvenbenchetrit  staff    3283 Jun 11 15:09 README.md
drwxr-xr-x@  4 reouvenbenchetrit  staff     128 Jul  6 13:05 configs
drwxr-xr-x@  3 reouvenbenchetrit  staff      96 Jul  6 13:05 data
drwxr-xr-x@  3 reouvenbenchetrit  staff      96 Jul  6 13:05 docs
-rw-r--r--@  1 reouvenbenchetrit  staff    2822 Jun  7 11:22 pyproject.toml
drwxr-xr-x@  3 reouvenbenchetrit  staff      96 Jul  6 13:05 reports
drwxr-xr-x@  3 reouvenbenchetrit  staff      96 Jul  6 13:16 src
drwxr-xr-x@ 12 reouvenbenchetrit  staff     384 Jul  6 13:05 tests
-rw-r--r--@  1 reouvenbenchetrit  staff  683776 Jun  7 11:00 uv.lock
[project]
name = "neurotrade"
version = "0.1.0"
description = "Système de recherche quantitative reproductible — réseau de neurones maison, backtest walk-forward honnête."
requires-python = ">=3.11"
dependencies = [
    "ccxt>=4.3.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "pyarrow>=15.0.0",
    "pydantic>=2.6.0",
    "pyyaml>=6.0.0",
    "typer>=0.12.0",
    "rich>=13.7.0",
    "matplotlib>=3.8.0",
    "torch>=2.2.0",
    "scipy>=1.12.0",
]

[project.scripts]
neurotrade = "neurotrade.cli.main:app"

[dependency-groups]
dev = [
    "ruff>=0.4.0",
    "black>=24.4.0",
    "mypy>=1.9.0",
    "pytest>=8.1.0",
    "pytest-cov>=5.0.0",
    "pandas-stubs>=2.2.0",
    "types-PyYAML>=6.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/neurotrade"]

# ── Ruff ──────────────────────────────────────────────────────────────────────
[tool.ruff]
src = ["src"]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "B", "UP", "RUF"]
ignore = [
    "B008",   # function calls in default arguments (typer pattern)
    "N803",   # convention ML : X majuscule pour matrice de features (argument)
    "N806",   # convention ML : X majuscule pour matrice de features (variable locale)
    "N812",   # torch.nn.functional importé comme F (convention universelle PyTorch)
    "RUF002", # caractères grecs (μ, σ) dans docstrings — voulus pour la notation mathématique
    "RUF003", # caractères grecs dans commentaires — idem
]

# ── Black ─────────────────────────────────────────────────────────────────────
[tool.black]
line-length = 100
target-version = ["py311"]

# ── Mypy ──────────────────────────────────────────────────────────────────────
[tool.mypy]
python_version = "3.11"
strict = true
files = ["src/"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
# Bibliothèques tierces sans stubs complets
module = ["ccxt", "ccxt.*", "matplotlib.*", "matplotlib.figure", "matplotlib.axes", "scipy.*"]
ignore_missing_imports = true

# ── Pytest ────────────────────────────────────────────────────────────────────
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
pythonpath = ["src"]
reouvenbenchetrit@macbook-air2 neurotrade % uv run neurotrade backtest --config configs/btc_1min.yaml

^C^C^C^C^C^C^C^C^C^C^C


Traceback (most recent call last):
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/bin/neurotrade", line 4, in <module>
    from neurotrade.cli.main import app
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/cli/main.py", line 14, in <module>
    from neurotrade.experiments import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/__init__.py", line 3, in <module>
    from .runner import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/runner.py", line 14, in <module>
    import torch
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/lib/python3.12/site-packages/torch/__init__.py", line 444, in <module>
    from torch._C import *  # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 463, in _lock_unlock_module
KeyboardInterrupt
reouvenbenchetrit@macbook-air2 neurotrade % 
reouvenbenchetrit@macbook-air2 neurotrade % 
reouvenbenchetrit@macbook-air2 neurotrade % 
reouvenbenchetrit@macbook-air2 neurotrade % 
reouvenbenchetrit@macbook-air2 neurotrade % >....                                                                                   
## Qualité

```bash
make check   # lint (ruff) + types (mypy strict) + tests (pytest)
make fmt     # formatage automatique (black + ruff --fix)
```

## Roadmap

- **Phase 0** ✅ Échafaudage, outillage qualité
- **Phase 1** 🔄 Recherche offline (étapes 1–9)
  1. `data/` — fetch OHLCV + cache
  2. `features/` — pipeline + anti-fuite scaler
  3. `labeling/` — triple-barrier
  4. `models/` — MLP, 1D-CNN, LSTM
  5. `backtest/` — walk-forward + purge/embargo
  6. `backtest/` — simulateur d'exécution
  7. `backtest/` — risque + kill-switch
  8. `evaluation/` — métriques + rapports
  9. `experiments/` — robustesse cross-asset
- **Phase 2** ⬜ Paper trading (uniquement après validation Phase 1)

Phase 1 complète. Voici le résumé de tout ce qui a été implémenté :

Phase 1 — tous les 9 steps sont terminés et commités :

Étape   Commit  Tests   Highlights
4 — models/     a70bcca 7       MLP/CNN1D/LSTM + Trainer early stopping (delta=1e-4), mypy strict
5 — walk_forward        072c394 15      Purge gap exact + embargo gap exact, no-overlap sur 3000 bars
6 — execution   072c394 7       PnL vérifié analytiquement (0 fees, slippage directionnel, reversal)
7 — risk        072c394 14      Sizing, stop journalier, kill-switch z-score, std=0 safe
8 — evaluation  019b057 21      Sharpe, DSR (Bailey 2014), PBO bootstrap, rapport matplotlib
9 — runner      019b057 —       Pipeline complet data→eval, neurotrade backtest opérationnel
État global : 116 tests (95 rapides + 21 évaluation), mypy --strict + ruff propres sur 35 fichiers. Phase 2 (live/paper) attend ta demande explicite.

bquote> 
reouvenbenchetrit@macbook-air2 neurotrade % 
reouvenbenchetrit@macbook-air2 neurotrade % cd ~/Desktop/neurotrade
uv run neurotrade --help



^CTraceback (most recent call last):
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/bin/neurotrade", line 4, in <module>
    from neurotrade.cli.main import app
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/cli/main.py", line 14, in <module>
    from neurotrade.experiments import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/__init__.py", line 3, in <module>
    from .runner import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/runner.py", line 20, in <module>
    from neurotrade.data.cache import load_or_fetch
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1091, in get_code
  File "<frozen importlib._bootstrap_external>", line 1191, in get_data
KeyboardInterrupt
reouvenbenchetrit@macbook-air2 neurotrade % pwd
/Users/reouvenbenchetrit/Desktop/neurotrade
reouvenbenchetrit@macbook-air2 neurotrade % uv run neurotrade --help


^CTraceback (most recent call last):
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/bin/neurotrade", line 4, in <module>
    from neurotrade.cli.main import app
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/cli/main.py", line 14, in <module>
    from neurotrade.experiments import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/__init__.py", line 3, in <module>
    from .runner import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/runner.py", line 20, in <module>
    from neurotrade.data.cache import load_or_fetch
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/data/__init__.py", line 4, in <module>
    from .fetcher import fetch_ohlcv
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/data/fetcher.py", line 12, in <module>
    import ccxt
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/lib/python3.12/site-packages/ccxt/__init__.py", line 130, in <module>
    from ccxt.coinbaseinternational import coinbaseinternational          # noqa: F401
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
reouvenbenchetrit@macbook-air2 neurotrade % uv run neurotrade backtest --config configs/btc_1min.yaml

^CTraceback (most recent call last):
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/bin/neurotrade", line 4, in <module>
    from neurotrade.cli.main import app
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/cli/main.py", line 14, in <module>
    from neurotrade.experiments import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/__init__.py", line 3, in <module>
    from .runner import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/runner.py", line 20, in <module>
    from neurotrade.data.cache import load_or_fetch
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/data/__init__.py", line 5, in <module>
    from .resample import resample_ohlcv
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1091, in get_code
  File "<frozen importlib._bootstrap_external>", line 1191, in get_data
KeyboardInterrupt
reouvenbenchetrit@macbook-air2 neurotrade % 
reouvenbenchetrit@macbook-air2 neurotrade % uv run python -c "import neurotrade.cli.main as m; print(dir(m))"
^CTraceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/cli/main.py", line 14, in <module>
    from neurotrade.experiments import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/__init__.py", line 3, in <module>
    from .runner import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/runner.py", line 22, in <module>
    from neurotrade.evaluation.metrics import BacktestMetrics, compute_metrics
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1091, in get_code
  File "<frozen importlib._bootstrap_external>", line 1191, in get_data
KeyboardInterrupt
reouvenbenchetrit@macbook-air2 neurotrade % which uv
/Library/Frameworks/Python.framework/Versions/3.14/bin/uv
reouvenbenchetrit@macbook-air2 neurotrade % uv sync
Resolved 96 packages in 16ms
Checked 74 packages in 10ms
reouvenbenchetrit@macbook-air2 neurotrade % uv run neurotrade --help
^CTraceback (most recent call last):
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/bin/neurotrade", line 4, in <module>
    from neurotrade.cli.main import app
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/cli/main.py", line 14, in <module>
    from neurotrade.experiments import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/__init__.py", line 3, in <module>
    from .runner import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/runner.py", line 22, in <module>
    from neurotrade.evaluation.metrics import BacktestMetrics, compute_metrics
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/evaluation/__init__.py", line 4, in <module>
    from .report import generate_report
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/evaluation/report.py", line 14, in <module>
    import matplotlib
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/lib/python3.12/site-packages/matplotlib/__init__.py", line 161, in <module>
    from . import _api, _version, cbook, _docstring, rcsetup
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/.venv/lib/python3.12/site-packages/matplotlib/cbook.py", line 32, in <module>
    from matplotlib import _api, _c_internal_utils
KeyboardInterrupt
reouvenbenchetrit@macbook-air2 neurotrade % uv --version
uv 0.11.15 (3cffe97c2 2026-05-18 aarch64-apple-darwin)
reouvenbenchetrit@macbook-air2 neurotrade % cd ~/Desktop/neurotrade
reouvenbenchetrit@macbook-air2 neurotrade % uv sync
Resolved 96 packages in 4ms
Checked 74 packages in 9ms
reouvenbenchetrit@macbook-air2 neurotrade % uv run python -c "print('OK PYTHON')"
OK PYTHON
reouvenbenchetrit@macbook-air2 neurotrade % uv run python -c "import neurotrade; print('OK IMPORT')"
OK IMPORT
reouvenbenchetrit@macbook-air2 neurotrade % uv run python -c "from neurotrade.cli.main import app; print('CLI OK')"



^CTraceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/cli/main.py", line 14, in <module>
    from neurotrade.experiments import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/__init__.py", line 3, in <module>
    from .runner import ExperimentRunner
  File "/Users/reouvenbenchetrit/Desktop/neurotrade/src/neurotrade/experiments/runner.py", line 24, in <module>
    from neurotrade.features.momentum import momentum_zscore
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1091, in get_code
  File "<frozen importlib._bootstrap_external>", line 1191, in get_data
KeyboardInterrupt
reouvenbenchetrit@macbook-air2 neurotrade % 

