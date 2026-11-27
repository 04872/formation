#!/usr/bin/env python3
"""Run the local-history frontend with its history-free J+H scoring path.

Arguments are forwarded to compare_formation_routes.py. Use --history-local-depth 1
so that candidate collection, selection and tie breaking match that frontend.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from formation.local_affine.history_local_backup import HistoryLocalBackup
from scripts.experiments.compare_formation_routes import main

_evaluate = HistoryLocalBackup.evaluate


def _raw_evaluate(self, *args, **kwargs):
    rounds = self.rounds
    try:
        self.rounds = []
        return _evaluate(self, *args, **kwargs)
    finally:
        self.rounds = rounds


if '--history-local-depth' not in sys.argv or sys.argv[sys.argv.index('--history-local-depth') + 1] != '1':
    raise SystemExit('This diagnostic requires --history-local-depth 1')
HistoryLocalBackup.evaluate = _raw_evaluate
main()
args = sys.argv
out = Path(args[args.index('--output') + 1]) if '--output' in args else Path('results/three_routes/matched_rates')
route = args[args.index('--route') + 1]
seed = args[args.index('--seed') + 1] if '--seed' in args else '0'
manifest = out / route / f'seed_{seed}' / 'configuration.json'
configuration = json.loads(manifest.read_text())
configuration['diagnostic_control'] = 'history-free J+H with local-backup candidate and tie rules'
manifest.write_text(json.dumps(configuration, indent=2))
