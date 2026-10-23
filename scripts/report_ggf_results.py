"""Compare completed GGF metrics with the immutable completed baseline report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from train import load_config
from ggf_score_matching import ggf_identity

BASELINE = REPO / 'data/ggf/baselines/non-fokker-planck'
BASELINE_RECEIPT = '577fc4b1b5e389f36be149c74684efb4c4a646c4917b4a74d5aaa00214fab1bc'
METHODS = ('diffusion', 'two_for_one', 'mixture')
NAMES = {'diffusion': 'Diffusion', 'two_for_one': 'Two-for-One', 'mixture': 'Mixture', 'ggf': 'GGF'}
PRIMARY = ('iid_jsd', 'simulation_jsd', 'iid_pmf', 'simulation_pmf')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, value):
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def format_cell(value):
    if value['mean'] is None:
        return 'N/A'
    return f"{value['mean']:.6g} ± {value['std_across_peptides']:.6g}"


def report(config):
    folder = Path(config['data']['root']) / 'results'
    summary_path = folder / 'summary.json'
    summary = json.loads(summary_path.read_bytes())
    identity = ggf_identity(config)
    if summary.get('ggf') != identity or set(summary['methods']) != {'ggf'}:
        raise ValueError('Expected the completed, source-bound GGF-only metric report')
    if summary['settings'] != config['evaluation']:
        raise ValueError('GGF summary evaluation settings differ')
    with (folder / 'per_peptide.csv').open() as stream:
        peptides = list(csv.DictReader(stream))
    expected = config['evaluation']['peptides']
    if (len(peptides) != len(expected) or {row['peptide'] for row in peptides} != set(expected)
            or any(row['method'] != 'ggf' for row in peptides)):
        raise ValueError('GGF per-peptide report has incomplete or duplicate coverage')
    receipt_path = BASELINE / 'report_receipt.json'
    if digest(receipt_path) != BASELINE_RECEIPT:
        raise ValueError('Pinned baseline report receipt changed')
    receipt = json.loads(receipt_path.read_bytes())
    baseline_path = BASELINE / 'summary.json'
    if digest(baseline_path) != receipt['output_sha256']['summary.json']:
        raise ValueError('Pinned baseline summary changed')
    baseline = json.loads(baseline_path.read_bytes())
    # Distribution metrics are compared on the same test protocol; the GGF IID
    # process is intentionally VE rather than the baselines' VP process.
    for name, value in config['evaluation'].items():
        if name != 'methods' and baseline['settings'][name] != value:
            raise ValueError(f'Baseline comparison protocol differs: {name}')
    comparisons = []
    for method in METHODS:
        for definition in ('paper', 'released_code'):
            for metric in PRIMARY:
                key = metric if definition == 'paper' else metric.replace('_jsd', '_code_jsd').replace('_pmf', '_code_pmf')
                actual, reference = summary['methods']['ggf'][key], baseline['methods'][method][key]
                mean, other = actual['mean'], reference['mean']
                difference = None if mean is None else mean - other
                comparisons.append({'baseline_method': method, 'metric_definition': definition, 'metric': metric,
                                    'ggf_mean': mean, 'ggf_std': actual['std_across_peptides'],
                                    'baseline_mean': other, 'baseline_std': reference['std_across_peptides'],
                                    'difference': difference,
                                    'percentage_change': None if difference is None or other == 0 else 100 * difference / other,
                                    'ggf_evaluated_peptides': actual['evaluated_peptides'],
                                    'expected_peptides': len(expected)})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
    atomic_text(folder / 'baseline_comparison.csv', buffer.getvalue())
    lines = ['# GGF compared with the completed dipeptide baselines', '',
             f"Generated: {datetime.now(timezone.utc).isoformat()}", '',
             'One peptide-conditioned, noise-unconditional energy model; paired singleton GGF loss.',
             f"Bandwidth h = sigma_min = {identity['sigma_min']:.17g}; sigma_max = {identity['sigma_max']:g}.",
             'IID generation uses the agreed experimental h²/sigma(t)² VE adapter. Simulation uses the raw energy gradient at minimum noise.',
             '', 'Values are mean ± population standard deviation across the configured peptides; lower is better.',
             'N/A indicates incomplete valid-sample coverage; no partial-panel average is substituted.',
             'The baseline rows are our completed measurements. There is no published GGF reference row.', '']
    keys = ('iid_jsd', 'iid_pmf', 'simulation_jsd', 'simulation_pmf')
    for definition in ('released_code', 'paper'):
        lines.extend([f'## {"Released-code definitions" if definition == "released_code" else "Alternative local paper-interpretation definitions"}', '',
                      '| Method | IID JSD | IID PMF error | Simulation JSD | Simulation PMF error |',
                      '|---|---:|---:|---:|---:|'])
        for method in METHODS + ('ggf',):
            values = summary['methods']['ggf'] if method == 'ggf' else baseline['methods'][method]
            fields = [key if definition == 'paper' else key.replace('_jsd', '_code_jsd').replace('_pmf', '_code_pmf') for key in keys]
            lines.append('| ' + NAMES[method] + ' | ' + ' | '.join(format_cell(values[key]) for key in fields) + ' |')
        lines.append('')
    lines.extend(['## Artifacts', '', '- [Per-peptide metrics](per_peptide.csv)',
                  '- [Full GGF summary](summary.json)', '- [Differences from each baseline](baseline_comparison.csv)',
                  f'- [Immutable baseline source]({baseline_path})', ''])
    atomic_text(folder / 'GGF_RESULTS.md', '\n'.join(lines))
    metadata = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'ggf': identity,
                'ggf_summary_sha256': digest(summary_path), 'baseline_receipt_sha256': BASELINE_RECEIPT,
                'baseline_summary_sha256': digest(baseline_path), 'comparison_rows': len(comparisons),
                'output_sha256': {name: digest(folder / name) for name in ('baseline_comparison.csv', 'GGF_RESULTS.md')}}
    atomic_text(folder / 'baseline_comparison_metadata.json', json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPO / 'configs/dipeptides_ggf.yaml'))
    args = parser.parse_args()
    print(json.dumps(report(load_config(args.config)), indent=2))
