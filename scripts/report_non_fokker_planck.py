#!/home/anirudh/.local/venvs/ggf_md/bin/python
"""CPU-only reporting of independently audited, completed non-FP phases.

An explicit fresh --output directory under NFS reports/non-fokker-planck is
required. Future completion receipts require operator-supplied exact SHA256:
  --accepted-receipt iid:diffusion /path/to/full-completion.json SHA256
This does not sample, load checkpoint tensors, acquire production locks, or
read incomplete phase payloads. Existing scientific metric functions are used.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

REPO = Path('/home/anirudh/ggf_md')
ROOT = Path('/dm7100f/logs/anirudh-ggf_md-20261022')
BASE = ROOT / '.storage-validation'
REPORT_ROOT = ROOT / 'reports/non-fokker-planck'
PYTHON = '/home/anirudh/.local/venvs/ggf_md/bin/python'
PEPTIDES = ['KS', 'AT', 'LW', 'KQ', 'NY', 'IM', 'TD', 'HT', 'NF', 'RL', 'ET', 'AC', 'RV', 'HP', 'AP']
METHODS = ('diffusion', 'two_for_one', 'mixture')
METRICS = ('jsd', 'js_distance', 'pmf', 'code_jsd', 'code_pmf', 'code_mjs')
NORM = 7.27558413079074
CHECKPOINTS = {'diffusion': '59737148472337eede85f308a49a2eb86b462c77acc4b7e87302f205539669ca',
               'two_for_one': '59737148472337eede85f308a49a2eb86b462c77acc4b7e87302f205539669ca',
               'mixture': '3f74dd075ddf495fa83461430dbbb96026a0ffaf2fa1e4e1745aa1eca3338e09'}
PINS = {'train.py': '0a57351495abcc3a03f0f14783d736dcc5da81b67596e52da96fa2906c1dd9f7',
        'model.py': '89284cbcac4ef6e25495932cf141d87ee8eb53e653f7debb66b87ec7dd646ecd',
        'evaluate.py': 'b9be6f91ed11a7a3e81f5fa85dc2c7d5215d36f57ed06f27705f58264b7ab2bc',
        'scripts/run_experiment.py': '505233da13e6a8c355a7e12a24fd0073c8e938c8865178445046e40bb3cf6914',
        'configs/dipeptides.yaml': '955338b896c42d0ad037bf9bd06161062a2ddd548ded15129b048bdfd83cada9'}
MANIFEST_SHA = '1a8719baaf7f906b8c2830b05c6b4576b5b90d044dd2e7a1a585fb80435f4d12'
DEFAULT_RECEIPTS = {
    'iid:mixture': (BASE / 'mixture-iid-completion-review.json',
                    '04b3e61cd94705c608dd1d711bf5c3038fe672d94641182450575d0bf5505abb'),
    'simulation:two_for_one': (BASE / 'two-for-one-simulation-completion-review.json',
                             '1af4e3620e4bc8f5a9e0510100a30f6acc39332b7cdb3dfed0427043f3fbb360')}
FUTURE_PHASES = ('iid:diffusion', 'simulation:diffusion', 'simulation:mixture')
AUDITORS = {'iid:diffusion': BASE / 'audit-diffusion-completed-iid.py',
            'simulation:diffusion': BASE / 'audit-diffusion-completed-simulation.py',
            'simulation:two_for_one': BASE / 'audit-two-for-one-completed-simulation.py',
            'simulation:mixture': BASE / 'audit-mixture-completed-simulation.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def source_pins():
    actual = {name: digest(REPO / name) for name in PINS}
    require(actual == PINS, 'Scientific source/config pins changed')
    return actual


def exact_live(identity):
    require(set(identity) == {'pid', 'start_ticks', 'boot_id'}, 'Invalid native process identity')
    require(type(identity['pid']) is int and identity['pid'] > 0 and
            type(identity['start_ticks']) is int and identity['start_ticks'] > 0 and
            isinstance(identity['boot_id'], str) and bool(identity['boot_id']), 'Invalid identity fields')
    folder = Path('/proc') / str(identity['pid'])
    try:
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        owner = folder.stat().st_uid
    except (FileNotFoundError, ProcessLookupError):
        return False
    native = {'pid': identity['pid'], 'start_ticks': int(fields[19]),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    if native != identity or fields[0] in ('Z', 'X'):
        return False
    require(owner == 1032, 'Exact original process has unexpected owner')
    return True


def original_handles(receipt):
    records = receipt.get('terminal_processes_before', receipt.get('terminal_processes'))
    require(records is not None, 'Full completion receipt lacks terminal identities')
    if isinstance(records, dict):
        require(set(records) in ({'worker', 'child'}, {'wrapper', 'child'}), 'Ambiguous terminal receipt schema')
        worker_key = 'worker' if 'worker' in records else 'wrapper'
        pair = {'worker': records[worker_key]['expected_identity'], 'child': records['child']['expected_identity']}
        witnesses = list(records.values())
    else:
        result = receipt['controller_task']['result']
        pair = {name: result[name] for name in ('worker', 'child')}
        witnesses = records
    require(len(witnesses) == 2 and all(row.get('exact_live') is False for row in witnesses),
            'Receipt must attest two terminal originals')
    require({json.dumps(row['expected_identity'], sort_keys=True) for row in witnesses} ==
            {json.dumps(value, sort_keys=True) for value in pair.values()} and
            pair['worker']['pid'] != pair['child']['pid'], 'Original handle witnesses differ')
    return pair


def expected_identity(phase, settings, protocol_keys):
    kind, method = phase.split(':')
    common = {'checkpoint': CHECKPOINTS[method], 'norm_factor': NORM, 'compiled': True,
              'inference_cache': True, 'factorized_attention': True,
              'factorized_attention_scope': 'potential_experts'}
    if kind == 'iid':
        return {**common, 'samples': settings['num_iid_samples'], 'steps': settings['diffusion_steps'],
                'seed': settings['seed'], 'batch': settings['inference_batch']}
    return {**common, **{key: settings[key] for key in protocol_keys}, 'method': method, 'cuda_graphs': True}


def expected_files(phase):
    kind = phase.split(':')[0]
    if kind == 'iid':
        return {'iid_metadata.json'} | {f'{sequence}_iid{suffix}' for sequence in PEPTIDES for suffix in ('.npy', '_state.pt')}
    return {'simulation_metadata.json', 'simulation_state.pt', 'simulation_complete.json'} | {
        f'{sequence}_simulation.npy' for sequence in PEPTIDES}


def phase_gate(phase, accepted, config, protocol_keys):
    # No phase metadata/state/array is touched until succeeded + accepted full
    # receipt + exact original worker/child terminal have all been established.
    status = read_json(REPO / 'data/experiment/status.json')
    require(status['identity']['config'] == config, 'Controller config differs')
    require(all(status['identity'][key] is True for key in
                ('compile', 'cuda_graphs', 'inference_cache', 'factorized_inference')), 'Execution flags differ')
    task = status['tasks'][phase]
    if task['status'] != 'succeeded' or phase not in accepted:
        return {'ready': False, 'phase': phase, 'controller_status': task['status'],
                'reason': 'controller_phase_not_succeeded' if task['status'] != 'succeeded' else 'accepted_full_completion_receipt_missing'}
    receipt_path, receipt_sha = accepted[phase]
    require(digest(receipt_path) == receipt_sha, f'{phase}: accepted receipt digest differs')
    receipt = read_json(receipt_path)
    require(receipt['status'] == 'PASS' and receipt['source_sha256'] == PINS,
            f'{phase}: full PASS/source-bound receipt required')
    pair = original_handles(receipt)
    require(task['worker'] == pair['worker'] and not any(exact_live(value) for value in pair.values()),
            f'{phase}: original writer/wrapper not terminal or task binding differs')
    handle_path = Path(task['handle_file'])
    require(handle_path.resolve().is_relative_to(ROOT / 'data/experiment'), 'Handle is outside own experiment')
    handle_sha = digest(handle_path)
    handles = read_json(handle_path)
    require(all(handles[name] == pair[name] for name in pair), 'Current handle identities differ')
    result_path = Path(task['result_file'])
    require(result_path.resolve().is_relative_to(ROOT / 'data/experiment'), 'Result is outside own experiment')
    result_sha = digest(result_path)
    result = read_json(result_path)
    require(type(result['returncode']) is int and result['returncode'] == 0 and result.get('error') is None and
            all(result[name] == pair[name] for name in pair), 'Successful original result differs')
    command = task['command']
    kind, method = phase.split(':')
    require(str(REPO / 'evaluate.py') in command and command[command.index('--method') + 1] == method and
            command[command.index('--config') + 1] == str(REPO / 'configs/dipeptides.yaml'), 'Stage command differs')
    require(all(flag in command for flag in (f'--{kind}-only', '--compile', '--inference-cache', '--factorized-attention')) and
            '--profile-only' not in command and '--metrics-only' not in command, 'Stage phase/flags differ')
    identity = expected_identity(phase, config['evaluation'], protocol_keys)
    require(receipt['identity'] == identity, 'Receipt scientific identity differs')
    require(set(receipt['artifact_sha256']) == expected_files(phase), 'Full receipt artifact inventory differs')
    auditor = None
    if phase in AUDITORS:
        auditor = {'path': str(AUDITORS[phase]), 'sha256': receipt['audit_helper_sha256']}
        require(digest(auditor['path']) == auditor['sha256'], 'Completion auditor source changed')
    if kind == 'iid':
        require(receipt['peptide_order'] == PEPTIDES and receipt['counts']['total_samples'] == 735000 and
                receipt['counts']['samples_per_peptide'] == 49000, 'Incomplete IID receipt')
    else:
        require(receipt['state_completed_steps'] == 6000000 and receipt['marker']['complete'] is True and
                receipt['protocol_counts']['saved_frames_per_path'] == 120000 and
                receipt['protocol_counts']['paths_in_single_batch'] == 150, 'Incomplete simulation receipt')
    # Repeat status + exact original identities before the first payload hash.
    fresh = read_json(REPO / 'data/experiment/status.json')
    require(fresh['identity'] == status['identity'] and fresh['tasks'][phase] == task and
            not any(exact_live(value) for value in pair.values()), 'Phase changed during terminal gate')
    directory = ROOT / 'data/results' / method
    require((REPO / 'data/results' / method).resolve() == directory, 'Physical output differs')
    return {'ready': True, 'phase': phase, 'controller_status': 'succeeded', 'receipt_path': str(receipt_path),
            'receipt_sha256': receipt_sha, 'original_handles': pair, 'identity': identity,
            'artifact_sha256': receipt['artifact_sha256'], 'directory': str(directory), 'auditor': auditor,
            'handle_path': str(handle_path), 'handle_sha256': handle_sha,
            'result_path': str(result_path), 'result_sha256': result_sha}


def verified_artifacts(gate):
    actual = {name: digest(Path(gate['directory']) / name) for name in gate['artifact_sha256']}
    require(actual == gate['artifact_sha256'], f"{gate['phase']}: completed artifacts changed")
    return actual


def load_state(path, torch):
    return torch.load(io.BytesIO(Path(path).read_bytes()), map_location='cpu', weights_only=False)


def phase_angles(gate, config, ev, np, torch):
    folder = Path(gate['directory']); kind = gate['phase'].split(':')[0]
    identity = gate['identity']; settings = config['evaluation']; output = {}
    require(read_json(folder / f'{kind}_metadata.json') == identity, 'Phase metadata differs')
    state = None
    if kind == 'simulation':
        state = load_state(folder / 'simulation_state.pt', torch)
        require(state['identity'] == identity and type(state['completed']) is int and state['completed'] == 6000000,
                'Full simulation state differs')
        require(isinstance(state['alive'], np.ndarray) and state['alive'].dtype == np.bool_ and
                state['alive'].shape == (150,), 'Invalid alive flags')
        require(state['x'].dtype == state['v'].dtype == torch.float32 and
                state['x'].shape == state['v'].shape == (150, 30), 'Invalid simulation state coordinates')
        marker = read_json(folder / 'simulation_complete.json')
        require(marker['complete'] is True and marker['steps_per_path'] == 6000000 and
                marker['alive_paths'] == state['alive'].tolist(), 'Full marker differs')
    for index, sequence in enumerate(PEPTIDES):
        if kind == 'iid':
            saved = load_state(folder / f'{sequence}_iid_state.pt', torch)
            require(saved['identity'] == identity and type(saved['completed']) is int and saved['completed'] == 49000,
                    'Incomplete IID state')
            coordinates = np.load(folder / f'{sequence}_iid.npy', mmap_mode='r')
            require(coordinates.shape == (49000, 30) and coordinates.dtype == np.float32, 'Invalid IID array')
            sample = ev.angles(coordinates)
            counts = {'raw_samples': len(coordinates), 'selected_samples': len(sample), 'alive_paths': None,
                      'dead_paths': None, 'total_paths': None}
            rng = saved['rng']
        else:
            trajectories = np.load(folder / f'{sequence}_simulation.npy', mmap_mode='r')
            require(trajectories.shape == (10, 120000, 30) and trajectories.dtype == np.float32,
                    'Invalid simulation array')
            alive = state['alive'][index * 10:(index + 1) * 10]
            # Exactly evaluate_outputs: preserve replica-major flattening, then
            # a fresh seed0 full permutation per peptide, then angles.
            simulation = np.asarray(trajectories[alive]).reshape(-1, 30)
            if len(simulation):
                indices = np.random.default_rng(settings['seed']).permutation(len(simulation))[:settings['num_iid_samples']]
                sample = ev.angles(simulation[indices])
            else:
                sample = np.empty((0, 2))
            counts = {'raw_samples': 1200000, 'surviving_raw_samples': len(simulation), 'selected_samples': len(sample),
                      'alive_paths': int(alive.sum()), 'dead_paths': int((~alive).sum()), 'total_paths': 10}
            rng = state['rng']
        require(torch.is_tensor(rng) and rng.device.type == 'cpu' and rng.dtype == torch.uint8 and rng.shape == (16,),
                'Saved RNG is not CPU uint8[16]')
        output[sequence] = {'angles': sample, 'valid_samples': int(np.isfinite(sample).all(axis=1).sum()), **counts}
    return output


def make_rows(phases, truth, metric_function, bins):
    rows = []
    for method in METHODS:
        iid_phase = 'iid:diffusion' if method == 'two_for_one' else f'iid:{method}'
        simulation_phase = f'simulation:{method}'
        for sequence in PEPTIDES:
            row = {'method': method, 'peptide': sequence, 'reference_samples': len(truth[sequence]),
                   'iid_source_phase': iid_phase, 'simulation_source_phase': simulation_phase}
            for kind, phase in (('iid', iid_phase), ('simulation', simulation_phase)):
                item = phases[phase].get('samples', {}).get(sequence)
                row[f'{kind}_status'] = 'complete' if item is not None else 'pending'
                row[f'{kind}_pending_reason'] = None if item is not None else phases[phase].get('reason', 'completed_phase_peptide_metric_missing')
                for count in ('raw_samples', 'surviving_raw_samples', 'selected_samples', 'valid_samples', 'alive_paths', 'dead_paths', 'total_paths'):
                    row[f'{kind}_{count}'] = None if item is None else item.get(count)
                values = dict.fromkeys(METRICS) if item is None else metric_function(truth[sequence], item['angles'], bins)
                row.update({f'{kind}_{key}': value for key, value in values.items()})
            row['status'] = 'complete' if row['iid_status'] == row['simulation_status'] == 'complete' else 'pending_phases'
            rows.append(row)
    return rows


def summaries(rows, np, paper, paper_metrics):
    summary, comparison = {}, []
    for method in METHODS:
        group = [row for row in rows if row['method'] == method]
        require(len(group) == 15 and {row['peptide'] for row in group} == set(PEPTIDES), 'Incomplete peptide rows')
        summary[method] = {}
        for kind in ('iid', 'simulation'):
            for metric in METRICS:
                name = f'{kind}_{metric}'
                values = [row[name] for row in group if row[name] is not None and np.isfinite(row[name])]
                complete = len(values) == 15 and all(row[f'{kind}_status'] == 'complete' for row in group)
                summary[method][name] = {'status': 'complete' if complete else 'pending_or_missing_values',
                                         'mean': float(np.mean(values)) if complete else None,
                                         'std_across_peptides': float(np.std(values, ddof=0)) if complete else None,
                                         'evaluated_peptides': len(values), 'expected_peptides': 15,
                                         'aggregate_requires_all_15_peptides': True}
        for index, name in enumerate(paper_metrics):
            for definition in ('paper', 'released_code'):
                current = name if definition == 'paper' else name.replace('_jsd', '_code_jsd').replace('_pmf', '_code_pmf')
                measured = summary[method][current]
                comparison.append({'method': method, 'metric_definition': definition, 'metric': name,
                                   'status': measured['status'], 'mean': measured['mean'],
                                   'std': measured['std_across_peptides'], 'paper_mean': paper[method][0][index],
                                   'paper_std': paper[method][1][index],
                                   'gap': None if measured['mean'] is None else measured['mean'] - paper[method][0][index],
                                   'evaluated_peptides': measured['evaluated_peptides'], 'expected_peptides': 15})
    require(len(rows) == 45 and len(comparison) == 24, 'Non-FP report row counts differ')
    return summary, comparison


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Fresh timestamped directory under NFS reports/non-fokker-planck')
    parser.add_argument('--accepted-receipt', nargs=3, action='append', default=[], metavar=('PHASE', 'PATH', 'SHA256'),
                        help='Root-accepted full completion receipt for a newly completed phase')
    parser.add_argument('--plots', action='store_true', help='Use existing plots/angle NPZ only for methods with both phases complete')
    args = parser.parse_args()
    require(sys.executable == PYTHON, 'Run with pinned Python')
    sys.dont_write_bytecode = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '4'
    output = Path(args.output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Output must be exclusive and fresh')
    output = output.resolve()
    require(output != REPORT_ROOT and output.is_relative_to(REPORT_ROOT), 'Output must be inside isolated NFS report root')
    accepted = dict(DEFAULT_RECEIPTS)
    for phase, path, value in args.accepted_receipt:
        require(phase in FUTURE_PHASES and phase not in accepted and len(value) == 64 and
                all(c in '0123456789abcdef' for c in value), 'Invalid/duplicate future accepted receipt')
        accepted[phase] = (Path(path).resolve(strict=True), value)
    before = source_pins()
    output.mkdir(parents=True, exist_ok=False)
    # evaluate imports Matplotlib even when plots are omitted. Keep any font or
    # configuration cache generated by that import inside this report only.
    os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib')
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    import evaluate as ev
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    config = ev.load_config(REPO / 'configs/dipeptides.yaml'); settings = config['evaluation']
    require(settings['peptides'] == PEPTIDES and settings['num_iid_samples'] == 49000 and
            settings['diffusion_steps'] == 1000 and settings['seed'] == 0 and settings['inference_batch'] == 256 and
            settings['simulation_steps'] == 6000000 and settings['save_every'] == 50 and settings['replicas'] == 10,
            'Full evaluation protocol required')
    gates = {phase: phase_gate(phase, accepted, config, ev.PROTOCOL_KEYS) for phase in
             ('iid:diffusion', 'iid:mixture', 'simulation:diffusion', 'simulation:two_for_one', 'simulation:mixture')}
    manifest_path = REPO / 'data/processed/manifest.json'
    require(digest(manifest_path) == MANIFEST_SHA and read_json(manifest_path)['norm_factor'] == NORM,
            'Processed manifest/normalization differs')
    truth_paths = {sequence: REPO / f'data/processed/test/{sequence}.npy' for sequence in PEPTIDES}
    truth_hashes = {sequence: digest(path) for sequence, path in truth_paths.items()}
    split = ev.load_split(config, 'test')
    truth = {sequence: ev.angles(split[sequence]['x']) for sequence in PEPTIDES}
    # Physical reference coordinates are used unchanged, as evaluate_outputs;
    # normalization is checked in phase identity, never applied a second time.
    phases = {}
    for phase, gate in gates.items():
        if not gate['ready']:
            phases[phase] = gate
            continue
        print(f'Reporting completed {phase}', flush=True)
        hashes = verified_artifacts(gate)
        samples = phase_angles(gate, config, ev, np, torch)
        require(verified_artifacts(gate) == hashes, 'Input changed while computing angles')
        fresh = phase_gate(phase, accepted, config, ev.PROTOCOL_KEYS)
        require(fresh == gate, 'Completed phase binding changed')
        phases[phase] = {**gate, 'samples': samples, 'artifact_hashes_before_after_equal': True}
    rows = make_rows(phases, truth, ev.metrics, settings['bins'])
    summary, comparison = summaries(rows, np, ev.PAPER, ev.PAPER_METRICS)
    require({sequence: digest(path) for sequence, path in truth_paths.items()} == truth_hashes and
            digest(manifest_path) == MANIFEST_SHA, 'Reference inputs changed')
    after = source_pins(); require(after == before and not torch.cuda.is_initialized(), 'Source/CUDA state changed')
    plotted = []
    if args.plots:
        for method in METHODS:
            if not all(row['status'] == 'complete' for row in rows if row['method'] == method):
                continue
            folder = output / method; folder.mkdir()
            iid_phase = 'iid:diffusion' if method == 'two_for_one' else f'iid:{method}'
            for sequence in PEPTIDES:
                iid = phases[iid_phase]['samples'][sequence]['angles']
                simulation = phases[f'simulation:{method}']['samples'][sequence]['angles']
                np.savez(folder / f'{sequence}_angles.npz', reference=truth[sequence], iid=iid, simulation=simulation)
                ev.plot_results(truth[sequence][np.isfinite(truth[sequence]).all(1)], iid[np.isfinite(iid).all(1)],
                                simulation[np.isfinite(simulation).all(1)], sequence, folder, settings)
            plotted.append(method)
    phase_receipts = {phase: {key: value for key, value in information.items() if key != 'samples'}
                      for phase, information in phases.items()}
    report = {'status': 'PHASE_AWARE_NON_FOKKER_PLANCK_REPORT', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'source': 'https://arxiv.org/pdf/2506.17139v3', 'methods': summary, 'comparison': comparison,
              'settings': settings, 'phase_coverage': phase_receipts, 'plots_completed_methods': plotted,
              'notes': ['Only full controller-succeeded, terminal and independently audited phases were read.',
                        'Pending phases have null metrics; no partial-peptide or partial-run means.',
                        'Every aggregate requires all15 peptide values; standard deviation uses population ddof0.',
                        'Two-for-One shares Diffusion IID samples, including its readiness gate.',
                        'Exact evaluate.angles/metrics definitions; paper and released definitions use identical samples.',
                        'Simulation sampling uses all frames of fully completed alive paths, replica-major, seed0 per peptide, at most49000 selected frames.',
                        'Dead raw paths remain untouched and excluded exactly as in evaluate_outputs; all counts are retained.',
                        'Fokker-Planck training/full experiment continue separately; this report does not establish their completion.']}
    write_json(output / 'per_peptide.json', rows); write_csv(output / 'per_peptide.csv', rows)
    write_json(output / 'summary.json', report); write_csv(output / 'comparison.csv', comparison)
    outputs = {str(path.relative_to(output)): digest(path) for path in output.rglob('*') if path.is_file()}
    receipt = {'status': 'PASS_COMPLETED_PHASES_REPORTED_PENDING_PHASES_NULL', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'output': str(output), 'script_sha256': digest(Path(__file__)), 'source5_before': before, 'source5_after': after,
               'processed_manifest_sha256': MANIFEST_SHA, 'reference_input_sha256': truth_hashes,
               'reference_coordinate_units': 'nm; unchanged physical test coordinates; no extra normalization',
               'phase_inputs': phase_receipts, 'per_peptide_rows': 45, 'comparison_rows': 24, 'output_sha256': outputs,
               'runtime': {'python': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
                           'CPU_threads': torch.get_num_threads(), 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                           'CUDA_initialized': torch.cuda.is_initialized()},
               'limits': ['Completed phase metrics only; pending phases require separate full completion audits.',
                          'No checkpoint tensor read, sampling, replay, GPU query/work, process action or production lock.',
                          'Only this exclusive isolated report directory was written; production metrics/summaries untouched.']}
    write_json(output / 'report_receipt.json', receipt)
    print(json.dumps({'output': str(output), 'receipt_sha256': digest(output / 'report_receipt.json'),
                      'ready_phases': [phase for phase, gate in gates.items() if gate['ready']],
                      'per_peptide_rows': 45, 'comparison_rows': 24}, indent=2))


if __name__ == '__main__':
    main()
