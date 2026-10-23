"""Small, isolated CUDA preflight; never marks full experiment completion."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--physical-gpu', default='3')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.physical_gpu:
        raise ValueError('Preflight must expose exactly its requested physical GPU')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                          '--format=csv,noheader,nounits'], text=True)
    selected = [row.split(',') for row in inventory.splitlines()
                if row.split(',')[0].strip() == args.physical_gpu][0]
    gpu_uuid = selected[1].strip()
    applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader'], text=True)
    if int(selected[2].strip()) >= 64 or any(row.split(',')[0].strip() == gpu_uuid for row in applications.splitlines()):
        raise RuntimeError('Requested preflight GPU is occupied')
    import numpy as np
    import torch
    import evaluate as ev
    from train import load_config, load_split
    from model import DipeptideModel
    from ggf_score_matching import ggf_identity
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    device = torch.device('cuda:0')
    config = load_config(args.config)
    objective = ggf_identity(config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: digest(REPO / name) for name in ('model.py', 'train.py', 'evaluate.py', 'ggf_score_matching.py')}
    split = load_split(config, 'test')
    norm = json.loads((Path(config['data']['root']) / 'processed/manifest.json').read_text())['norm_factor']
    model = DipeptideModel('ggf', config).to(device).eval().requires_grad_(False)
    assert len(model.experts) == 1 and model.experts[0].potential
    assert sum(p.numel() for p in model.parameters()) == 1398497
    # This checkpoint is explicitly untrained and restricted to the preflight.
    checkpoint = output / 'untrained-preflight.pt'
    torch.save({'model': model.state_dict(), 'ema': model.state_dict(), 'method': 'ggf',
                'ggf': objective, 'config': config, 'norm_factor': norm,
                'completed': False, 'preflight_only': True, 'epoch': 0, 'global_step': 0}, checkpoint)
    checkpoint_hash = digest(checkpoint)
    loaded, loaded_norm, _, loaded_hash = ev.load_model(config, 'ggf', device, checkpoint)
    assert loaded_hash == checkpoint_hash and loaded_norm == norm
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    del loaded
    sequence = config['evaluation']['peptides'][0]
    x = torch.as_tensor(np.asarray(split[sequence]['x'][:2]).copy(), device=device) * norm
    features = torch.as_tensor(split[sequence]['features'], device=device).unsqueeze(0).expand(2, -1, -1)
    references = [model(x, features, torch.full((2, 1), value, device=device)).detach()
                  for value in (-1., 0., .02, 1., 20.)]
    assert all(torch.equal(references[0], item) for item in references)
    ev.factorized_inference_checked(model, split, config, norm, device)
    ev.cache_inference_checked(model, split, config, norm, device)
    compiled_score = ev.make_score(model, True)
    actual = compiled_score(x, features, torch.zeros((2, 1), device=device), 20.)
    ev.assert_inference_score_close(actual, references[0])
    relative = float(torch.linalg.vector_norm((actual - references[0]).double()) /
                     torch.linalg.vector_norm(references[0].double()))
    iid_config = copy.deepcopy(config)
    iid_config['evaluation'].update(peptides=[sequence], num_iid_samples=4, inference_batch=2)
    iid_folder = output / 'iid-smoke'
    began = time.monotonic()
    ev.generate_iid(iid_config, split, model, norm, checkpoint_hash, iid_folder, device, compiled=True)
    iid_seconds = time.monotonic() - began
    iid = np.load(iid_folder / f'{sequence}_iid.npy')
    assert iid.shape == (4, 30) and np.isfinite(iid).all()
    iid_hash = digest(iid_folder / f'{sequence}_iid.npy')
    ev.generate_iid(iid_config, split, model, norm, checkpoint_hash, iid_folder, device, compiled=True)
    assert digest(iid_folder / f'{sequence}_iid.npy') == iid_hash
    sim_config = copy.deepcopy(config)
    sim_config['evaluation'].update(simulation_steps=100, chunk_frames=2)
    sim_folder = output / 'simulation-smoke'
    began = time.monotonic()
    ev.simulate(sim_config, split, model, norm, checkpoint_hash, sim_folder, device,
                compiled=True, cuda_graphs=True)
    sim_seconds = time.monotonic() - began
    state = torch.load(sim_folder / 'simulation_state.pt', map_location='cpu', weights_only=False)
    assert state['completed'] == 100 and state['alive'].tolist() == [True] * 150
    assert state['identity']['ggf'] == objective and state['rng'].dtype == torch.uint8 and state['rng'].shape == (16,)
    assert torch.isfinite(state['x']).all() and torch.isfinite(state['v']).all()
    hashes = {}
    for index, peptide in enumerate(config['evaluation']['peptides']):
        path = sim_folder / f'{peptide}_simulation.npy'
        array = np.load(path)
        assert array.shape == (10, 2, 30) and np.isfinite(array).all()
        assert array[:, -1].tobytes() == state['x'][index*10:(index+1)*10].numpy().tobytes()
        hashes[peptide] = digest(path)
    ev.simulate(sim_config, split, model, norm, checkpoint_hash, sim_folder, device,
                compiled=True, cuda_graphs=True)
    assert hashes == {p: digest(sim_folder / f'{p}_simulation.npy') for p in hashes}
    assert sources == {name: digest(REPO / name) for name in sources}
    report = {'status': 'PASS_CUDA_PREFLIGHT_NOT_FULL_EXPERIMENT', 'physical_gpu': args.physical_gpu,
              'gpu_uuid': gpu_uuid, 'torch': torch.__version__, 'numpy': np.__version__, 'ggf': objective,
              'source_before_after': sources, 'parameter_count': 1398497,
              'time_invariance_exact': True, 'compiled_cached_original_relative_error': relative,
              'iid': {'samples': 4, 'steps_per_sample': 1000, 'all_finite': True,
                      'completed_resume_unchanged': True, 'seconds': iid_seconds},
              'simulation': {'steps_per_path': 100, 'paths': 150, 'all_alive': True,
                             'last_frames_saved_state_byte_exact': True, 'cuda_graphs': True,
                             'completed_resume_unchanged': True, 'seconds': sim_seconds},
              'limits': ['Untrained model and private smoke outputs only; no sampling-quality or full-run claim.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
