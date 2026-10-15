"""Run protocol6201344116, or software controls using the available LR archive.

`run` never substitutes controls/retrained policies for missing epoch-reuse data.
All original diagonal paths must compare byte-for-byte BEFORE any off-diagonal
measurement is collected. Uses the unchanged archived native evaluator.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from crossover_core import *
from pack_inputs import FULL_SHA256, wanted_paths
from physical_reader import read_group, summarize


def check_input_bundle(root: Path) -> dict:
    require(root.is_dir(), 'Original epoch-reuse input bundle is missing')
    origin = json.loads((root/'ORIGIN.json').read_text())
    require(origin['source_full_sha256'] == FULL_SHA256 and origin['protocol'] == PROTOCOL,
            'Wrong original experiment identity')
    require(digest(root/'INPUT_MANIFEST.json') == origin['input_manifest_sha256'], 'Input manifest digest changed')
    manifest = json.loads((root/'INPUT_MANIFEST.json').read_text())
    require(set(manifest) == set(wanted_paths()) | {'SOURCE_MANIFEST.json'}, 'Incomplete input bundle manifest')
    for name, h in manifest.items():
        require((root/name).is_file() and digest(root/name) == h, 'Missing or changed input: '+name)
    original = json.loads((root/'SOURCE_MANIFEST.json').read_text())
    require(all(original[n] == h for n,h in manifest.items() if n != 'SOURCE_MANIFEST.json'), 'Lost original member binding')
    for seed in SEEDS:
        for epochs in (4, 10):
            p = root/f'results/{seed}-decay{epochs}/{CHECKPOINT:07d}'
            receipt = json.loads((p/'RECEIPT.json').read_text())
            require(receipt['seed'] == seed and receipt['steps'] == CHECKPOINT and
                    receipt['arm'] == f'decay{epochs}' and receipt['epochs'] == epochs,
                    'Wrong source checkpoint')
            require(receipt['policy_sha256'] == digest(p/'policy.npz') and
                    receipt['normalizer_sha256'] == digest(p/'NORMALIZER.json'), 'Wrong model/statistics pairing')
            policy_checked(load_npz(p/'policy.npz'))
            normalizer_checked(json.loads((p/'NORMALIZER.json').read_text()))
    return origin


def load_reference(root: Path, out: Path):
    # Delayed imports: input failure occurs before Torch/native initialization.
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for name, expected in manifest.items():
        require(digest(root/name) == expected, 'Changed reference input: '+name)
    sys.path.insert(0, str(root/'code'))
    import run_study
    import torch
    ref = run_study.runtime(out/'runtime')
    return run_study, ref, torch


def install(model, weights, normalizer, torch):
    policy_checked(weights); normalizer_checked(normalizer)
    model.policy.load_state_dict({k: torch.from_numpy(v.copy()) for k,v in weights.items()}, strict=True)
    env = model.get_vec_normalize_env(); require(env is not None, 'Expected normalized model')
    env.training = False
    env.obs_rms.mean = np.array(normalizer['mean'], float)
    env.obs_rms.var = np.array(normalizer['var'], float)
    env.obs_rms.count = normalizer['count']
    require(model.num_timesteps == 0 and not model.policy.optimizer.state, 'Unexpected learning/optimizer activity')


def guard_optimizer(model):
    def reject(*_):
        raise RuntimeError('Optimizer step forbidden in inference-only crossover')
    return model.policy.optimizer.register_step_pre_hook(reject)


def measure(old, ref, model, seed, panel, mode, selection, path, weights, norm):
    require(not path.exists(), 'Refusing to overwrite an observed trajectory')
    path.parent.mkdir(parents=True, exist_ok=True)
    count = old.eval_group(ref, model, seed, panel, mode, selection, path)
    arrays = load_npz(path)
    base, error = read_group(arrays, weights, norm, mode)
    extra = effort_metrics(arrays)
    require([r['key'] for r in base] == [r['key'] for r in extra], 'Effort case mismatch')
    rows = [dict(a, **{k:v for k,v in b.items() if k != 'key'}) for a,b in zip(base,extra)]
    require(count == sum(r['steps'] for r in rows), 'Native transition count mismatch')
    write_json(path.with_suffix('.json'), dict(native_transitions=count,
        trace_sha256=digest(path), max_independent_inference_error=error, cases=rows))
    return arrays, rows, count


def run_protocol(inputs: Path, reference: Path, output: Path):
    origin = check_input_bundle(inputs)
    require(not output.exists(), 'Output directory already exists')
    output.mkdir(parents=True)
    write_json(output/'INPUT_ORIGIN.json', origin)
    study, ref, torch = load_reference(reference, output)
    old = study.old
    selection = inputs/'parent/results/selection'
    total = 0
    # The first phase writes only diagonal results. A single mismatch stops
    # the entire experiment before ANY crossed-policy outcome is collected.
    for phase, arms in [('diagonal', ('M4S4','M10S10')), ('crossed', ('M4S10','M10S4'))]:
        if phase == 'crossed':
            require((output/'DIAGONALS_EXACT.json').exists(), 'Exact diagonal gate not passed')
        for seed in SEEDS:
            model, _ = old.build(ref, seed, 'obsnorm'); guard = guard_optimizer(model)
            try:
                for arm in arms:
                    mean_epochs, sigma_epochs = ARMS[arm]
                    mean_path = inputs/f'results/{seed}-decay{mean_epochs}/{CHECKPOINT:07d}'
                    sigma_path = inputs/f'results/{seed}-decay{sigma_epochs}/{CHECKPOINT:07d}'
                    weights = crossed_policy(load_npz(mean_path/'policy.npz'), load_npz(sigma_path/'policy.npz'))
                    norm = json.loads((mean_path/'NORMALIZER.json').read_text())
                    install(model, weights, norm, torch)
                    dest = output/f'results/{seed}/{arm}'; dest.mkdir(parents=True)
                    np.savez_compressed(dest/'policy.npz', **weights)
                    write_json(dest/'NORMALIZER.json', norm)
                    write_json(dest/'BINDING.json', dict(seed=seed, arm=arm, checkpoint=CHECKPOINT,
                        mean_policy=digest(mean_path/'policy.npz'), sigma_policy=digest(sigma_path/'policy.npz'),
                        normalizer=digest(mean_path/'NORMALIZER.json'),
                        policy_sha256=digest(dest/'policy.npz')))
                    for panel in PANELS:
                        for mode in MODES:
                            file = dest/f'{panel}-{mode}.npz'
                            arrays, _, count = measure(old, ref, model, seed, panel, mode, selection, file, weights, norm)
                            total += count
                            if phase == 'diagonal':
                                expected = load_npz(mean_path/file.name)
                                require(equal_arrays(arrays, expected), 'Original diagonal raw trajectory mismatch')
                            elif mode == 'deterministic':
                                own_arm = f'M{mean_epochs}S{mean_epochs}'
                                require(equal_arrays(arrays, load_npz(output/f'results/{seed}/{own_arm}/{file.name}')),
                                        'Deterministic output changed with sigma')
                            print('GROUP_OK', seed, arm, panel, mode, count, flush=True)
                    require(old.normal_state(model) == norm and model.num_timesteps == 0 and
                            not model.policy.optimizer.state, 'Frozen state changed')
            finally:
                guard.remove(); model.env.close()
        if phase == 'diagonal':
            write_json(output/'DIAGONALS_EXACT.json', dict(groups=32, cases=2048,
                native_transitions=total, original_inputs_manifest=digest(inputs/'INPUT_MANIFEST.json')))
    write_json(output/'COMPLETE.json', dict(protocol=PROTOCOL, status='complete',
        groups=64, trajectories=4096, native_transitions=total, training_steps=0,
        optimizer_steps=0, new_independent_cases=0))


def software_controls(reference: Path, output: Path):
    """Available genuine ten-pass checkpoints only; NOT the registered M4 test."""
    require(not output.exists(), 'Control output directory already exists')
    output.mkdir(parents=True)
    study, ref, torch = load_reference(reference, output)
    old = study.old
    expected = json.loads((reference/'analysis/CASES.json').read_text())
    total = 0; records = []; max_error = 0.
    for seed in SEEDS:
        model, _ = old.build(ref, seed, 'obsnorm'); guard = guard_optimizer(model)
        p = reference/f'results/{seed}-decay/{CHECKPOINT:07d}'
        w = load_npz(p/'policy.npz'); norm = json.loads((p/'NORMALIZER.json').read_text())
        install(model, w, norm, torch)
        try:
            for panel in PANELS:
                for mode in MODES:
                    dest = output/f'{seed}/{panel}-{mode}.npz'
                    a, rows, count = measure(old,ref,model,seed,panel,mode,
                        reference/'parent/results/selection',dest,w,norm)
                    total += count
                    prior = [r for r in expected if r['seed']==seed and r['arm']=='decay' and
                             r['checkpoint']==CHECKPOINT and r['panel']==panel and r['mode']==mode]
                    require(len(prior)==CASES, 'Missing prior reference cases')
                    # Original full paths are not in the compact archive. Compare
                    # exactly the saved case summaries, not unavailable arrays.
                    for new, saved in zip(rows,prior):
                        for key in ('key','steps','ending','completed','raw_return','discounted_return','margin','minimum_tick','margin50mm_complete'):
                            require(new[key] == saved[key], 'Historical ten-pass summary mismatch: '+key)
                    records.extend(dict(r,seed=seed,panel=panel,mode=mode) for r in rows)
                    print('REFERENCE_CONTROL_OK',seed,panel,mode,count,flush=True)
            # A real alternate learned sigma is used ONLY for a deterministic
            # invariance control. It is not S4 and not an interpreted outcome.
            donor=load_npz(reference/f'results/{seed}-constant/{CHECKPOINT:07d}/policy.npz')
            require(not np.array_equal(donor['log_std'],w['log_std']), 'Control sigma must actually differ')
            changed=crossed_policy(w,donor);install(model,changed,norm,torch)
            dst=output/f'{seed}/sigma_changed_deterministic_control.npz'
            a,_,count=measure(old,ref,model,seed,'broad','deterministic',reference/'parent/results/selection',dst,changed,norm)
            total += count
            require(equal_arrays(a,load_npz(output/f'{seed}/broad-deterministic.npz')), 'Sigma deterministic invariance failed')
            require(model.num_timesteps==0 and not model.policy.optimizer.state, 'Unexpected training')
            print('SIGMA_INVARIANCE_OK',seed,count,flush=True)
        finally:
            guard.remove();model.env.close()
    write_json(output/'CASES.json',records)
    write_json(output/'CONTROL_VERIFIED.json',dict(status='software_controls_only',
        registered_crossover_complete=False, reference_summaries_exact=1024,
        deterministic_sigma_control_trajectories=256, total_trajectories=1280,
        native_transitions=total,training_steps=0,optimizer_steps=0,
        four_pass_checkpoint_available=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['check-inputs','run','controls'])
    ap.add_argument('--inputs',type=Path)
    ap.add_argument('--reference',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    try:
        if a.stage=='check-inputs':
            require(a.inputs is not None, '--inputs required')
            data=check_input_bundle(a.inputs);write_json(a.output,data)
        elif a.stage=='run':
            require(a.inputs is not None and a.reference is not None, '--inputs and --reference required')
            run_protocol(a.inputs,a.reference,a.output)
        else:
            require(a.reference is not None,'--reference required')
            software_controls(a.reference,a.output)
    except (ValueError, FileNotFoundError, AssertionError) as error:
        # Never emit a success receipt after a missing input or failed gate.
        print(json.dumps(dict(status='blocked',stage=a.stage,error=str(error),
                              registered_crossover_complete=False)), file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
