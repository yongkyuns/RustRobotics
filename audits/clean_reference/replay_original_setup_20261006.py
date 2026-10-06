#!/usr/bin/env python3
"""Use the original archived model construction/RNG setup, not a bare policy.
Keeps every exact replay gate and the evaluator/tracing implementation unchanged.
The initial pinned attempt with a directly constructed ActorCriticPolicy matched
all seed201/202 outcomes but differed in 11 seed203 checkpoint-zero numeric
records. That attempt remains preserved; no tolerance or outcome is changed.
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch
import replay_checkpoints_20261006 as replay


def original_policy(folder: Path, cp: int):
    source=folder.parent/'clean-ppo-native/sources/audits/clean_reference/reference.py'
    spec=importlib.util.spec_from_file_location('original_construction',source)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    seed=int(folder.name.rsplit('-',1)[1])
    actor=np.fromfile(folder/'initial-actor.bin',dtype='<f4')
    critic=np.fromfile(folder/'initial-critic.bin',dtype='<f4')
    model,env,transform=ref.make_model('sb3-defaults',seed,actor,critic)
    replay.need(transform==2,'stock original setup')
    with np.load(folder/f'policy-{cp}.npz',allow_pickle=False) as z:
        state={k:torch.from_numpy(z[k].copy()) for k in z.files}
    model.policy.load_state_dict(state,strict=True)
    # No learn(), train(), optimizer.step() or environment transitions occur.
    env.close()
    return model


if __name__=='__main__':
    replay.policy=original_policy
    replay.main()
