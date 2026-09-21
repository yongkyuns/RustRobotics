#!/usr/bin/env python3
"""Independent NumPy reconstruction of RustRobotics PPO update 4140.

This intentionally does not import RustRobotics, Burn, torch, jax, or an RL library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import zipfile

import numpy as np

DIMS = ((4, 64), (64, 64), (64, 1))
ACTION_LIMIT = 20.0
ACTION_STD = 2.0
LATENT_STD = ACTION_STD / ACTION_LIMIT
CLIP = 0.2
UPDATE = 4140


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_member(names, suffix: str) -> str:
    matches = [n for n in names if n == suffix or n.endswith("/" + suffix)]
    if len(matches) != 1:
        raise AssertionError(f"expected one member ending {suffix!r}, got {matches[:10]}")
    return matches[0]


def load_json(z: zipfile.ZipFile, names, suffix: str):
    return json.loads(z.read(find_member(names, suffix)))


def load_bytes(z: zipfile.ZipFile, names, suffix: str) -> bytes:
    return z.read(find_member(names, suffix))


def decode_flat(data: bytes) -> np.ndarray:
    if len(data) != 4545 * 4:
        raise AssertionError(f"unexpected network byte count {len(data)}")
    a = np.frombuffer(data, dtype="<f4").copy()
    if a.size != 4545 or not np.isfinite(a).all():
        raise AssertionError("invalid network snapshot")
    return a


def unpack(flat: np.ndarray):
    off = 0
    layers = []
    for ni, no in DIMS:
        n = ni * no
        w = flat[off:off+n].reshape(ni, no).astype(np.float64)
        off += n
        b = flat[off:off+no].astype(np.float64)
        off += no
        layers.append((w, b))
    assert off == 4545
    return layers


def forward(flat: np.ndarray, x: np.ndarray, cache=False):
    x = np.asarray(x, dtype=np.float64)
    (w1,b1),(w2,b2),(w3,b3) = unpack(flat)
    z1 = x @ w1 + b1
    h1 = np.maximum(z1, 0.0)
    z2 = h1 @ w2 + b2
    h2 = np.maximum(z2, 0.0)
    y = (h2 @ w3 + b3)[:,0]
    if cache:
        return y, (x,z1,h1,z2,h2,w1,w2,w3)
    return y


def backprop_mlp(flat: np.ndarray, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    _, c = forward(flat, x, cache=True)
    x,z1,h1,z2,h2,w1,w2,w3 = c
    dy = np.asarray(dy, dtype=np.float64).reshape(-1,1)
    dw3 = h2.T @ dy
    db3 = dy.sum(axis=0)
    dh2 = dy @ w3.T
    dz2 = dh2 * (z2 > 0.0)
    dw2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ w2.T
    dz1 = dh1 * (z1 > 0.0)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return np.concatenate([dw1.ravel(),db1,dw2.ravel(),db2,dw3.ravel(),db3])


def log_jac(z):
    a = np.abs(z)
    return 2.0 * (math.log(2.0) - a - np.log1p(np.exp(-2.0*a)))


def log_prob(mean, latent):
    mean = np.asarray(mean, dtype=np.float64)
    latent = np.asarray(latent, dtype=np.float64)
    standardized = (latent - mean) / LATENT_STD
    return (-0.5*standardized*standardized
            - math.log(LATENT_STD)
            - 0.5*math.log(2.0*math.pi)
            - math.log(ACTION_LIMIT)
            - log_jac(latent))


def latent_log_prob(mean, latent):
    standardized = (latent - mean) / LATENT_STD
    return (-0.5*standardized*standardized
            - math.log(LATENT_STD)
            - 0.5*math.log(2.0*math.pi))


def policy_loss_and_grad(flat, obs, latent, oldlp, adv):
    mean = forward(flat, obs)
    newlp = log_prob(mean, latent)
    ratio = np.exp(newlp - oldlp)
    unclipped = ratio * adv
    clipped_ratio = np.clip(ratio, 1.0-CLIP, 1.0+CLIP)
    clipped = clipped_ratio * adv
    term = np.minimum(unclipped, clipped)
    loss = -term.mean()

    active = np.where(adv >= 0.0, ratio <= 1.0+CLIP, ratio >= 1.0-CLIP)
    dlog_dmean = (latent - mean) / (LATENT_STD*LATENT_STD)
    dmean = np.where(active, -(adv * ratio * dlog_dmean) / len(adv), 0.0)
    grad = backprop_mlp(flat, obs, dmean)
    return float(loss), grad, mean, ratio


def value_loss_and_grad(flat, obs, returns):
    pred = forward(flat, obs)
    err = pred - returns
    loss = float(np.mean(err*err))
    dpred = 2.0*err/len(err)
    grad = backprop_mlp(flat, obs, dpred)
    return loss, grad, pred


def seq_f32_sum(values):
    acc = np.float32(0.0)
    for v in values:
        acc = np.float32(acc + np.float32(v))
    return acc


def normalize_f32(values):
    v = np.asarray(values, dtype=np.float32)
    if len(v) == 0:
        return v.copy()
    mean = np.float32(seq_f32_sum(v) / np.float32(len(v)))
    sq = []
    for x in v:
        c = np.float32(x - mean)
        sq.append(np.float32(c*c))
    var = np.float32(seq_f32_sum(sq) / np.float32(len(v)))
    std = np.float32(max(float(np.sqrt(var, dtype=np.float32)), 1.0e-6))
    out = np.empty_like(v)
    for i,x in enumerate(v):
        out[i] = np.float32(np.float32(x - mean) / std)
    return out


def gae_paths_f32(batch, gamma=np.float32(0.995), lam=np.float32(1.0)):
    rewards = np.asarray(batch["rewards"], dtype=np.float32)
    values = np.asarray(batch["values"], dtype=np.float32)
    terminals = np.asarray(batch["terminals"], dtype=bool)
    ends = np.asarray(batch["ends"], dtype=bool)
    bootstraps = np.asarray(batch["bootstraps"], dtype=np.float32)
    ret = np.empty_like(rewards)
    adv = np.empty_like(rewards)
    start = 0
    for end in np.flatnonzero(ends):
        next_value = np.float32(bootstraps[end])
        gae = np.float32(0.0)
        for i in range(int(end), start-1, -1):
            nt = np.float32(0.0 if terminals[i] else 1.0)
            delta = np.float32(
                np.float32(rewards[i] + np.float32(gamma * np.float32(next_value*nt)))
                - values[i]
            )
            gae = np.float32(delta + np.float32(np.float32(gamma*lam) * np.float32(nt*gae)))
            adv[i] = gae
            ret[i] = np.float32(gae + values[i])
            next_value = values[i]
        start = int(end)+1
    if start != len(rewards):
        raise AssertionError(f"GAE paths did not consume batch: {start}/{len(rewards)}")
    return ret, adv


def bit_mismatch(a,b):
    a=np.asarray(a,dtype=np.float32)
    b=np.asarray(b,dtype=np.float32)
    if a.shape != b.shape:
        return int(max(a.size,b.size))
    return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))


def directional_check(flat, loss_fn, grad, seed):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=flat.size)
    d /= np.linalg.norm(d)
    eps = 1e-6
    p = flat.astype(np.float64)
    fd = (loss_fn(p + eps*d) - loss_fn(p - eps*d))/(2.0*eps)
    analytic = float(np.dot(grad,d))
    denom = max(1.0,abs(fd),abs(analytic))
    return fd, analytic, abs(fd-analytic)/denom


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("out")
    ap.add_argument("--expected-sha256", required=True)
    args=ap.parse_args()
    artifact=Path(args.artifact)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    raw=artifact.read_bytes()
    outer_sha=sha256_bytes(raw)
    if outer_sha != args.expected_sha256:
        raise AssertionError(f"outer artifact digest mismatch: {outer_sha}")

    results={"artifact_sha256":outer_sha,"update":UPDATE,"checks":{},"steps":[]}
    with zipfile.ZipFile(artifact) as z:
        names=z.namelist()
        (out/"members.txt").write_text("\n".join(names)+"\n")
        manifest_name=find_member(names,"manifest.json")
        manifest=json.loads(z.read(manifest_name))
        manifest_errors=[]
        for name,expected in manifest.items():
            # Prefer the manifest's exact root-relative member. Only fall back to
            # an enclosing artifact prefix when the exact member is absent.
            # Using suffix matches even when the exact member exists makes names
            # such as actor-0.bin ambiguous with update-*/actor-0.bin.
            candidates=[name] if name in names else [n for n in names if n.endswith("/"+name)]
            if len(candidates)!=1:
                manifest_errors.append([name,"missing-or-ambiguous",len(candidates)])
                continue
            got=sha256_bytes(z.read(candidates[0]))
            if got != expected:
                manifest_errors.append([name,expected,got])
        results["checks"]["manifest_errors"]=manifest_errors
        if manifest_errors:
            raise AssertionError(f"manifest verification failed for {len(manifest_errors)} members")

        union=load_json(z,names,f"update-{UPDATE}/union.json")
        batch=load_json(z,names,f"update-{UPDATE}/batch.json")
        actor0=decode_flat(load_bytes(z,names,f"update-{UPDATE}/actor-0.bin"))
        critic0=decode_flat(load_bytes(z,names,f"update-{UPDATE}/critic-0.bin"))

        obs=np.asarray(union["observations"],dtype=np.float64)
        lat=np.asarray(union["latents"],dtype=np.float64)
        oldlp=np.asarray(union["old_log_probs"],dtype=np.float64)
        returns=np.asarray(union["returns"],dtype=np.float64)
        raw_adv=np.asarray(union["raw"],dtype=np.float64)
        adv=np.asarray(union["normalized"],dtype=np.float64)
        if obs.shape!=(1024,4):
            raise AssertionError(f"unexpected union observations {obs.shape}")
        for name,v in [("latents",lat),("old_log_probs",oldlp),("returns",returns),("raw",raw_adv),("normalized",adv)]:
            if v.shape!=(1024,) or not np.isfinite(v).all():
                raise AssertionError(f"bad {name} shape/values")

        # Exact/sequential-f32 target checks for the original 512-row live rollout.
        ref_ret,ref_raw=gae_paths_f32(batch)
        b_ret=np.asarray(batch["returns1"],dtype=np.float32)
        b_raw=np.asarray(batch["raw1"],dtype=np.float32)
        b_norm=np.asarray(batch["normalized1"],dtype=np.float32)
        norm_ref=normalize_f32(ref_raw)
        target_check={
            "returns_bit_mismatches":bit_mismatch(ref_ret,b_ret),
            "raw_advantage_bit_mismatches":bit_mismatch(ref_raw,b_raw),
            "normalized_bit_mismatches":bit_mismatch(norm_ref,b_norm),
        }
        results["checks"]["primary_gae"]=target_check

        union_norm_ref=normalize_f32(np.asarray(union["raw"],dtype=np.float32))
        results["checks"]["union_normalization"]={
            "bit_mismatches":bit_mismatch(union_norm_ref,np.asarray(union["normalized"],dtype=np.float32)),
            "mean":float(np.mean(np.asarray(union["normalized"],dtype=np.float64))),
            "std":float(np.std(np.asarray(union["normalized"],dtype=np.float64))),
        }

        # The incoming actor must explain the stored behavior likelihoods.
        incoming_mean=forward(actor0,obs)
        oldlp_ref=log_prob(incoming_mean,lat)
        oldlp_err=np.abs(oldlp_ref-oldlp)
        results["checks"]["old_log_prob"]={
            "max_abs":float(oldlp_err.max()),
            "mean_abs":float(oldlp_err.mean()),
            "p99_abs":float(np.quantile(oldlp_err,0.99)),
        }

        actor_prev=actor0
        critic_prev=critic0
        max_policy_err=0.0; max_value_err=0.0; max_mean_err=0.0
        max_ratio_equiv=0.0; max_actor_fd=0.0; max_critic_fd=0.0
        positive_descent=0
        for step in range(1,17):
            rec=load_json(z,names,f"update-{UPDATE}/optimizer/step-{step}.json")
            idx=np.asarray(rec["indices"],dtype=np.int64)
            if idx.size!=256 or np.unique(idx).size!=256 or idx.min()<0 or idx.max()>=1024:
                raise AssertionError(f"bad minibatch indices step {step}")
            x=obs[idx]; zz=lat[idx]; ol=oldlp[idx]; aa=adv[idx]; rr=returns[idx]
            ploss,pgrad,pmean,ratio=policy_loss_and_grad(actor_prev,x,zz,ol,aa)
            vloss,vgrad,vpred=value_loss_and_grad(critic_prev,x,rr)
            policy_err=abs(ploss-float(rec["policy_loss"]))
            value_err=abs(vloss-float(rec["value_loss"]))
            max_policy_err=max(max_policy_err,policy_err)
            max_value_err=max(max_value_err,value_err)

            actor_after=decode_flat(load_bytes(z,names,f"update-{UPDATE}/optimizer/actor-{step}.bin"))
            critic_after=decode_flat(load_bytes(z,names,f"update-{UPDATE}/optimizer/critic-{step}.bin"))
            post_means=forward(actor_after,obs)
            recorded=np.asarray(rec["means"],dtype=np.float64)
            mean_err=float(np.max(np.abs(post_means-recorded)))
            max_mean_err=max(max_mean_err,mean_err)

            # The tanh/action Jacobian is state-independent with respect to the policy mean
            # for a fixed stored latent, so it must cancel from the PPO ratio.
            new_action_lp=log_prob(pmean,zz)
            old_action_lp=ol
            old_latent_lp=latent_log_prob(incoming_mean[idx],zz)
            new_latent_lp=latent_log_prob(pmean,zz)
            ratio_lat=np.exp(new_latent_lp-old_latent_lp)
            ratio_action=np.exp(new_action_lp-old_action_lp)
            ratio_equiv=float(np.max(np.abs(ratio_lat-ratio_action)))
            max_ratio_equiv=max(max_ratio_equiv,ratio_equiv)

            # Independent analytic backprop vs numerical directional derivative.
            def pf(q):
                return policy_loss_and_grad(np.asarray(q,dtype=np.float64),x,zz,ol,aa)[0]
            def vf(q):
                return value_loss_and_grad(np.asarray(q,dtype=np.float64),x,rr)[0]
            pfd,pan,perr=directional_check(actor_prev,pf,pgrad,1000+step)
            vfd,van,verr=directional_check(critic_prev,vf,vgrad,2000+step)
            max_actor_fd=max(max_actor_fd,perr); max_critic_fd=max(max_critic_fd,verr)

            delta=actor_after.astype(np.float64)-actor_prev.astype(np.float64)
            grad_dot_delta=float(np.dot(pgrad,delta))
            if grad_dot_delta < 0: positive_descent+=1
            results["steps"].append({
                "step":step,
                "policy_loss_reference":ploss,
                "policy_loss_recorded":float(rec["policy_loss"]),
                "policy_loss_abs_error":policy_err,
                "value_loss_reference":vloss,
                "value_loss_recorded":float(rec["value_loss"]),
                "value_loss_abs_error":value_err,
                "post_mean_max_abs_error":mean_err,
                "ratio_latent_vs_action_max_abs":ratio_equiv,
                "actor_grad_norm":float(np.linalg.norm(pgrad)),
                "critic_grad_norm":float(np.linalg.norm(vgrad)),
                "actor_delta_norm":float(np.linalg.norm(delta)),
                "actor_grad_dot_actual_delta":grad_dot_delta,
                "actor_directional_fd":pfd,
                "actor_directional_analytic":pan,
                "actor_directional_rel_error":perr,
                "critic_directional_fd":vfd,
                "critic_directional_analytic":van,
                "critic_directional_rel_error":verr,
            })
            actor_prev=actor_after
            critic_prev=critic_after

        results["checks"]["optimizer_trace"]={
            "max_policy_loss_abs_error":max_policy_err,
            "max_value_loss_abs_error":max_value_err,
            "max_post_mean_abs_error":max_mean_err,
            "max_latent_vs_action_ratio_abs_error":max_ratio_equiv,
            "max_actor_directional_gradient_rel_error":max_actor_fd,
            "max_critic_directional_gradient_rel_error":max_critic_fd,
            "minibatches_with_negative_grad_dot_actual_delta":positive_descent,
            "minibatches":16,
        }

        # Full incoming/final batch objectives, independent from the logged per-minibatch scalars.
        incoming_full=policy_loss_and_grad(actor0,obs,lat,oldlp,adv)[0]
        final_full=policy_loss_and_grad(actor_prev,obs,lat,oldlp,adv)[0]
        results["checks"]["full_batch"]={
            "incoming_policy_loss":incoming_full,
            "final_policy_loss":final_full,
            "surrogate_gain":incoming_full-final_full,
            "final_actor_l2_displacement":float(np.linalg.norm(actor_prev.astype(np.float64)-actor0.astype(np.float64))),
        }

    # Classify. Tolerances are intentionally much tighter than controller-effect sizes.
    failures=[]
    if any(results["checks"]["primary_gae"].values()):
        failures.append("primary GAE/return reconstruction is not bit-exact")
    if results["checks"]["union_normalization"]["bit_mismatches"]:
        failures.append("union advantage normalization is not bit-exact")
    if results["checks"]["old_log_prob"]["max_abs"] > 2e-4:
        failures.append("incoming actor does not reproduce stored old log probabilities")
    o=results["checks"]["optimizer_trace"]
    if o["max_policy_loss_abs_error"] > 2e-5:
        failures.append("independent policy losses disagree with Rust")
    if o["max_value_loss_abs_error"] > 3e-4:
        failures.append("independent critic losses disagree with Rust")
    if o["max_post_mean_abs_error"] > 3e-5:
        failures.append("exported actor does not reproduce recorded Burn means")
    if o["max_latent_vs_action_ratio_abs_error"] > 5e-4:
        failures.append("action-space and latent PPO ratios materially disagree")
    if o["max_actor_directional_gradient_rel_error"] > 2e-4:
        failures.append("independent actor gradient fails finite-difference check")
    if o["max_critic_directional_gradient_rel_error"] > 2e-4:
        failures.append("independent critic gradient fails finite-difference check")

    results["classification"]="PASS" if not failures else "MISMATCH"
    results["failures"]=failures
    (out/"reference.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Independent PPO reference audit — update 4140",
        "",
        f"Classification: **{results['classification']}**",
        "",
        "This independently reconstructs the captured harmful update without Burn or an RL library.",
        "",
        "## Target and normalization checks",
        "",
        f"- Primary return bit mismatches: {results['checks']['primary_gae']['returns_bit_mismatches']}",
        f"- Primary raw-advantage bit mismatches: {results['checks']['primary_gae']['raw_advantage_bit_mismatches']}",
        f"- Primary normalized-advantage bit mismatches: {results['checks']['primary_gae']['normalized_bit_mismatches']}",
        f"- Union normalization bit mismatches: {results['checks']['union_normalization']['bit_mismatches']}",
        "",
        "## Policy/value implementation checks",
        "",
        f"- Max old-log-prob absolute error: {results['checks']['old_log_prob']['max_abs']:.9g}",
        f"- Max minibatch policy-loss absolute error: {o['max_policy_loss_abs_error']:.9g}",
        f"- Max minibatch value-loss absolute error: {o['max_value_loss_abs_error']:.9g}",
        f"- Max exported-actor vs recorded-mean absolute error: {o['max_post_mean_abs_error']:.9g}",
        f"- Max latent-ratio vs squashed-action-ratio absolute error: {o['max_latent_vs_action_ratio_abs_error']:.9g}",
        "",
        "## Independent gradient checks",
        "",
        f"- Max actor analytic-vs-finite-difference directional relative error: {o['max_actor_directional_gradient_rel_error']:.9g}",
        f"- Max critic analytic-vs-finite-difference directional relative error: {o['max_critic_directional_gradient_rel_error']:.9g}",
        f"- Actual actor Adam displacement has negative dot product with the current minibatch gradient in {o['minibatches_with_negative_grad_dot_actual_delta']}/16 minibatches.",
        "  This last quantity is descriptive only because Adam carries moments from previous minibatches/updates.",
        "",
        "## Full-batch update",
        "",
        f"- Incoming policy loss: {results['checks']['full_batch']['incoming_policy_loss']:.9g}",
        f"- Final policy loss: {results['checks']['full_batch']['final_policy_loss']:.9g}",
        f"- Independent surrogate gain: {results['checks']['full_batch']['surrogate_gain']:.9g}",
        "",
    ]
    if failures:
        lines += ["## Mismatches requiring investigation",""] + [f"- {x}" for x in failures] + [""]
    else:
        lines += [
            "## Interpretation",
            "",
            "The captured update's target recursion, normalization, transformed-policy likelihood, PPO clipped loss, critic loss, network serialization, and independent loss gradients are mutually consistent under the tested tolerances.",
            "",
            "This does **not** certify hidden Adam moments because the historical artifact did not export them. It narrows any remaining simple training-path bug toward optimizer-state handling, data-generation semantics not represented in the captured files, or higher-level objective/generalization behavior.",
            "",
        ]
    (out/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"classification":results["classification"],"failures":failures,"surrogate_gain":results["checks"]["full_batch"]["surrogate_gain"]},sort_keys=True))
    if failures:
        raise SystemExit(3)


if __name__=="__main__":
    main()
