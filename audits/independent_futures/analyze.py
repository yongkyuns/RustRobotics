#!/usr/bin/env python3
"""Independent score-gradient analysis for the frozen future-diversity diagnostic."""
from __future__ import annotations
import argparse,csv,io,json,math,zipfile
from pathlib import Path
import numpy as np

DIMS=((4,64),(64,64),(64,1))
SIGMA=0.1

def require(cond,msg):
    if not cond: raise AssertionError(msg)

def flat_network(data: bytes) -> np.ndarray:
    a=np.frombuffer(data,dtype="<f4").astype(np.float64)
    require(a.shape==(4545,) and np.isfinite(a).all(),"bad actor snapshot")
    return a

def layers(flat):
    off=0; out=[]
    for ni,no in DIMS:
        n=ni*no
        w=flat[off:off+n].reshape(ni,no); off+=n
        b=flat[off:off+no]; off+=no
        out.append((w,b))
    require(off==4545,"network layout")
    return out

def forward(flat,x,cache=False):
    (w1,b1),(w2,b2),(w3,b3)=layers(flat)
    x=np.asarray(x,float)
    z1=x@w1+b1; h1=np.maximum(z1,0)
    z2=h1@w2+b2; h2=np.maximum(z2,0)
    y=(h2@w3+b3)[:,0]
    return (y,(x,z1,h1,z2,h2,w1,w2,w3)) if cache else y

def grad_mu(flat,x,coeff):
    mu,c=forward(flat,x,True)
    x,z1,h1,z2,h2,w1,w2,w3=c
    coeff=np.asarray(coeff,float)
    d3=coeff[:,None]
    g3=h2.T@d3; gb3=d3.sum(0)
    d2=(d3@w3.T)*(z2>0); g2=h1.T@d2; gb2=d2.sum(0)
    d1=(d2@w2.T)*(z1>0); g1=x.T@d1; gb1=d1.sum(0)
    return np.concatenate([g1.ravel(),gb1,g2.ravel(),gb2,g3.ravel(),gb3])

def normalize(a):
    a=np.asarray(a,float)
    sd=max(a.std(),1e-6)
    return (a-a.mean())/sd

def score_gradient(flat,x,latent,adv):
    mu=forward(flat,x)
    coeff=np.asarray(adv,float)*(np.asarray(latent,float)-mu)/(SIGMA**2)/len(mu)
    return grad_mu(flat,x,coeff)

def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else float("nan")

def mean_pairwise(gs):
    vals=[]
    for i in range(len(gs)):
        for j in range(i):
            vals.append(cosine(gs[i],gs[j]))
    return float(np.mean(vals))

def mean_leave_one_out(gs):
    vals=[]
    total=np.sum(gs,axis=0)
    for i,g in enumerate(gs):
        ref=(total-g)/(len(gs)-1)
        vals.append(cosine(g,ref))
    return float(np.mean(vals))

def read_selected(path):
    rows=list(csv.DictReader(Path(path).read_text().splitlines()))
    require(len(rows)==32*5*512,f"unexpected selected rows {len(rows)}")
    return rows

def reference_gradient(zip_path,actor):
    with zipfile.ZipFile(zip_path) as z:
        union=json.loads(z.read("original/union.json"))
        credit=list(csv.DictReader(io.StringIO(z.read("paired-credit.csv").decode())))
    require(len(union["observations"])==1024 and len(credit)==8192,"paired source size")
    diffs=np.zeros((1024,8),float)
    for r in credit:
        row=int(r["row"]); draw=int(r["draw"])
        require(0<=row<1024 and 0<=draw<8,"paired index")
        diffs[row,draw]=float(r["difference"])
    adv=normalize(diffs.mean(axis=1))
    x=np.asarray(union["observations"],float)
    latent=np.asarray(union["latents"],float)
    return score_gradient(actor,x,latent,adv)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("selected")
    ap.add_argument("actor")
    ap.add_argument("paired")
    ap.add_argument("out")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    actor=flat_network(Path(args.actor).read_bytes())
    rows=read_selected(args.selected)
    ref=reference_gradient(args.paired,actor)
    require(np.isfinite(ref).all() and np.linalg.norm(ref)>0,"bad reference gradient")

    # Likelihood/inference consistency on every exported selected row.
    max_lp=0.0
    by={}
    for r in rows:
        key=(int(r["environments"]),int(r["replicate"]))
        by.setdefault(key,[]).append(r)
    require(len(by)==5*32,"group count")
    gradients={0.95:{},1.0:{}}
    path_parts={0.95:{},1.0:{}}
    raw_sds={0.95:{},1.0:{}}
    for key,group in by.items():
        group=sorted(group,key=lambda r:int(r["row"]))
        require([int(r["row"]) for r in group]==list(range(512)),"row order")
        x=np.asarray([[float(r[f"o{i}"]) for i in range(4)] for r in group])
        latent=np.asarray([float(r["latent"]) for r in group])
        old=np.asarray([float(r["old_log_prob"]) for r in group])
        mu=forward(actor,x)
        calc=-0.5*((latent-mu)/SIGMA)**2-math.log(SIGMA)-0.5*math.log(2*math.pi)
        max_lp=max(max_lp,float(np.max(np.abs(calc-old))))
        path=np.asarray([(int(r["stream"]),int(r["path"])) for r in group],dtype=int)
        for lam,suffix in [(0.95,"95"),(1.0,"1")]:
            raw=np.asarray([float(r[f"raw{suffix}"]) for r in group])
            adv=np.asarray([float(r[f"norm{suffix}"]) for r in group])
            require(np.max(np.abs(normalize(raw)-adv))<3e-5,"normalization mismatch")
            g=score_gradient(actor,x,latent,adv)
            # Exact physical-path mean/residual decomposition.
            mean_component=np.empty_like(adv)
            for ident in {tuple(v) for v in path}:
                mask=(path[:,0]==ident[0])&(path[:,1]==ident[1])
                mean_component[mask]=adv[mask].mean()
            within=adv-mean_component
            gm=score_gradient(actor,x,latent,mean_component)
            gw=score_gradient(actor,x,latent,within)
            require(np.max(np.abs(g-gm-gw))<2e-10,"gradient decomposition")
            gradients[lam][key]=g
            path_parts[lam][key]=(gm,gw,len({tuple(v) for v in path}))
            raw_sds[lam][key]=float(raw.std())

    records=[]
    for lam in [0.95,1.0]:
        for n in [1,2,4,8,16]:
            gs=np.stack([gradients[lam][(n,rep)] for rep in range(32)])
            mean=gs.mean(0)
            deviations=gs-mean
            rms_noise=float(np.sqrt(np.mean(np.sum(deviations*deviations,axis=1))))
            signal=float(np.linalg.norm(mean))
            mean_parts=[path_parts[lam][(n,rep)] for rep in range(32)]
            records.append(dict(
                gae_lambda=lam,
                environments=n,
                replicates=32,
                fitting_rows=512,
                simulated_rows_per_lambda_per_replicate=n*1536,
                mean_gradient_norm=signal,
                rms_gradient_deviation=rms_noise,
                signal_to_rms_noise=signal/rms_noise if rms_noise else float("inf"),
                mean_pairwise_cosine=mean_pairwise(gs),
                mean_leave_one_out_cosine=mean_leave_one_out(gs),
                mean_cosine_vs_paired_credit=float(np.mean([cosine(g,ref) for g in gs])),
                cosine_mean_vs_paired_credit=cosine(mean,ref),
                mean_trajectory_component_norm=float(np.mean([np.linalg.norm(v[0]) for v in mean_parts])),
                mean_within_trajectory_component_norm=float(np.mean([np.linalg.norm(v[1]) for v in mean_parts])),
                mean_physical_paths=float(np.mean([v[2] for v in mean_parts])),
                mean_raw_advantage_sd=float(np.mean([raw_sds[lam][(n,rep)] for rep in range(32)])),
            ))
    fields=list(records[0])
    with (out/"statistics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    summary={
        "classification":"diagnostic_only",
        "training_updates":0,
        "max_old_log_prob_abs_error":max_lp,
        "reference_gradient_norm":float(np.linalg.norm(ref)),
        "records":records,
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
