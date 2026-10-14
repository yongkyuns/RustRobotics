"""Read-only coverage of an archived PPO batch; no training or native calls."""
import numpy as np

def need(ok, message):
    if not ok:
        raise AssertionError(message)

def batch_coverage(b):
    r=b['packets'];need(r.shape==(2048,22) and r.dtype==np.float32 and np.isfinite(r).all(),'batch packet schema')
    n=len(r);age=r[:,19].astype(int);need(np.array_equal(age,r[:,19]) and np.all((age>=0)&(age<256)),'packet age integers')
    term=r[:,20];timeout=r[:,21];need(np.isin(term,[0.,1.]).all() and np.isin(timeout,[0.,1.]).all() and not np.any(term*timeout),'boundary flags')
    flags=(term+timeout)>0
    need(np.array_equal(b['observations'].reshape(n,4),r[:,4:8]),'original observations')
    need(np.array_equal(np.clip(b['actions'].reshape(n),-1,1),r[:,16]),'original actions')
    for i in range(1,n):
        need(bool(b['episode_starts'][i,0])==bool(flags[i-1]),'recorded reset boundary')
        need(age[i]==(0 if flags[i-1] else age[i-1]+1),'recorded age continuity')
        if not flags[i-1]:need(r[i,:8].tobytes()==r[i-1,8:16].tobytes(),'recorded transition continuity')
    need(np.all(age[timeout>0]==255),'timeout age')
    ends=np.unique(np.r_[np.flatnonzero(flags),n-1]);idx=np.arange(n);stop=ends[np.searchsorted(ends,idx)];remain=stop-idx+1
    outward=(abs(r[:,2])>=.15)&(abs(r[:,3])>=.25)&(r[:,2]*r[:,3]>0)&(age<100)
    early=age<100
    return dict(rows=n,true_terminals=int(term.sum()),episode_timeouts=int(timeout.sum()),live_rollout_cutoffs=int(not flags[-1]),
       boundaries=[dict(row=int(i),pre_action_age=int(age[i]),kind='terminal' if term[i] else 'timeout' if timeout[i] else 'rollout',post_state=r[i,8:12].tolist()) for i in ends],
       early_outward_rows=np.flatnonzero(outward).tolist(),early_outward_count=int(outward.sum()),early_outward_remaining_min=int(remain[outward].min()),early_outward_remaining_max=int(remain[outward].max()),
       early_outward_remaining=remain[outward].tolist(),early_age_count=int(early.sum()),
       support=[dict(cutoff=k,all_rows_at_most=int((remain<=k).sum()),early_outward_at_most=int(((remain<=k)&outward).sum()),early_age_at_most=int(((remain<=k)&early).sum())) for k in [1,16,32,64,128,256]],
       limitation='Pre-existing instantaneous outward stratum is not all recovery/braking states. No calibration or gradient conclusion follows from support counts alone.')

