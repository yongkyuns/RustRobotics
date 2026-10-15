"""Preregistered rollout-reuse test. No outcome-dependent settings."""
from dataclasses import dataclass
from prior_schedule import learning_rate as old_rate
PROTOCOL=6194736723
SEEDS=(201,202,203,204)
ARMS=('constant10','decay10','decay4')
EPOCHS={'constant10':10,'decay10':10,'decay4':4}
CHECKPOINTS=(0,65536,262144,524288,786432,1048576,1310720)
TOTAL=1310720
ORDER={201:('constant10','decay10','decay4'),202:('decay10','decay4','constant10'),203:('decay4','constant10','decay10'),204:('constant10','decay4','decay10')}

def rate(t,arm):
    if arm not in ARMS:raise ValueError('Undeclared arm')
    return old_rate(t,'constant' if arm=='constant10' else 'decay')

def transactions(t,arm):
    if type(t) is not int or t<0 or t%2048 or arm not in ARMS:raise ValueError('Invalid checkpoint')
    return t//2048*32*EPOCHS[arm]

@dataclass
class AbsoluteSchedule:
    model:object
    arm:str
    def __call__(self,progress_remaining):
        import math
        if not math.isfinite(progress_remaining):raise ValueError('Nonfinite progress')
        return rate(int(self.model.num_timesteps),self.arm)
