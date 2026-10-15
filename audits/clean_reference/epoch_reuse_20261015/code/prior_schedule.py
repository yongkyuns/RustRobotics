"""Absolute-interaction PPO schedule. Protocol6183008716; no reward feedback."""
from dataclasses import dataclass
import math

INITIAL=3e-4
FLOOR=3e-5
DECAY_STEPS=1_048_576
TOTAL_STEPS=1_310_720
CHECKPOINTS=(0,65_536,262_144,524_288,786_432,1_048_576,1_310_720)
SEEDS=(201,202,203,204)
ARMS=('constant','decay')

def learning_rate(steps:int,arm:str)->float:
    if type(steps) is not int or steps<0: raise ValueError('Nonnegative integer training steps required')
    if arm not in ARMS: raise ValueError('Undeclared arm')
    if arm=='constant': return INITIAL
    if steps>=DECAY_STEPS: return FLOOR
    return INITIAL*(1-.9*steps/DECAY_STEPS)

@dataclass
class AbsoluteSchedule:
    """SB3 callback argument is intentionally ignored: checkpoint learn calls reset it."""
    model: object
    arm: str
    def __call__(self, progress_remaining:float)->float:
        if not math.isfinite(progress_remaining):raise ValueError('Nonfinite progress')
        return learning_rate(int(self.model.num_timesteps),self.arm)
