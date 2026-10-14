"""Fixed temporal interventions in issue35/comment6176905839. No learning."""
from __future__ import annotations
HORIZON = 1500
PROTOCOL = 6176905839
SCHEDULES = ('OLD','NEW','FIRST1','EARLY64','LATE64','EARLY128','LATE128')
NEW_SCHEDULES = SCHEDULES[2:]

def policy_at(schedule: str, tick: int) -> int:
    if schedule not in SCHEDULES:
        raise ValueError(f'Unknown schedule: {schedule}')
    if type(tick) is not int or not 0 <= tick < HORIZON:
        raise ValueError(f'Invalid action tick: {tick}')
    if schedule == 'OLD': return 410
    if schedule == 'NEW': return 411
    if schedule == 'FIRST1': return 411 if tick == 0 else 410
    if schedule.startswith('EARLY'):
        return 411 if tick < int(schedule[5:]) else 410
    return 410 if tick < int(schedule[4:]) else 411

def prefix_reference(schedule: str) -> tuple[int,int]:
    """(original array arm, exact guaranteed prefix length)."""
    return {'OLD':(0,1500),'NEW':(1,1500),'FIRST1':(1,1),
            'EARLY64':(1,64),'LATE64':(0,64),
            'EARLY128':(1,128),'LATE128':(0,128)}[schedule]
