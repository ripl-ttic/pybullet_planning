#!/usr/bin/env python3
import functools
import pybullet as p
from pybullet_planning.utils import CLIENT


def preserve_state(func):
    @functools.wraps(func)
    def wrapper_preserve_pos_and_vel(*args, **kwargs):
        state_id = p.saveState(physicsClientId=CLIENT)
        ret = func(*args, **kwargs)
        p.restoreState(stateId=state_id, physicsClientId=CLIENT)
        return ret
    return wrapper_preserve_pos_and_vel
