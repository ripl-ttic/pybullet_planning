#!/usr/bin/env python3
import functools
import pybullet as p
from pybullet_planning.utils import CLIENT
from pybullet_planning.interfaces.robots.joint import get_joint_positions, get_joint_velocities, set_joint_positions_and_velocities
from pybullet_planning.interfaces.robots.body import get_pose, get_velocity, set_pose, set_velocity


# def preserve_state(func):
#     @functools.wraps(func)
#     def wrapper_preserve_pos_and_vel(*args, **kwargs):
#         state_id = p.saveState(physicsClientId=CLIENT)
#         ret = func(*args, **kwargs)
#         p.restoreState(stateId=state_id, physicsClientId=CLIENT)
#         return ret
#     return wrapper_preserve_pos_and_vel

def preserve_state(func):
    @functools.wraps(func)
    def wrapper_preserve_state(body, joints, *args, **kwargs):
        # save initial pose and velocities
        joint_pos = get_joint_positions(body, joints)
        joint_vel = get_joint_velocities(body, joints)
        # state_id = p.saveState(physicsClientId=CLIENT)

        ret = func(body, joints, *args, **kwargs)

        # put positions and velocities back!
        # p.restoreState(stateId=state_id, physicsClientId=CLIENT)
        set_joint_positions_and_velocities(body, joints, joint_pos, joint_vel)
        return ret
    return wrapper_preserve_state

def preserve_state_wholebody(func):
    @functools.wraps(func)
    def wrapper_preserve_state_wholebody(cube_body, joints, finger_body, finger_joints, *args, **kwargs):
        # save initial pose and velocities
        cube_pose = get_pose(cube_body)
        cube_vel = get_velocity(cube_body)
        joint_pos = get_joint_positions(finger_body, finger_joints)
        joint_vel = get_joint_velocities(finger_body, finger_joints)
        # state_id = p.saveState(physicsClientId=CLIENT)

        ret = func(cube_body, joints, finger_body, finger_joints, *args, **kwargs)

        # put positions and velocities back!
        # p.restoreState(stateId=state_id, physicsClientId=CLIENT)
        set_pose(cube_body, cube_pose)
        set_velocity(cube_body, linear=cube_vel[0], angular=cube_vel[1])
        set_joint_positions_and_velocities(finger_body, finger_joints, joint_pos, joint_vel)
        return ret
    return wrapper_preserve_state_wholebody
