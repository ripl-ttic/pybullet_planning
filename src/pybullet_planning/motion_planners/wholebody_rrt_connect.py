import time

from itertools import takewhile
from .smoothing import wholebody_smooth_path
from .rrt import TreeNode, configs, extract_ik_solutions
from .utils import irange, argmin, RRT_ITERATIONS, RRT_RESTARTS, RRT_SMOOTHING, INF, elapsed_time, negate

__all__ = [
    'wholebody_rrt_connect',
    'wholebody_birrt',
    'wholebody_direct_path',
    ]

def asymmetric_extend(q1, q2, extend_fn, backward=False):
    """directional extend_fn
    """
    if backward:
        return reversed(list(extend_fn(q2, q1)))
    return extend_fn(q1, q2)

def wholebody_extend_towards(tree, target, distance_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik, swap, tree_frequency):
    import functools
    import numpy as np
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision
    last = argmin(lambda n: distance_fn(n.config, target), tree)
    extend = list(asymmetric_extend(last.config, target, extend_fn, swap))
    # safe = list(takewhile(negate(collision_fn), extend))
    safe = []

    # for each pose in extend, it checks if any IK solution exists that is not in collison.
    # while such solution exist, it keeps appending the cube_pose to 'safe'
    # once it reaches a point where no such solution exist, it exits the loop.
    ik_solutions = []
    for cube_pose in extend:
        tip_positions = calc_tippos_fn(cube_pose)
        ik_sols = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, cube_pose),
                                                    sample_joint_conf_fn, tip_positions, num_samples=3)
        if len(ik_sols) == 0:
            break
        ik_solutions.append(ik_sols)
        safe.append(cube_pose)

    # print('safe length', len(safe))  # DEBUG

    # add the sequence of safe nodes to the Tree.
    # each node in safe has a corresponding set of IK solutions.
    # We create nodes for each of those IK solutions, and regard them as in a same group.
    for i, q in enumerate(safe):
        if (i % tree_frequency == 0) or (i == len(safe) - 1):
            # append node for every ik solution
            group = []
            for ik_sol in ik_solutions[i]:
                # find the argmin over ik solutions in the parent group
                ik_last = argmin(lambda node: distance_fn(node.ik_solution, ik_sol), last.group)
                last = TreeNode(q, parent=ik_last, ik_solution=ik_sol)
                group.append(last)
                tree.append(last)

            # add the nodes in the same group to node.group
            for node in group:
                node.group = group

    success = len(extend) == len(safe)
    return last, success


def wholebody_rrt_connect(q1, q2, init_joint_conf, end_joint_conf, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                          iterations=RRT_ITERATIONS, tree_frequency=1, max_time=INF):
    """[summary]

    Parameters
    ----------
    q1 : [type]
        [description]
    q2 : [type]
        [description]
    distance_fn : [type]
        [description]
    sample_fn : [type]
        [description]
    extend_fn : [type]
        [description]
    collision_fn : [type]
        [description]
    iterations : [type], optional
        [description], by default RRT_ITERATIONS
    tree_frequency : int, optional
        the frequency of adding tree nodes when extending. For example, if tree_freq=2, then a tree node is added every three nodes,
        by default 1
    max_time : [type], optional
        [description], by default INF

    Returns
    -------
    [type]
        [description]
    """
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision

    start_time = time.time()
    assert tree_frequency >= 1

    # create a node for each configuration
    tip_positions1 = calc_tippos_fn(q1)
    # ik_solutions1 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q1),
    #                                                   sample_joint_conf_fn, tip_positions1, num_samples=3)

    tip_positions2 = calc_tippos_fn(q2)
    # ik_solutions2 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q2),
    #                                                   sample_joint_conf_fn, tip_positions2, num_samples=3)

    nodes1, nodes2 = [TreeNode(q1, ik_solution=init_joint_conf)], [TreeNode(q2, ik_solution=end_joint_conf)]
    for iteration in irange(iterations):
        if max_time <= elapsed_time(start_time):
            break
        swap = len(nodes1) > len(nodes2)
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1
        sample = sample_fn()
        last1, _ = wholebody_extend_towards(tree1, sample_fn(), distance_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                                  swap, tree_frequency)
        last2, success = wholebody_extend_towards(tree2, last1.config, distance_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                                        not swap, tree_frequency)

        if success:
             path1, joint_conf_path1 = last1.retrace_all()
             path2, joint_conf_path2 = last2.retrace_all()
             if swap:
                 path1, path2 = path2, path1
                 joint_conf_path2 = joint_conf_path2
             entire_path = path1[:-1] + path2[::-1]
             return configs(entire_path), extract_ik_solutions(entire_path)
    return None, None


def wholebody_direct_path(q1, q2, init_joint_conf, end_joint_conf, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik):
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision
    # TEMP
    # if collision_fn(q1) or collision_fn(q2):
    #     return None
    path = [q1]
    joint_conf_path = [init_joint_conf]
    for q in extend_fn(q1, q2):
        tip_positions = calc_tippos_fn(q)
        if (q == q2).all():
            ik_solutions = [end_joint_conf]
        else:
            ik_solutions = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q, diagnosis=False),
                                                            sample_joint_conf_fn, tip_positions, num_samples=3)
        if len(ik_solutions) == 0:
            return None, None
        else:
            ik_solution = ik_solutions[0]
            if collision_fn(q, joint_conf=ik_solution):
                return None, None

        path.append(q)
        joint_conf_path.append(ik_solution)
    print('DIRECT PATH is found!!')
    return path, joint_conf_path


def wholebody_birrt(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                    init_joint_conf=None, restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING, max_time=INF, **kwargs):
    import functools
    from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision

    # collision and IK check on initial and end configuration
    tip_positions1 = calc_tippos_fn(q1)
    if init_joint_conf is None:
        ik_solutions1 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q1, diagnosis=False),
                                                        sample_joint_conf_fn, tip_positions1, num_samples=3)
    else:
        ik_solutions1 = [] if collision_fn(q1, init_joint_conf) else [init_joint_conf]

    if len(ik_solutions1) == 0:
        print('Initial Configuration in collision')
        return None, None

    tip_positions2 = calc_tippos_fn(q2)
    ik_solutions2 = sample_multiple_ik_with_collision(ik, functools.partial(collision_fn, q2, diagnosis=False),
                                                      sample_joint_conf_fn, tip_positions2, num_samples=3)
    if len(ik_solutions2) == 0:
        print('End Configuration in collision')
        return None, None

    ik_sol1 = ik_solutions1[0]
    ik_sol2 = ik_solutions2[0]

    start_time = time.time()
    path, joint_conf_path = wholebody_direct_path(q1, q2, ik_sol1, ik_sol2, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik)
    if path is not None:
        return path, joint_conf_path

    for _ in irange(restarts + 1):
        if max_time <= elapsed_time(start_time):
            break
        path, joint_conf_path = wholebody_rrt_connect(q1, q2, ik_sol1, ik_sol2, distance_fn, sample_fn, extend_fn, collision_fn, calc_tippos_fn, sample_joint_conf_fn, ik,
                                                      max_time=max_time - elapsed_time(start_time), **kwargs)
        if path is not None:
            if smooth is None:
                return path, joint_conf_path
            return wholebody_smooth_path(path, joint_conf_path, extend_fn, collision_fn, ik, calc_tippos_fn, sample_joint_conf_fn, iterations=smooth)
    return None, None
