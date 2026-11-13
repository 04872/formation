"""Paper Eqs. (7)-(17), with 2-D differential-drive inputs."""
import numpy as np


def goal_cost(center, goal, kg):
    return kg * np.linalg.norm(center-goal)


def shape_cost(a, mode, ks):
    # Uniform mode is the paper's scalar scale; affine uses a matrix norm.
    return ks * (abs(a[0,0]-1.) if mode == 'uniform' else np.linalg.norm(a-np.eye(2)))


def safe_cost(targets, regions, ksafe):
    return sum(0 if any(np.all(h@p <= b+1e-9) for h,b in regions) else ksafe for p in targets)


def risk_cost(targets, current, obstacle_distance, drisk):
    distances=np.maximum(obstacle_distance,0.)
    weights=np.where(distances>drisk,distances/(distances.sum()+1e-12),0.)
    return np.sum(weights*np.linalg.norm(targets-current,axis=1))


def continuity_cost(a, previous, angle, previous_angle, mode, ksc, kac):
    shape_change=abs(a[0,0]-previous[0,0]) if mode=='uniform' else np.linalg.norm(a-previous)
    turn=0. if previous_angle is None else abs(np.arctan2(np.sin(angle-previous_angle),np.cos(angle-previous_angle)))
    return ksc*shape_change+kac*turn


def dynamic_cost(v, acceleration, vmax, amax, weight=2700.):
    return weight*((np.sign(np.abs(v)-vmax)+np.sign(acceleration-amax))>=0)


def smoothness_cost(control, weight=1000.):
    return weight*np.sum(control*control,axis=-1)


def distance_penalty(distance, low, high, weight, exponent):
    return np.where(distance<low,weight,np.where(distance<=high,weight*(np.maximum(high-distance,0)/(high-low))**exponent,0.))


def guidance_cost(position, target, weight=2000.):
    return weight*np.linalg.norm(position-target,axis=-1)
