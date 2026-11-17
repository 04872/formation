"""Independent polynomial audit of centralized affine Bezier trajectories.

Extrema are evaluated at real stationary roots, independently of the planner's
Bernstein certificate. This is a floating-point numerical audit, not formal
interval arithmetic.
"""
from itertools import combinations
from math import comb
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.spatial import cKDTree


def power_polynomials(control):
    control = np.asarray(control, float)
    n = len(control) - 1
    u, one_minus_u = P([0, 1]), P([1, -1])
    return [sum((comb(n, k) * control[k, j] * u**k * one_minus_u**(n-k)
                 for k in range(n+1)), P([0])) for j in range(control.shape[1])]


def extrema(poly, low=0., high=1.):
    roots = poly.deriv().roots()
    points = [low, high] + [float(z.real) for z in roots
                          if abs(z.imag) < 1e-7 and low < z.real < high]
    values = poly(np.asarray(points))
    return float(np.min(values)), float(np.max(values))


def angular_extremum(robot_control, duration, interval=(0., 1.)):
    """Cancel stationary endpoint factors before evaluating angular velocity.

    v=f*w implies cross(v,v')/|v|^2=cross(w,w')/|w|^2.
    Factoring Bernstein coefficients avoids splitting repeated endpoint roots.
    """
    v = 5*np.diff(robot_control, axis=0)
    n = len(v)-1
    zero = np.linalg.norm(v, axis=1) < 1e-8
    if zero.all():
        return 0.
    first = next(i for i in range(n+1) if not zero[i])
    last = next(i for i in range(n, -1, -1) if not zero[i])
    degree = last-first
    reduced = np.array([v[i]*comb(n,i)/comb(degree,i-first)
                        for i in range(first,last+1)])
    vx, vy = power_polynomials(reduced)
    speed2 = vx*vx+vy*vy
    cross = vx*vy.deriv()-vy*vx.deriv()
    low, high = interval
    if extrema(speed2, low, high)[0] <= 1e-14:
        return float('inf')
    stationary = cross.deriv()*speed2-cross*speed2.deriv()
    points = [low, high] + [z.real for z in stationary.roots()
                          if abs(z.imag)<1e-7 and low<z.real<high]
    return max(abs(float(cross(t)/speed2(t)))/duration for t in points)


def stable_drive_rates(control, reference, duration):
    """Independent endpoint-factor cancellation for sampled input checks."""
    control=np.asarray(control)
    velocity=5*np.diff(control,axis=0)
    v=velocity[:,None,:2]+np.einsum('kij,nj->kni',velocity[:,2:].reshape(-1,2,2),reference)
    start=2 if np.array_equal(control[0],control[1]) and np.array_equal(control[1],control[2]) else 0
    end=2 if np.array_equal(control[-1],control[-2]) and np.array_equal(control[-2],control[-3]) else 0
    degree=4-start-end
    reduced=np.array([v[i]*comb(4,i)/comb(degree,i-start) for i in range(start,5-end)])
    rates=[]
    for i in range(len(reference)):
        vx,vy=power_polynomials(reduced[:,i,:])
        rates.append(((vx*vy.deriv()-vy*vx.deriv())/duration,vx*vx+vy*vy))
    return rates


def audit_segment(control, duration, reference, H, b, limits, interval=(0., 1.)):
    """Check a whole segment or the executed parameter interval."""
    q = power_polynomials(control)
    low, high = interval
    positions = [[q[0]+r[0]*q[2]+r[1]*q[3],
                  q[1]+r[0]*q[4]+r[1]*q[5]] for r in reference]
    max_speed = max_acceleration = max_omega = 0.
    omega_margin = float('inf')
    for p in positions:
        v = [x.deriv()/duration for x in p]
        a = [x.deriv(2)/duration**2 for x in p]
        speed2 = v[0]**2 + v[1]**2
        acceleration2 = a[0]**2 + a[1]**2
        cross = v[0]*a[1]-v[1]*a[0]
        max_speed = max(max_speed, np.sqrt(max(0., extrema(speed2, low, high)[1])))
        max_acceleration = max(max_acceleration, np.sqrt(max(0., extrema(acceleration2, low, high)[1])))
        for sign in (-1, 1):
            omega_margin = min(omega_margin, extrema(limits['omega_max']*speed2+sign*cross, low, high)[0])
    for r in reference:
        robot_control = np.asarray(control)[:, :2] + np.einsum(
            'kij,j->ki', np.asarray(control)[:, 2:].reshape(-1, 2, 2), r)
        max_omega = max(max_omega, angular_extremum(robot_control, duration, interval))
    pair_distance = float('inf')
    for i, j in combinations(range(len(positions)), 2):
        d = [positions[i][k]-positions[j][k] for k in range(2)]
        pair_distance = min(pair_distance, np.sqrt(max(0., extrema(d[0]**2+d[1]**2, low, high)[0])))
    H, b = np.asarray(H), np.asarray(b)
    violation = float(np.max(np.asarray(control) @ H.T-b))
    # Independent all-time singular-value checks through Gram-matrix minors.
    aa, ab, ac, ad = q[2:]
    gram00, gram11, gram01 = aa*aa+ac*ac, ab*ab+ad*ad, aa*ab+ac*ad
    shape_margin = float('inf')
    for bound, sign in ((limits.get('s_min', .42)**2, 1), (limits.get('s_max', 1.35)**2, -1)):
        m0, m1, m01 = sign*(gram00-bound), sign*(gram11-bound), sign*gram01
        for polynomial in (m0, m1, m0*m1-m01*m01):
            shape_margin = min(shape_margin, extrema(polynomial, low, high)[0])
    determinant_min = extrema(aa*ad-ab*ac, low, high)[0]
    max_shear = max(extrema(s*ab+t*ac, low, high)[1]
                    for s in (-1, 1) for t in (-1, 1))
    return {'max_speed': float(max_speed), 'max_acceleration': float(max_acceleration),
            'max_angular_speed': max_omega,
            'angular_polynomial_margin': omega_margin, 'minimum_pair_distance': float(pair_distance),
            'control_point_region_violation': violation, 'shape_psd_margin': shape_margin,
            'minimum_determinant': determinant_min,
            'max_shear': max_shear,
            'passed': bool(max_speed <= limits['v_max']+1e-6 and
                           max_acceleration <= limits['a_max']+1e-6 and
                           max_omega <= limits['omega_max']+1e-6 and
                           omega_margin >= -1e-7 and pair_distance >= limits.get('pair_min', .4)-1e-6 and
                           violation <= 1e-6 and shape_margin >= -1e-6 and determinant_min > 0 and
                           max_shear <= 1.5+1e-6)}


def obstacle_clearance_bound(control, reference, tree, resolution, radius=.16,
                             interval=(0., 1.), spatial_step=.005):
    """Cover the continuous curve using a derivative convex-hull Lipschitz bound."""
    control = np.asarray(control)
    reference = np.asarray(reference)
    robot_controls = control[:, None, :2] + np.einsum(
        'kij,nj->kni', control[:, 2:].reshape(-1, 2, 2), reference)
    derivative = (len(control)-1)*np.diff(robot_controls, axis=0)
    bound = np.linalg.norm(derivative, axis=-1).max()
    low, high = interval
    count = max(1, int(np.ceil((high-low)*bound/spatial_step)))
    # A negative coarse lower bound is inconclusive when every sample is safe.
    # Refine only those intervals; never turn a nonpositive bound into a pass.
    for _ in range(8):
        u = np.linspace(low, high, count+1)
        basis = np.array([comb(5, k)*u**k*(1-u)**(5-k) for k in range(6)]).T
        points = np.einsum('uk,kni->uni', basis, robot_controls)
        sampled_clearance = tree.query(points.reshape(-1, 2))[0].min()-resolution/np.sqrt(2)-radius
        lower_bound = sampled_clearance-bound*(high-low)/(2*count)
        if lower_bound > 0 or sampled_clearance <= 0:
            return float(lower_bound)
        count *= 2
    return float(lower_bound)


def audit_tree(nodes):
    """Verify cumulative costs and root connectivity after rewiring."""
    parent = [int(n['parent']) for n in nodes]
    residual = 0.
    for i, node in enumerate(nodes):
        if i == 0:
            if parent[i] != -1 or abs(node['cost']) > 1e-9:
                return {'passed': False, 'reason': 'invalid root'}
            continue
        if not 0 <= parent[i] < len(nodes):
            return {'passed': False, 'reason': 'invalid parent'}
        residual = max(residual, abs(node['cost']-nodes[parent[i]]['cost']-node['edge_cost']))
        seen, j = set(), i
        while j != 0:
            if j in seen or not 0 <= j < len(nodes):
                return {'passed': False, 'reason': 'cycle or disconnected node'}
            seen.add(j)
            j = parent[j]
    return {'passed': residual < 1e-7, 'maximum_cost_residual': residual,
            'nodes': len(nodes)}


def audit(directory):
    directory = Path(directory)
    # NpzFile decompresses an array on every __getitem__. Load audit inputs once
    # before the per-tick loop; high-rate traces otherwise incur quadratic I/O.
    with np.load(directory/'trajectory.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in ('reference','resolution','robot_radius',
                'v_max','a_max','omega_max','occupancy','known_history','dt',
                'states','controls','configurations')}
    records = json.loads((directory/'plans.json').read_text())
    reference = data['reference']
    resolution = float(data['resolution'])
    radius = float(data['robot_radius'])
    limits = {k: float(data[k]) for k in ('v_max', 'a_max', 'omega_max')}
    checks, tree_checks = [], []
    clearance = float('inf')
    join_error = np.zeros(3)
    for plan in records['plans']:
        if 'tree' in plan:
            tree_checks.append(audit_tree(plan['tree']))
        segments = plan.get('segments', [])
        if not segments:
            continue
        blocked = data['occupancy'] | ~data['known_history'][plan['known_index']]
        tree = cKDTree((np.argwhere(blocked)+.5)*resolution)
        for k, segment in enumerate(segments):
            c = np.asarray(segment['control_points'])
            duration = segment['duration']
            check = audit_segment(c, duration, reference, segment['H'], segment['b'], limits)
            check['plan'] = len(tree_checks)-1
            check['segment'] = k
            checks.append(check)
            clearance = min(clearance, obstacle_clearance_bound(c, reference, tree, resolution, radius))
            if k:
                previous = segments[k-1]
                before = power_polynomials(previous['control_points'])
                after = power_polynomials(c)
                for order in range(3):
                    residual = [abs(a.deriv(order)(0)/duration**order-
                                    b.deriv(order)(1)/previous['duration']**order)
                                for a,b in zip(after,before)]
                    join_error[order] = max(join_error[order], max(residual))
    execution_error = 0.
    heading_error = 0.
    stationary_heading_error = 0.
    turn_error = 0.
    replan_join_error = np.zeros(2)
    previous_derivatives = [np.zeros(6), np.zeros(6)]
    duration_error = 0.
    control_error = 0.
    polynomial_cache = {}
    for event in records['execution']:
        tick = event['tick']
        if event['kind'] == 'curve':
            segment = records['plans'][event['plan']]['segments'][event['segment']]
            key = (event['plan'], event['segment'])
            if key not in polynomial_cache:
                pp = power_polynomials(segment['control_points'])
                polynomial_cache[key] = (pp, [p.deriv()/segment['duration'] for p in pp],
                                         [p.deriv(2)/segment['duration']**2 for p in pp],
                                         stable_drive_rates(segment['control_points'],reference,segment['duration']))
            polynomials, first_derivatives, second_derivatives, drive_rates = polynomial_cache[key]
            duration_error = max(duration_error, abs((event['u1']-event['u0'])*segment['duration']-float(data['dt'])))
            for order in (1,2):
                derivatives = (first_derivatives,second_derivatives)[order-1]
                initial = np.array([p(event['u0']) for p in derivatives])
                replan_join_error[order-1] = max(replan_join_error[order-1], float(np.max(abs(initial-previous_derivatives[order-1]))))
                previous_derivatives[order-1] = np.array([p(event['u1']) for p in derivatives])
            for offset, u in ((0,event['u0']), (1,event['u1'])):
                expected = np.array([p(u) for p in polynomials])
                execution_error = max(execution_error, float(np.abs(expected-data['configurations'][tick+offset]).max()))
                velocity = np.array([p(u) for p in first_derivatives])
                rv = velocity[:2]+reference@velocity[2:].reshape(2,2).T
                yaw = data['states'][tick+offset,:,2]
                speed = np.linalg.norm(rv,axis=1)
                moving = speed > 1e-7
                lateral = -rv[:,0]*np.sin(yaw)+rv[:,1]*np.cos(yaw)
                if moving.any():
                    heading_error = max(heading_error, float(np.max(abs(lateral[moving])/speed[moving])))
                    forward = rv[:,0]*np.cos(yaw)+rv[:,1]*np.sin(yaw)
                    heading_error = max(heading_error, float(np.max(abs(forward[moving]/speed[moving]-1))))
                if offset == 1:
                    expected_command = np.zeros((len(reference),2))
                    expected_command[:,0] = speed
                    # At 50 Hz, expanded speed-squared polynomials lose precision
                    # near stationary endpoints. Evaluate the cancelled ratio.
                    for i,(numerator,denominator) in enumerate(drive_rates):
                        if denominator(u)>1e-24:
                            expected_command[i,1]=numerator(u)/denominator(u)
                    endpoint_control=np.asarray(segment['control_points'])
                    endpoint_is_stationary=(np.array_equal(endpoint_control[-1],endpoint_control[-2]) and
                                            np.array_equal(endpoint_control[-2],endpoint_control[-3]))
                    if endpoint_is_stationary and u>=1.-1e-10:
                        expected_command[:]=0.
                    control_error = max(control_error,float(np.max(abs(expected_command-data['controls'][tick]))))
                if u == 0. or u == 1.:
                    control = np.asarray(segment['control_points'])
                    robot_control = control[:,None,:2]+np.einsum('kij,nj->kni',control[:,2:].reshape(6,2,2),reference)
                    for i in np.flatnonzero(~moving):
                        vectors = robot_control[1:,i]-robot_control[0,i] if u == 0. else robot_control[-1,i]-robot_control[-2::-1,i]
                        nonzero = np.flatnonzero(np.linalg.norm(vectors,axis=1)>1e-7)
                        if len(nonzero):
                            direction = vectors[nonzero[0]]
                            expected_yaw = np.arctan2(direction[1],direction[0])
                            error = abs(np.arctan2(np.sin(expected_yaw-yaw[i]),np.cos(expected_yaw-yaw[i])))
                            stationary_heading_error = max(stationary_heading_error,float(error))
        elif event['kind'] == 'turn':
            before, after = data['states'][tick:tick+2]
            turn_error = max(turn_error,float(np.max(abs(after[:,:2]-before[:,:2]))))
            omega = np.asarray(event['omega'])
            control_error = max(control_error,float(np.max(abs(data['controls'][tick,0:,0]))),
                                float(np.max(abs(data['controls'][tick,:,1]-omega))))
            turn_error = max(turn_error, float(np.max(abs(after[:,2]-before[:,2]-float(data['dt'])*omega))))
            if np.max(abs(omega)) > limits['omega_max']+1e-8:
                turn_error = float('inf')
        if event['kind'] != 'curve':
            for order in (1,2):
                replan_join_error[order-1] = max(replan_join_error[order-1], float(np.max(abs(previous_derivatives[order-1]))))
                previous_derivatives[order-1] = np.zeros(6)
    result = {'segments': len(checks), 'trees': len(tree_checks),
              'all_segment_checks_passed': bool(checks and all(c['passed'] for c in checks)),
              'all_tree_checks_passed': bool(tree_checks and all(c['passed'] for c in tree_checks)),
              'known_grid_clearance_lower_bound': clearance,
              'maximum_C0_C1_C2_residuals': join_error.tolist(),
              'execution_C1_C2_join_residuals': replan_join_error.tolist(),
              'execution_tick_duration_residual': duration_error,
              'saved_control_residual': control_error,
              'final_configuration_speed_norm': float(np.linalg.norm(previous_derivatives[0])),
              'execution_configuration_residual': execution_error,
              'moving_heading_lateral_ratio': heading_error, 'in_place_turn_residual': turn_error,
              'stationary_boundary_heading_error': stationary_heading_error,
              'max_speed': max((c['max_speed'] for c in checks),default=0),
              'max_acceleration': max((c['max_acceleration'] for c in checks),default=0),
              'max_angular_speed': max((c['max_angular_speed'] for c in checks),default=0),
              'minimum_pair_distance': min((c['minimum_pair_distance'] for c in checks),default=0),
              'failed_segments': [c for c in checks if not c['passed']],
              'note': 'Independent floating-point polynomial extrema and Lipschitz grid bounds; analytic flat-output execution, not a motor/controller tracking experiment.'}
    result['passed'] = bool(result['all_segment_checks_passed'] and result['all_tree_checks_passed'] and
                            clearance > 0 and join_error.max()<1e-6 and execution_error<1e-6 and
                            replan_join_error.max()<1e-6 and duration_error<1e-6 and
                            control_error<1e-6 and heading_error<1e-6 and stationary_heading_error<1e-5 and turn_error<1e-7)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
