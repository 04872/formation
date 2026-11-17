"""Centralized C2 quintic Bézier optimization and differential-drive checks."""
from itertools import combinations
from math import comb
import time
import cvxpy as cp
import numpy as np
from numpy.polynomial import Polynomial
from .geometry import certify_edge
from .trajectory_template import TrajectoryTemplate, gram
from .trajectory_scp import AngularRefinement


def matrix(reference):
    """Return E_i with p_i = E_i q for centered references."""
    return np.array([[[1,0,r[0],r[1],0,0],[0,1,0,0,r[0],r[1]]] for r in reference], float)


def power_basis(control):
    """Convert Bézier coefficients to power-basis polynomials in u."""
    n=len(control)-1
    return [Polynomial([sum(control[k,j]*comb(n,k)*comb(n-k,m-k)*(-1)**(m-k)
                            for k in range(m+1)) for m in range(n+1)])
            for j in range(control.shape[1])]


def _minimum(poly):
    roots=poly.deriv().roots()
    u=[0.,1.]+[float(r.real) for r in roots if abs(r.imag)<1e-7 and 0<r.real<1]
    return min(float(poly(x)) for x in u)


def _bernstein_value(control, u):
    values=np.asarray(control,dtype=float).copy()
    while len(values)>1:
        values=(1.-u)*values[:-1]+u*values[1:]
    return values[0]


def differential_failure(control, duration, reference, omega_max):
    """Check omega over each full segment via polynomial stationary points.

    Check both signed polynomial inequalities; divide endpoint zero factors
    before evaluation to avoid absolute tolerances masking a zero-speed limit.
    Interior zero-speed points are rejected because heading can be undefined.
    """
    E=matrix(reference)
    for robot,e in enumerate(E):
        velocity=5*np.diff(control @ e.T,axis=0)
        # If the imposed endpoint velocity and acceleration vanish, p'(u)
        # has a known common u^2 or (1-u)^2 factor. Removing it before
        # forming speed squared avoids splitting multiple endpoint roots.
        if np.array_equal(control[0],control[1]) and np.array_equal(control[1],control[2]):
            start=2
        else:
            start=0
        if np.array_equal(control[5],control[4]) and np.array_equal(control[4],control[3]):
            end=2
        else:
            end=0
        degree=4-start-end
        reduced=np.array([velocity[i]*comb(4,i)/comb(degree,i-start)
                          for i in range(start,5-end)])
        if np.all(reduced==0.):
            continue
        vx,vy=power_basis(reduced)
        ax,ay=vx.deriv(),vy.deriv()
        speed2=vx*vx+vy*vy
        cross=vx*ay-vy*ax
        stationary=cross.deriv()*speed2-cross*speed2.deriv()
        points=[0.,1.]
        for poly in (stationary,speed2.deriv()):
            points.extend(float(r.real) for r in poly.roots()
                          if abs(r.imag)<1e-7 and 0<r.real<1)
        derivative=degree*np.diff(reduced,axis=0)
        for u in points:
            v=_bernstein_value(reduced,u)
            norm2=float(v@v)
            if norm2<=1e-14:
                return {'reason':'interior_near_zero_speed','robot':robot,'u':u,
                        'reduced_speed_squared':norm2}
            a=_bernstein_value(derivative,u) if degree else np.zeros(2)
            omega=float((v[0]*a[1]-v[1]*a[0])/(duration*norm2))
            if abs(omega)>omega_max:
                return {'reason':'angular_inequality','robot':robot,'u':u,'omega':omega}
    return None


def differential_certificate(control, duration, reference, omega_max):
    """Return whether the numerical whole-curve angular check succeeds."""
    return differential_failure(control,duration,reference,omega_max) is None


class TrajectoryOptimizer:
    """Bind one planning round to a reusable, parameterized cone program."""

    def __init__(self, route, regions, reference, initial_velocity, initial_acceleration,
                 mode, vmax=.65, amax=1., omega_max=1.5, template_cache=None,
                 terminal_goal=True, terminal_stop=False, reserve=False):
        started=time.perf_counter()
        self.route=route;self.regions=regions;self.reference=reference
        self.mode=mode;self.vmax=vmax;self.amax=amax
        self.initial_velocity=initial_velocity;self.initial_acceleration=initial_acceleration
        self.omega_max=omega_max;self.terminal_goal=terminal_goal
        self.terminal_stop=terminal_stop;self.reserve=reserve
        self._angular_refinement=None
        certs=[certify_edge(a,b,regions,reference) for a,b in zip(route[:-1],route[1:])]
        if not certs or any(c is None for c in certs):
            self.problem=None
            self.last_diagnostic={'stage':'edge_certificate'}
            return
        E=matrix(reference);pairs=list(combinations(range(len(reference)),2));self.pairs=pairs
        corridors=[(np.concatenate([regions[idx][0]@E[i] for i,idx in enumerate(indices)]),
                    np.concatenate([regions[idx][1] for idx in indices])) for indices,_ in certs]
        capacity=tuple(len(b) for _,b in corridors)
        key=(len(certs),capacity,mode,vmax,amax,terminal_goal,terminal_stop,reserve,'physical_integral_v1',E.shape,E.tobytes())
        cache={} if template_cache is None else template_cache
        self.template_cache_hit=key in cache
        if key not in cache:
            cache[key]=TrajectoryTemplate(len(certs),capacity,E,mode,vmax,amax,terminal_goal,terminal_stop,reserve)
        template=cache[key];self.template=template
        template.route.value=np.asarray(route)
        template.initial_velocity.value=initial_velocity
        template.initial_acceleration.value=initial_acceleration
        for k,((indices,normals),(H,bound)) in enumerate(zip(certs,corridors)):
            a,b=(route[k],route[k+1]) if k<len(route)-1 else (route[-2],route[-1])
            u,_,vh=np.linalg.svd(((a[2:]+b[2:])/2).reshape(2,2))
            template.rotations[k].value=u@vh
            template.corridors[k].value=H;template.bounds[k].value=bound
            template.pairs[k].value=np.array([normal@(E[i]-E[l]) for normal,(i,l) in zip(normals,pairs)])
        self.problem=template.problem;self.x=template.x;self.certs=certs;self.E=E
        self.inverse_duration=template.inv_t;self.inverse_duration_squared=template.inv_t2
        self.duration_parameter=template.duration;self.inverse_duration_cubed=template.inv_t3
        self.build_seconds=time.perf_counter()-started

    def solve(self, duration):
        if self.problem is None:
            return None
        route,regions,reference=self.route,self.regions,self.reference
        initial_velocity,initial_acceleration=self.initial_velocity,self.initial_acceleration
        omega_max=self.omega_max
        problem,x,certs,E=self.problem,self.x,self.certs,self.E
        m=len(certs)
        durations=np.broadcast_to(np.asarray(duration,dtype=float),(m,)).copy()
        self.duration_parameter.value=durations;self.inverse_duration_cubed.value=1./durations**3
        if not self.reserve:
            workspace=np.concatenate(self.E,axis=0)
            for k in range(len(self.certs)):
                tracking_reference=np.array([(1-j/5)*self.route[k]+(j/5)*self.route[k+1] for j in range(6)])
                self.template.tracking_linear[k].value=durations[k]*gram(5)@tracking_reference@workspace.T
                self.template.velocity_reference[k].value=workspace@(self.route[k+1]-self.route[k])/durations[k]
        self.inverse_duration.value=1./durations
        self.inverse_duration_squared.value=1./durations**2
        t0=time.perf_counter()
        data,chain,inverse=problem.get_problem_data('CLARABEL',enforce_dpp=True)
        # DPP retains parameter entries even when their current value is zero.
        # Remove those stored zeros before sparse factorization: the numerical
        # cone program then has the same sparsity as the uncached formulation.
        data=dict(data)
        data['A']=data['A'].copy()
        data['A'].eliminate_zeros()
        try:
            solution=chain.solve_via_data(problem,data,warm_start=False)
            problem.unpack_results(solution,chain,inverse)
        except cp.error.SolverError:
            self.last_diagnostic={'stage':'optimization_solver_error','status':'solver_error',
                'build_seconds':self.build_seconds,'template_cache_hit':self.template_cache_hit,
                'solve_wall_seconds':time.perf_counter()-t0,'solver_calls':1,
                'compile_seconds':float(problem.compilation_time or 0.),'solver_seconds':None}
            return None
        timing={'solver_calls':1,'build_seconds':self.build_seconds,'template_cache_hit':self.template_cache_hit,'solve_wall_seconds':time.perf_counter()-t0,
                'compile_seconds':float(problem.compilation_time or 0.),
                'solver_seconds':float(problem.solver_stats.solve_time or 0.)}
        if problem.status not in ('optimal','optimal_inaccurate'):
            self.last_diagnostic={'stage':'optimization','status':problem.status,**timing}
            return None
        def restore_equalities():
            controls=np.asarray(x.value).reshape(m,6,6).copy()
            controls[0,0]=route[0]
            controls[0,1]=controls[0,0]+initial_velocity*durations[0]/5
            controls[0,2]=initial_acceleration*durations[0]**2/20+2*controls[0,1]-controls[0,0]
            for k in range(1,m):
                previous=controls[k-1]
                controls[k,0]=previous[5]
                controls[k,1]=controls[k,0]+(previous[5]-previous[4])*durations[k]/durations[k-1]
                controls[k,2]=(previous[5]-2*previous[4]+previous[3])*(durations[k]/durations[k-1])**2+2*controls[k,1]-controls[k,0]
            if self.terminal_goal:
                controls[-1,5]=route[-1]
            if self.terminal_goal or self.terminal_stop:
                controls[-1,4]=controls[-1,5];controls[-1,3]=controls[-1,5]
            return controls
        controls=restore_equalities();scp_iterations=0
        if not all(differential_certificate(c,durations[k],reference,omega_max) for k,c in enumerate(controls)):
            scp_started=time.perf_counter();scp_solver_seconds=0.
            if self._angular_refinement is None:
                self._angular_refinement=AngularRefinement(problem,x,E,m,
                    np.all(initial_velocity==0.) and np.all(initial_acceleration==0.),
                    self.terminal_goal or self.terminal_stop,omega_max)
            refinement=self._angular_refinement
            refinement.extra.clear();solver_errors=0
            timing['scp_omega_target']=omega_max-.05
            timing['scp_initial_omega_limit']=omega_max
            for scp_iterations in range(1,4):
                violations=[(k,f) for k,c in enumerate(controls) if (f:=differential_failure(c,durations[k],reference,omega_max)) is not None]
                if any(f['reason']=='interior_near_zero_speed' for _,f in violations):
                    timing['scp_iterations']=scp_iterations-1
                    self.last_diagnostic={'stage':'angular_singularity','angular_failure':violations[0][1],**timing}
                    return None
                refinement.add_failures(violations)
                refinement.update(controls,durations)
                status=refinement.solve()
                timing['solver_calls']+=1
                solver_errors+=status=='solver_error'
                if status!='solver_error':
                    scp_solver_seconds+=float(refinement.problem.solver_stats.solve_time or 0.)
                if status not in ('optimal','optimal_inaccurate'):
                    timing['scp_wall_seconds']=time.perf_counter()-scp_started
                    timing['scp_solver_seconds']=scp_solver_seconds
                    timing['solve_wall_seconds']=time.perf_counter()-t0
                    self.last_diagnostic={'stage':'omega_scp_infeasible','scp_iterations':scp_iterations,
                                          'scp_solver_errors':solver_errors,
                                          'status':status,**timing}
                    return None
                controls=restore_equalities()
                if all(differential_certificate(c,durations[k],reference,omega_max) for k,c in enumerate(controls)):
                    break
            else:
                timing['scp_wall_seconds']=time.perf_counter()-scp_started
                timing['scp_solver_seconds']=scp_solver_seconds
                timing['solve_wall_seconds']=time.perf_counter()-t0
                failure=next(({'segment':k,**f} for k,c in enumerate(controls)
                              if (f:=differential_failure(c,durations[k],reference,omega_max)) is not None),None)
                self.last_diagnostic={'stage':'omega','angular_failure':failure,'scp_iterations':scp_iterations,
                                      'scp_solver_errors':solver_errors,
                                      'status':problem.status,**timing}
                return None
            timing['scp_wall_seconds']=time.perf_counter()-scp_started
            timing['scp_solver_seconds']=scp_solver_seconds
            timing['scp_solver_errors']=solver_errors
        timing['scp_iterations']=scp_iterations
        timing['solver_calls']=1+scp_iterations
        x.value=controls.reshape(-1,6)
        max_violation=max(float(np.max(np.asarray(c.violation()))) for c in problem.constraints)
        if max_violation>1e-6:
            self.last_diagnostic={'stage':'postcheck_constraints','max_violation':max_violation,**timing}
            return None
        timing['max_constraint_violation']=max_violation
        timing['solve_wall_seconds']=time.perf_counter()-t0
        segments=[]
        for k,(indices,normals) in enumerate(certs):
            H=np.concatenate([regions[idx][0]@E[i] for i,idx in enumerate(indices)])
            b=np.concatenate([regions[idx][1] for idx in indices])
            segments.append({'control_points':controls[k].tolist(),'duration':float(durations[k]),
                             'H':H.tolist(),'b':b.tolist(),'region_indices':indices.tolist(),
                         'pair_normals':normals.tolist(),
                         'role':'braking_reserve' if self.reserve else 'tracking'})
        self.last_diagnostic={'stage':'success','status':problem.status,**timing}
        return segments


def optimize(route, regions, reference, initial_velocity, initial_acceleration,
             mode, duration=5., vmax=.65, amax=1., omega_max=1.5):
    """One-shot interface; rolling planning reuses TrajectoryOptimizer instead."""
    optimizer=TrajectoryOptimizer(route,regions,reference,initial_velocity,
                                  initial_acceleration,mode,vmax,amax,omega_max)
    result=optimizer.solve(duration)
    optimize.last_diagnostic=optimizer.last_diagnostic
    return result
