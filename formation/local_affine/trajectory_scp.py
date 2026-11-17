"""Sequential convex angular-speed refinement for Bézier candidates."""
from math import comb
import cvxpy as cp
import clarabel
import numpy as np
from scipy import sparse
from types import SimpleNamespace
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL, dims_to_solver_cones


def _bernstein(n,u):
    return np.array([comb(n,k)*u**k*(1-u)**(n-k) for k in range(n+1)])


def _velocity_map(start_stationary,end_stationary):
    """Map six configuration controls to reduced velocity controls."""
    D=5*np.diff(np.eye(6),axis=0)
    if start_stationary:
        for _ in range(2):
            n=len(D)-1
            D=D[1:]*np.array([n/(i+1) for i in range(n)])[:,None]
    if end_stationary:
        for _ in range(2):
            n=len(D)-1
            D=D[:-1]*np.array([n/(n-i) for i in range(n)])[:,None]
    return D


class AngularRefinement:
    """Linearize sampled omega inequalities; whole-curve check remains required."""
    def __init__(self, problem, variable, E, segments, start_stationary, end_stationary,
                 omega_max, trust=.35):
        self.x=variable;self.E=E;self.segments=segments;self.omega_max=omega_max
        self.samples=np.linspace(0.,1.,9);self.maps=[];self.extra={}
        self.gradients=[];self.bounds=[];self.base_problem=problem
        self.trust=trust
        self.center=cp.Parameter(variable.shape)
        for k in range(segments):
            D=_velocity_map(start_stationary and k==0,end_stationary and k==segments-1)
            self.maps.append(D)
            rows=len(E)*(len(self.samples)+6)*2
            gradient=cp.Parameter((rows,36))
            bound=cp.Parameter(rows)
            self.gradients.append(gradient);self.bounds.append(bound)
        self.problem=SimpleNamespace(solver_stats=SimpleNamespace(solve_time=0.))

    def add_failures(self, failures):
        """Use the same violating positions returned by the final certificate."""
        for segment,failure in failures:
            u=failure['u'];robot=failure['robot']
            for sign in (-1.,1.):
                points=self.extra.setdefault((segment,robot,sign),[])
                if all(abs(u-p)>1e-7 for p in points) and len(points)<6:
                    points.append(u)

    def update(self, controls, duration):
        self.center.value=controls.reshape(-1,6)
        for k,P in enumerate(controls):
            D=self.maps[k];n=len(D)-1;rows=[];bounds=[]
            for i,e in enumerate(self.E):
                for sign in (-1.,1.):
                    extra=self.extra.get((k,i,sign),[])
                    for u in list(self.samples)+extra+[0.]*(6-len(extra)):
                        velocity_coeff=_bernstein(n,u)@D
                        deriv_coeff=(n*_bernstein(n-1,u)@np.diff(D,axis=0) if n else np.zeros(6))
                        qv=velocity_coeff@P;qa=deriv_coeff@P
                        v=e@qv;a=e@qa;cross=v[0]*a[1]-v[1]*a[0]
                        # The first endpoint is fixed by the actual current
                        # state; only later samples use an interior margin.
                        target=self.omega_max if k==0 and u==0 else self.omega_max-.05
                        f=target*duration[k]*(v@v)+sign*cross
                        d_v=2*target*duration[k]*v+sign*np.array([a[1],-a[0]])
                        d_a=sign*np.array([-v[1],v[0]])
                        grad=velocity_coeff[:,None]*(e.T@d_v)[None,:]+deriv_coeff[:,None]*(e.T@d_a)[None,:]
                        row=grad.ravel();bound=float(np.sum(grad*P)-f)
                        scale=np.linalg.norm(row)
                        if scale>0:
                            row=row/scale;bound=bound/scale
                        rows.append(row);bounds.append(bound)
            self.gradients[k].value=np.asarray(rows);self.bounds[k].value=np.asarray(bounds)

    def solve(self):
        # Insert linearized inequalities in the existing canonical cone data.
        # The underlying quadratic objective and original constraints are unchanged.
        data,_,_=self.base_problem.get_problem_data('CLARABEL',enforce_dpp=True)
        A=data['A'].tocsc();b=data['b'];q=data['c'];P=data.get('P')
        n=len(q); offset=data['param_prob'].var_id_to_col[self.x.id]
        nr,nc=self.x.shape
        extra_rows=[];extra_b=[]
        for k,(gradient,bound) in enumerate(zip(self.gradients,self.bounds)):
            g=gradient.value;rows,cols=np.nonzero(g)
            # Gradient columns are C-order (control point, configuration index),
            # whereas CVXPY's canonical variable uses Fortran order.
            variable_cols=offset+(6*k+cols//6)+nr*(cols%6)
            extra_rows.append(sparse.coo_matrix((-g[rows,cols],(rows,variable_cols)),shape=(len(g),n)).tocsr())
            extra_b.append(-bound.value)
        robot_map=np.concatenate(self.E,axis=0)
        trust_matrix=sparse.kron(sparse.csr_matrix(robot_map),sparse.eye(nr),format='csr')
        trust_matrix=sparse.hstack((sparse.csr_matrix((trust_matrix.shape[0],offset)),trust_matrix,
                                   sparse.csr_matrix((trust_matrix.shape[0],n-offset-nr*nc))),format='csr')
        center=(self.center.value@robot_map.T).reshape(-1,order='F')
        extra_rows.extend((trust_matrix,-trust_matrix));extra_b.extend((self.trust+center,self.trust-center))
        additional=sparse.vstack(extra_rows,format='csc')
        dims=data['dims'];split=dims.zero+dims.nonneg
        A=sparse.vstack((A[:split],additional,A[split:]),format='csc');A.eliminate_zeros()
        b=np.r_[b[:split],np.concatenate(extra_b),b[split:]]
        cones=[]
        if dims.zero: cones.append(clarabel.ZeroConeT(dims.zero))
        cones.append(clarabel.NonnegativeConeT(dims.nonneg+additional.shape[0]))
        original=dims_to_solver_cones(dims)
        cones.extend(original[int(dims.zero>0)+int(dims.nonneg>0):])
        settings=clarabel.DefaultSettings();settings.verbose=False
        target=self.problem
        target.solver_stats.solve_time=0.
        solver=clarabel.DefaultSolver(sparse.triu(P,format='csc'),q,A,b,cones,settings)
        solution=solver.solve()
        target.solver_stats.solve_time=solution.solve_time
        status=CLARABEL.STATUS_MAP.get(str(solution.status),'solver_error')
        if status in ('optimal','optimal_inaccurate'):
            values=np.asarray(solution.x)
            self.x.value=values[offset:offset+nr*nc].reshape((nr,nc),order='F')
        return status
