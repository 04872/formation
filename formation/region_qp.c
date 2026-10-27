/* Small convex programs in the lifted chart z = (x, y, rho * theta), solved by a primal-dual interior-point
 * method (Boyd & Vandenberghe, Sec. 11.7) on smooth convex constraints f_i(w) <= 0:
 *
 *   halfspaces  a_k^T z - b_k + tau                         <= 0
 *   cones       sqrt(|z_xy - c_xy|^2 + eps^2) - (g - s (z_t - c_t)) + tau <= 0
 *
 * tau (a fourth variable) exists only in phase I, which maximises the common margin; its start (the first cone's
 * seed with tau below every slack) is strictly feasible.  Phase II minimises 1/2 |z - p|^2 from a phase I point
 * with tau > 0.
 *
 * region_qp_project: 1 and z on success, 0 if no point has tau > 0, -1 on numerical failure.
 * region_qp_margin:  1 and (z, tau*) with the maximal margin tau*, -1 on numerical failure.
 */
#include <math.h>

#define MAXV 4
#define MAXC 128
#define CONE_EPS 1e-12

long region_qp_newton = 0, region_qp_backtracks = 0;
static double last_eta = 0.0, last_dual = 0.0;

typedef struct {
    const double *a, *b, *cones, *p;
    int m, q, nvar; /* nvar = 4 in phase I (tau last), 3 in phase II */
} Problem;

/* f_i, its gradient (nvar) and, for cones, the xy Hessian block (hxx, hxy, hyy). */
static double constraint(const Problem *P, const double *w, int i, double *g, double *h) {
    const int phase_one = P->nvar == 4;
    const double tau = phase_one ? w[3] : 0.0;
    if (i < P->m) {
        const double *a = P->a + 3 * i;
        if (g) {
            g[0] = a[0];
            g[1] = a[1];
            g[2] = a[2];
            if (phase_one) g[3] = 1.0;
        }
        if (h) h[0] = h[1] = h[2] = 0.0;
        return a[0] * w[0] + a[1] * w[1] + a[2] * w[2] - P->b[i] + tau;
    }
    const double *c = P->cones + 5 * (i - P->m); /* cx, cy, ct, s, g */
    const double vx = w[0] - c[0], vy = w[1] - c[1];
    const double r = sqrt(vx * vx + vy * vy + CONE_EPS * CONE_EPS);
    if (g) {
        g[0] = vx / r;
        g[1] = vy / r;
        g[2] = c[3];
        if (phase_one) g[3] = 1.0;
    }
    if (h) {
        h[0] = (1.0 - vx * vx / (r * r)) / r;
        h[1] = -vx * vy / (r * r * r);
        h[2] = (1.0 - vy * vy / (r * r)) / r;
    }
    return r - (c[4] - c[3] * (w[2] - c[2])) + tau;
}

static void objective_grad(const Problem *P, const double *w, double *g) {
    if (P->nvar == 4) {
        g[0] = g[1] = g[2] = 0.0;
        g[3] = -1.0;
    } else {
        for (int i = 0; i < 3; ++i) g[i] = w[i] - P->p[i];
    }
}

static int all_strict(const Problem *P, const double *w) {
    for (int i = 0; i < P->m + P->q; ++i)
        if (!(constraint(P, w, i, 0, 0) < 0.0)) return 0;
    return 1;
}

static int cholesky_solve(double *h, double *r, int n) {
    for (int j = 0; j < n; ++j) {
        double d = h[j * n + j];
        for (int k = 0; k < j; ++k) d -= h[j * n + k] * h[j * n + k];
        if (!(d > 0.0)) return 0;
        d = sqrt(d);
        h[j * n + j] = d;
        for (int i = j + 1; i < n; ++i) {
            double v = h[i * n + j];
            for (int k = 0; k < j; ++k) v -= h[i * n + k] * h[j * n + k];
            h[i * n + j] = v / d;
        }
    }
    for (int i = 0; i < n; ++i) {
        double v = r[i];
        for (int k = 0; k < i; ++k) v -= h[i * n + k] * r[k];
        r[i] = v / h[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double v = r[i];
        for (int k = i + 1; k < n; ++k) v -= h[k * n + i] * r[k];
        r[i] = v / h[i * n + i];
    }
    return 1;
}

/* Status when the iterations cannot continue (singular system, stalled line search, iteration limit): the
 * current point is accepted if it is already near optimal. */
static int stalled(int n, const double *w, double eta, double dual) {
    if (dual <= 1e-6 && eta <= 1e-8) return 1;
    if (n == 4 && dual <= 1e-6 && w[3] + eta < -1e-9) return 3;
    return -1;
}

/* Primal-dual iterations from a strictly feasible w.  Returns 1 converged (surrogate gap <= gap_tol and dual
 * residual <= 1e-9), 2 when stop_positive_tau and tau > 0, 3 when phase I proves tau* < 0, -1 on failure. */
static int solve(const Problem *P, double *w, double gap_tol, int stop_positive_tau) {
    const int n = P->nvar, k = P->m + P->q;
    double lam[MAXC], f[MAXC], dlam[MAXC], gall[MAXC][MAXV];
    double hess[MAXV * MAXV], step[MAXV], wt[MAXV], lt[MAXC], g0[MAXV], h[3];
    if (k > MAXC) return -1;
    for (int i = 0; i < k; ++i) lam[i] = 1.0 / -constraint(P, w, i, 0, 0);
    for (int it = 0; it < 100; ++it) {
        ++region_qp_newton;
        double eta = 0.0;
        for (int j = 0; j < n * n; ++j) hess[j] = 0.0;
        if (n == 3)
            for (int j = 0; j < 3; ++j) hess[j * n + j] = 1.0;
        objective_grad(P, w, g0);
        double rd[MAXV];
        for (int j = 0; j < n; ++j) rd[j] = g0[j];
        for (int i = 0; i < k; ++i) {
            f[i] = constraint(P, w, i, gall[i], h);
            eta -= f[i] * lam[i];
            for (int j = 0; j < n; ++j) rd[j] += lam[i] * gall[i][j];
            hess[0] += lam[i] * h[0];
            hess[1] += lam[i] * h[1];
            hess[n] += lam[i] * h[1];
            hess[n + 1] += lam[i] * h[2];
            const double weight = lam[i] / -f[i];
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < n; ++c) hess[r * n + c] += weight * gall[i][r] * gall[i][c];
        }
        double dual = 0.0;
        for (int j = 0; j < n; ++j) dual += rd[j] * rd[j];
        dual = sqrt(dual);
        if (stop_positive_tau && w[3] > 0.0) return 2;
        if (dual <= 1e-9 && eta <= gap_tol) return 1;
        if (n == 4 && dual <= 1e-9 && w[3] + eta < -1e-9) return 3;
        last_eta = eta;
        last_dual = dual;
        const double t = 10.0 * k / eta;
        for (int j = 0; j < n; ++j) {
            step[j] = -g0[j];
            for (int i = 0; i < k; ++i) step[j] -= gall[i][j] / (t * -f[i]);
        }
        if (!cholesky_solve(hess, step, n)) {
            return stalled(n, w, eta, dual);
        }
        double smax = 1.0;
        for (int i = 0; i < k; ++i) {
            double gd = 0.0;
            for (int j = 0; j < n; ++j) gd += gall[i][j] * step[j];
            dlam[i] = -lam[i] - 1.0 / (t * f[i]) - lam[i] * gd / f[i];
            if (dlam[i] < 0.0 && -lam[i] / dlam[i] < smax) smax = -lam[i] / dlam[i];
        }
        double s = 0.99 * smax;
        int accepted = 0;
        for (; s > 1e-14; s *= 0.5) {
            ++region_qp_backtracks;
            for (int j = 0; j < n; ++j) wt[j] = w[j] + s * step[j];
            if (!all_strict(P, wt)) continue;
            for (int i = 0; i < k; ++i) lt[i] = lam[i] + s * dlam[i];
            accepted = 1;
            break;
        }
        if (!accepted) {
            return stalled(n, w, eta, dual);
        }
        for (int j = 0; j < n; ++j) w[j] = wt[j];
        for (int i = 0; i < k; ++i) lam[i] = lt[i];
    }
    return stalled(n, w, last_eta, last_dual);
}

static void phase_one_start(const Problem *P, double *w) {
    const double *c0 = P->cones;
    w[0] = c0[0];
    w[1] = c0[1];
    w[2] = c0[2];
    w[3] = 0.0;
    double worst = -INFINITY;
    for (int i = 0; i < P->m + P->q; ++i) {
        const double f = constraint(P, w, i, 0, 0);
        if (f > worst) worst = f;
    }
    w[3] = -worst - 1.0;
}

int region_qp_margin(const double *a, const double *b, int m, const double *cones, int q, double *z, double *tau) {
    Problem P = {a, b, cones, 0, m, q, 4};
    double w[4];
    phase_one_start(&P, w);
    const int status = solve(&P, w, 1e-11, 0);
    z[0] = w[0];
    z[1] = w[1];
    z[2] = w[2];
    *tau = w[3];
    return status == 1 || status == 3 ? 1 : -1;
}

int region_qp_project(const double *a, const double *b, int m, const double *cones, int q, const double *p, double *z) {
    Problem P = {a, b, cones, p, m, q, 4};
    double w[4];
    phase_one_start(&P, w);
    const int status = solve(&P, w, 1e-11, 1);
    if (status == 1 || status == 3) return w[3] > 0.0 ? 1 : 0;
    if (status != 2) return -1;
    P.nvar = 3;
    if (solve(&P, w, 1e-12, 0) != 1) return -1;
    z[0] = w[0];
    z[1] = w[1];
    z[2] = w[2];
    return 1;
}
