/* Batched rejection sampler using NumPy's own bit generator and normal draw. */
#include <math.h>
#include <stdint.h>
#include "numpy/random/distributions.h"

static int valid(const double *a, const double *ref, int n, double pair_min) {
    const double trace=((a[0]*a[0]+a[1]*a[1])+a[2]*a[2])+a[3]*a[3];
    const double det=a[0]*a[3]-a[1]*a[2];
    const double d=trace*trace-4.*det*det;
    const double disc=sqrt(fmax(0.,d));
    const double mn=(trace-disc)/2.,mx=(trace+disc)/2.;
    if (!(det>0. && mn>=.42*.42 && mx<=1.35*1.35 && fabs(a[1])+fabs(a[2])<=1.5)) return 0;
    const double min_distance=2.*.16+.08;
    if (mn*(pair_min*pair_min)>=min_distance*min_distance) return 1;
    for (int i=0;i<n;i++) for (int j=i+1;j<n;j++) {
        const double x=ref[2*i]-ref[2*j],y=ref[2*i+1]-ref[2*j+1];
        const double px=a[0]*x+a[1]*y,py=a[2]*x+a[3]*y;
        if (sqrt(px*px+py*py)<min_distance) return 0;
    }
    return 1;
}

/* Outputs (angle, radius, A00, A01, A10, A11), attempts includes rejects. */
uint64_t sample_affine(bitgen_t *rng, double direction, const double *q,
                       const double *ref, int n, double pair_min,
                       int count, double *output) {
    uint64_t attempts=0;
    const double pi=3.141592653589793238462643383279502884;
    for (int k=0;k<count;k++) {
        double a[4],angle,radius;
        do {
            angle=direction+(-pi/2.+pi*rng->next_double(rng->state));
            radius=.1+(.2-.1)*rng->next_double(rng->state);
            for (int i=0;i<4;i++) a[i]=q[i]+.12*random_standard_normal(rng);
            attempts++;
        } while (!valid(a,ref,n,pair_min));
        output[k*6]=angle;output[k*6+1]=radius;
        for (int i=0;i<4;i++)output[k*6+2+i]=a[i];
    }
    return attempts;
}
