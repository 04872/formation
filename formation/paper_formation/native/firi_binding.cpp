// C interface to the GCOPTER old FIRI derivative with planar disk seed, commit e0444f6d47b84f972ced91746b05feb36ce1fd4f.
#include "firi_body_seed.hpp"
#include <Eigen/Eigen>

extern "C" int gcopter_firi(const double *points, int count, const double *anchor,
                            double halfwidth, double halfheight, int iterations, double seed_radius,
                            double *output, int capacity) {
    Eigen::MatrixX4d boundary(6, 4);
    boundary << 1,0,0,-(anchor[0]+halfwidth),
               -1,0,0, anchor[0]-halfwidth,
                0,1,0,-(anchor[1]+halfwidth),
                0,-1,0,anchor[1]-halfwidth,
                0,0,1,-halfheight,
                0,0,-1,-halfheight;
    Eigen::Matrix3Xd cloud(3,count);
    for (int i=0;i<count;++i) cloud.col(i) << points[2*i],points[2*i+1],0.;
    Eigen::Vector3d seed(anchor[0],anchor[1],0.);
    Eigen::MatrixX4d poly;
    if (!firi::firi(boundary,cloud,seed,seed,poly,iterations,1.0e-6,seed_radius)) return -1;
    if (poly.rows()>capacity) return -2;
    for (int i=0;i<poly.rows();++i)
        for (int j=0;j<4;++j) output[4*i+j]=poly(i,j);
    return poly.rows();
}
