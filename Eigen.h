#ifndef _EIGEN_H
#define _EIGEN_H

#include <math.h>
#include <float.h>

#include "../util/ArrayMath.h"

using namespace util;

namespace dsp {
  
  /**
   * Special-purpose eigensolvers for digital signal processing.
   * Methods of this class solve small eigen-problems efficiently.
   * @author Luming Liang 
   * translated from Mines Java Toolkit written by Dave Hale.
   * @version 2012.06.06
   */

  class Eigen {
    /**
     * Computes eigenvalues and eigenvectors for a symmetric 2x2 matrix A.
     * If the eigenvectors are placed in columns in a matrix V, and the
     * eigenvalues are placed in corresponding columns of a diagonal
     * matrix D, then AV = VD.
     * @param a the symmetric matrix A.
     * @param v the array of eigenvectors v[0] and v[1].
     * @param d the array of eigenvalues d[0] and d[1].
     */

    public:
    static void solveSymmetric22(float a[2][2], float v[2][2], float d[2]) {
      // Copy matrix to local variables.
      float a00 = a[0][0], a01 = a[0][1], a11 = a[1][1];

      // Initial eigenvectors.
      float v00 = 1.0f, v01 = 0.0f;
      float v10 = 0.0f, v11 = 1.0f; 
      // If off-diagonal element is non-zero, zero it with a Jacobi rotation.
      if (a01!=0.0f) {
        float tiny = 0.1f*sqrt(FLT_EPSILON); // avoid overflow in r*r below
        float c,r,s,t,u,vpr,vqr;
        u = a11-a00;
        if (fabs(a01)<tiny*fabs(u)) {
          t = a01/u;
        } else {
          r = 0.5f*u/a01;
          t = (r>=0.0f)?1.0f/(r+sqrt(1.0f+r*r)):1.0f/(r-sqrt(1.0f+r*r));
        }
        c = 1.0f/sqrt(1.0f+t*t);
        s = t*c;
        u = s/(1.0f+c);
        r = t*a01;
        a00 -= r;
        a11 += r;
        //a01 = 0.0f;
        vpr = v00;
        vqr = v10;
        v00 = vpr-s*(vqr+vpr*u);
        v10 = vqr+s*(vpr-vqr*u);
        vpr = v01;
        vqr = v11;
        v01 = vpr-s*(vqr+vpr*u);
        v11 = vqr+s*(vpr-vqr*u);
      }
      // Copy eigenvalues and eigenvectors to output arrays.
      d[0] = a00;
      d[1] = a11;
      v[0][0] = v00; v[0][1] = v01;
      v[1][0] = v10; v[1][1] = v11;

      // Sort eigenvalues (and eigenvectors) in descending order.
      if (d[0]<d[1]) {
        float dt = d[1];
        d[1] = d[0];
        d[0] = dt;
        float vt0 = v[1][0], vt1 = v[1][1];
        v[1][0] = v[0][0]; v[1][1] = v[0][1];
        v[0][0] = vt0; v[0][1] = vt1;
      }
    }
    
    static void solveSymmetric22(double** a, double** v, double* d) {
      // Copy matrix to local variables.
      double a00 = a[0][0],  a01 = a[0][1], a11 = a[1][1];

      // Initial eigenvectors.
      double v00 = 1.0f, v01 = 0.0f;
      double v10 = 0.0f, v11 = 1.0f; 
      // If off-diagonal element is non-zero, zero it with a Jacobi rotation.
      if (a01!=0.0f) {
        double tiny = 0.1f*sqrt(FLT_EPSILON); // avoid overflow in r*r below
        double c,r,s,t,u,vpr,vqr;
        u = a11-a00;
        if (fabs(a01)<tiny*fabs(u)) {
          t = a01/u;
        } else {
          r = 0.5f*u/a01;
          t = (r>=0.0f)?1.0f/(r+sqrt(1.0f+r*r)):1.0f/(r-sqrt(1.0f+r*r));
        }
        c = 1.0f/sqrt(1.0f+t*t);
        s = t*c;
        u = s/(1.0f+c);
        r = t*a01;
        a00 -= r;
        a11 += r;
        //a01 = 0.0f;
        vpr = v00;
        vqr = v10;
        v00 = vpr-s*(vqr+vpr*u);
        v10 = vqr+s*(vpr-vqr*u);
        vpr = v01;
        vqr = v11;
        v01 = vpr-s*(vqr+vpr*u);
        v11 = vqr+s*(vpr-vqr*u);
      }
      // Copy eigenvalues and eigenvectors to output arrays.
      d[0] = a00;
      d[1] = a11;
      v[0][0] = v00; v[0][1] = v01;
      v[1][0] = v10; v[1][1] = v11;

      // Sort eigenvalues (and eigenvectors) in descending order.
      if (d[0]<d[1]) {
        double dt = d[1];
        d[1] = d[0];
        d[0] = dt;
        double* vt = v[1];
        v[1] = v[0];
        v[0] = vt;
        delete [] vt;
      }
    }

    /**
     * Computes eigenvalues and eigenvectors for a symmetric 3x3 matrix A.
     * If the eigenvectors are placed in columns in a matrix V, and the
     * eigenvalues are placed in corresponding columns of a diagonal
     * matrix D, then AV = VD.
     * @param a the symmetric matrix A.
     * @param v the array of eigenvectors v[0], v[1], and v[2].
     * @param d the array of eigenvalues d[0], d[1], and d[2].
     */
    static void solveSymmetric33(double** a, double** v, double* d) {
      solveSymmetric33Jacobi(a,v,d); // slow but more accurate
    }

    ///////////////////////////////////////////////////////////////////////////
    // private
    private: 

    /**
     * Sorts eigenvalues d and eigenvectors v in descending order.
     */
    static void sortDescending33(double** v, double* d) {
      double dj, *vj;
      for (int i=0; i<3; ++i)
        for (int j=i; j>0 && d[j-1]<d[j]; --j) {
          dj = d[j]; d[j] = d[j-1]; d[j-1] = dj;
          vj = v[j]; v[j] = v[j-1]; v[j-1] = vj;
        }
      delete [] vj;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Old iterative Jacobi method for symmetric 3x3 matrices. For random
    // matrices, this Jacobi solver is about 6 times slower than the
    // hybrid method.

    /**
     * Old iterative Jacobi solver. Slower than the current solver.
     */
    static void solveSymmetric33Jacobi(double** a, double** v, double* d) {
      // Copy matrix to local variables.
      double a00 = a[0][0],
             a01 = a[0][1], a11 = a[1][1],
             a02 = a[0][2], a12 = a[1][2], a22 = a[2][2];
 
      // Initial eigenvectors.
      double v00 = 1.0, v01 = 0.0, v02 = 0.0,
             v10 = 0.0, v11 = 1.0, v12 = 0.0,
             v20 = 0.0, v21 = 0.0, v22 = 1.0;

      // Tiny constant to avoid overflow of r*r (in computation of t) below.
      double tiny = 0.1*sqrt(DBL_EPSILON);
     
      // Absolute values of off-diagonal elements.
      double aa01 = fabs(a01);
      double aa02 = fabs(a02);
      double aa12 = fabs(a12);

      // Apply Jacobi rotations until all off-diagonal elements are zero.
      // Count rotations, just in case this does not converge.
      for (int nrot=0; aa01+aa02+aa12>0.0; ++nrot) {
        double c,r,s,t,u,vpr,vqr,apr,aqr;

        // If a01 is the largest off-diagonal element, ...
        if (aa01>=aa02 && aa01>=aa12) {
          u = a11-a00;
          if (fabs(a01)<tiny*fabs(u)) {
            t = a01/u;
          } else {
            r = 0.5*u/a01;
            t = (r>=0.0)?1.0/(r+sqrt(1.0+r*r)):1.0/(r-sqrt(1.0+r*r));
          }
          c = 1.0/sqrt(1.0+t*t);
          s = t*c;
          u = s/(1.0+c);
          r = t*a01;
          a00 -= r;
          a11 += r;
          a01 = 0.0;
          apr = a02;
          aqr = a12;
          a02 = apr-s*(aqr+apr*u);
          a12 = aqr+s*(apr-aqr*u);
          vpr = v00;
          vqr = v10;
          v00 = vpr-s*(vqr+vpr*u);
          v10 = vqr+s*(vpr-vqr*u);
          vpr = v01;
          vqr = v11;
          v01 = vpr-s*(vqr+vpr*u);
          v11 = vqr+s*(vpr-vqr*u);
          vpr = v02;
          vqr = v12;
          v02 = vpr-s*(vqr+vpr*u);
          v12 = vqr+s*(vpr-vqr*u);
        }
      
        // Else if a02 is the largest off-diagonal element, ...
        else if (aa02>=aa01 && aa02>=aa12) {
          u = a22-a00;
          if (fabs(a02)<tiny*fabs(u)) {
            t = a02/u;
          } else {
            r = 0.5*u/a02;
            t = (r>=0.0)?1.0/(r+sqrt(1.0+r*r)):1.0/(r-sqrt(1.0+r*r));
          }
          c = 1.0/sqrt(1.0+t*t);
          s = t*c;
          u = s/(1.0+c);
          r = t*a02;
          a00 -= r;
          a22 += r;
          a02 = 0.0;
          apr = a01;
          aqr = a12;
          a01 = apr-s*(aqr+apr*u);
          a12 = aqr+s*(apr-aqr*u);
          vpr = v00;
          vqr = v20;
          v00 = vpr-s*(vqr+vpr*u);
          v20 = vqr+s*(vpr-vqr*u);
          vpr = v01;
          vqr = v21;
          v01 = vpr-s*(vqr+vpr*u);
          v21 = vqr+s*(vpr-vqr*u);
          vpr = v02;
          vqr = v22;
          v02 = vpr-s*(vqr+vpr*u);
          v22 = vqr+s*(vpr-vqr*u);
        }

        // Else if a12 is the largest off-diagonal element, ...
        else {
          u = a22-a11;
          if (fabs(a12)<tiny*fabs(u)) {
            t = a12/u;
          } else {
            r = 0.5*u/a12;
            t = (r>=0.0)?1.0/(r+sqrt(1.0+r*r)):1.0/(r-sqrt(1.0+r*r));
          }
          c = 1.0/sqrt(1.0+t*t);
          s = t*c;
          u = s/(1.0+c);
          r = t*a12;
          a11 -= r;
          a22 += r;
          a12 = 0.0;
          apr = a01;
          aqr = a02;
          a01 = apr-s*(aqr+apr*u);
          a02 = aqr+s*(apr-aqr*u);
          vpr = v10;
          vqr = v20;
          v10 = vpr-s*(vqr+vpr*u);
          v20 = vqr+s*(vpr-vqr*u);
          vpr = v11;
          vqr = v21;
          v11 = vpr-s*(vqr+vpr*u);
          v21 = vqr+s*(vpr-vqr*u);
          vpr = v12;
          vqr = v22;
          v12 = vpr-s*(vqr+vpr*u);
          v22 = vqr+s*(vpr-vqr*u);
        }

        // Update absolute values of all off-diagonal elements.
        aa01 = fabs(a01);
        aa02 = fabs(a02);
        aa12 = fabs(a12);
      }

      // Copy eigenvalues and eigenvectors to output arrays.
      d[0] = a00;
      d[1] = a11;
      d[2] = a22;
      v[0][0] = v00; v[0][1] = v01; v[0][2] = v02;
      v[1][0] = v10; v[1][1] = v11; v[1][2] = v12;
      v[2][0] = v20; v[2][1] = v21; v[2][2] = v22;

      // Sort eigenvalues (and eigenvectors) in descending order.
      sortDescending33(v,d);
    }
  };
}

#endif
