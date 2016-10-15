#ifndef _LOCALDIFFUSIONKERNEL_H
#define _LOCALDIFFUSIONKERNEL_H

#include <math.h>
#include <float.h>
#include <iostream>

#include "Tensors2.h"
#include "Tensors3.h"
#include "../util/ArrayMath.h"

using namespace std;
using namespace util;

/**
* A local diffusion kernel for use in anisotropic diffusion filtering.
* <p>
* This kernel is a filter that computes y += G'DGx where G is a
* gradient operator, G' is its adjoint, and D is a local diffusion
* tensor field that determines for each image sample the filter
* coefficients.
* <p>
* A local diffusion kernel is typically used in combinations with others.
* For example, the filter implied by (I+G'DG)y = G'DGx acts as a notch
* filter. It attenuates features for which G'DG is zero while preserving
* other features. The diffusion tensors in D control the width, orientation,
* and anisotropy of the spectral notch. Note that application of this filter
* requires solution of a sparse symmetric positive-definite system of
* equations.
* <p>
* An even simpler example is the filter implied by (I+G'DG)y = x. This
* filter smooths features in the directions implied by the tensors D.
* Again, application of this filter requires solving a sparse symmetric
* positive-definite system of equations.
* <p>
* The accumulation of the kernel output in y = y+G'DGx is useful when
* constructing such filter combinations. Given y = 0, this kernel
* computes y = G'DGx. Given y = x, it computes y = (I+G'DG)x.
*
* Translated by Luming Liang from Mines Java Toolkit. 
* @version 2012.06.05
*/

namespace dsp {
  
  class LocalDiffusionKernel {
    
   /**
     * The stencil used in finite-difference approximation of derivatives.
     * In each stencil name, the first digit equals the number of samples
     * used in the direction of the derivative, and the second digit equals
     * the number of samples in the orthogonal direction. Names correspond
     * to 2D stencils, but each has a natural 3D extension.
     * <p>
     * Note that the stencil implied by G'DG is larger than that used to
     * approximate the derivatives in G. For example, a 2x2 derivative
     * approximation implies a 3x3 stencil for G'DG.
     */
    public:
    enum Stencil {
      /**
       * A 2x1 stencil.
       * The 2D version has 3 non-zero coefficients.
       * The 3D version has 4 non-zero coefficients.
       * This stencil should be specified for isotropic diffusion only.
       * When using this stencil, any specified tensors D are ignored.
       */
      D21,
      /**
       * A 2x2 stencil.
       * The 2D version has 4 non-zero coefficients.
       * The 3D version has 8 non-zero coefficients.
       * This stencil is the default.
       */
      D22,
      /**
       * A 2x4 stencil.
       * The 2D version has 8 non-zero coefficients.
       * The 3D version has 24 non-zero coefficients.
       * <em>The 3D version is not yet implemented.</em>
       */
      D24,
      /**
       * A 3x3 stencil.
       * The 2D version has 6 non-zero coefficients.
       * The 3D version has 18 non-zero coefficients.
       */
      D33,
      /** A 7x1 stencil.
       * Both 2D and 3D versions have 6 non-zero coefficients.
       */
      D71,
      /** A 9x1 stencil.
       * Both 2D and 3D versions have 8 non-zero coefficients.
       * <em>The 3D version is not yet implemented.</em>
       */
      D91,
    };
    
    ////////////////////////////////////////////////////////////////////////
    // private fields
    private:
    Stencil _stencil;
    int _npass;
    bool _parallel;
    class IDT2: public Tensors2 {  
      public: void getTensor(int i1, int i2, float* d) {
       d[0] = 1.0f;
       d[1] = 0.0f;
       d[2] = 1.0f;
      }    
    } IDENTITY_TENSORS2;
    class IDT3: public Tensors3 {
      public: void getTensor(int i1, int i2, int i3, float* d) {
        d[0] = 1.0f;
        d[1] = 0.0f;
        d[2] = 0.0f;
        d[3] = 1.0f;
        d[4] = 0.0f;
        d[5] = 1.0f;
      }
    } IDENTITY_TENSORS3;
    
    ////////////////////////////////////////////////////////////////////////
    // private functions

    //////////////////////////////////////////////////////////////////////////
    // gradients (forward and transpose) for stencils with mx1 coefficients

    static void gf(float* c, float* x, float* y, int nc, int n1) {
      //int nc = c.length-1;
      //int n1 = x.length;
      int n1m1 = n1-1, n1nc = n1-nc;
      for (int i1=0; i1<min(nc,n1nc); ++i1) {
        float yi = 0.0f;
        for (int ic=1; ic<=nc; ++ic) {
          float ci = c[ic];
          int im = i1-ic; if (im<0 ) im = 0;
          int ip = i1+ic; if (ip>n1m1) ip = n1m1;
          yi += ci*(x[ip]-x[im]);
        }
        y[i1] = yi;
      }
      if (nc==3 && n1>6) { // middle part optimized for nc = 3
        float c1 = c[1], c2 = c[2], c3 = c[3];
        float xm3, xm2 = x[0], xm1 = x[1], xp0 = x[2],
                   xp1 = x[3], xp2 = x[4], xp3 = x[5];
        for (int i1=3; i1<n1nc; ++i1) {
          xm3 = xm2; xm2 = xm1; xm1 = xp0;
          xp0 = xp1; xp1 = xp2; xp2 = xp3;
          xp3 = x[i1+3];
          y[i1] = c1*(xp1-xm1)+c2*(xp2-xm2)+c3*(xp3-xm3);
        }
      } else { // middle part for general case
        for (int i1=nc; i1<n1nc; ++i1) {
          float yi = 0.0f;
          for (int ic=1; ic<=nc; ++ic)
            yi += c[ic]*(x[i1+ic]-x[i1-ic]);
          y[i1] = yi;
        }
      }
      for (int i1=max(n1nc,0); i1<n1; ++i1) {
        float yi = 0.0f;
        for (int ic=1; ic<=nc; ++ic) {
          float ci = c[ic];
          int im = i1-ic; if (im<0 ) im = 0;
          int ip = i1+ic; if (ip>n1m1) ip = n1m1;
          yi += ci*(x[ip]-x[im]);
        }
        y[i1] = yi;
      }
    }

    static void gt(float* c, float* x, float* y, int nc, int n1) {
      //int nc = c.length-1;
      //int n1 = x.length;
      int n1m1 = n1-1, n1nc = n1-nc;
      for (int i1=0; i1<min(2*nc,n1); ++i1) {
        float xi = x[i1];
        for (int ic=1; ic<=nc; ++ic) {
          float ci = c[ic];
          int im = i1-ic; if (im<0 ) im = 0;
          int ip = i1+ic; if (ip>n1m1) ip = n1m1;
          if (im<nc) y[im] -= ci*xi;
          if (ip<nc) y[ip] += ci*xi;
        }
      }
      if (nc==3 && n1>6) { // middle part optimized for nc = 3
        float c1 = c[1], c2 = c[2], c3 = c[3];
        float xm3, xm2 = x[0], xm1 = x[1], xp0 = x[2],
                   xp1 = x[3], xp2 = x[4], xp3 = x[5];
        for (int i1=3; i1<n1nc; ++i1) {
          xm3 = xm2; xm2 = xm1; xm1 = xp0;
          xp0 = xp1; xp1 = xp2; xp2 = xp3;
          xp3 = x[i1+3];
          y[i1] += c1*(xm1-xp1)+c2*(xm2-xp2)+c3*(xm3-xp3);
        }
      } else { // middle part for general case
        for (int i1=nc; i1<n1nc; ++i1) {
          float yi = y[i1];
          for (int ic=1; ic<=nc; ++ic)
            yi += c[ic]*(x[i1-ic]-x[i1+ic]);
          y[i1] = yi;
        }
      }
      n1nc = max(n1nc,nc);
      for (int i1=max(n1-2*nc,0); i1<n1; ++i1) {
        float xi = x[i1];
        for (int ic=1; ic<=nc; ++ic) {
          float ci = c[ic];
          int im = i1-ic; if (im<0 ) im = 0;
          int ip = i1+ic; if (ip>n1m1) ip = n1m1;
          if (im>=n1nc) y[im] -= ci*xi;
          if (ip>=n1nc) y[ip] += ci*xi;
        }
      }
    }

    static void gf1(float* c, float** x, float** g1, int nc, int n1, int n2) {
      for (int i2=0; i2<n2; ++i2)
        gf(c,x[i2],g1[i2],nc,n1);
    }
    static void gf2(float* c, float** x, float** g2, int nc, int n1, int n2) {
      if (nc==3) { // optimized for nc = 3
        float* C71 = new float[4];
        C71[0] =0.0f; C71[1] = 0.830893f; 
        C71[2] = -0.227266f; C71[3] =  0.042877f;
        float c1 = C71[1], c2 = C71[2], c3 = C71[3];
        delete [] C71;
        int n2m1 = n2-1, n2m2 = n2-2, n2m3 = n2-3;
        for (int i2=0; i2<n2; ++i2) {
          float* xm3 = (i2>=3)?x[i2-3]:x[0];
          float* xm2 = (i2>=2)?x[i2-2]:x[0];
          float* xm1 = (i2>=1)?x[i2-1]:x[0];
          float* xp1 = (i2<n2m1)?x[i2+1]:x[n2m1];
          float* xp2 = (i2<n2m2)?x[i2+2]:x[n2m1];
          float* xp3 = (i2<n2m3)?x[i2+3]:x[n2m1];
          float* g2i = g2[i2];
          for (int i1=0; i1<n1; ++i1) {
            g2i[i1] = c1*(xp1[i1]-xm1[i1]) +
                      c2*(xp2[i1]-xm2[i1]) +
                      c3*(xp3[i1]-xm3[i1]);
          }
        }
      } else { // not optimized
        int n2m1 = n2-1;
        for (int i2=0; i2<n2; ++i2) {
          float* g2i = g2[i2];
          zero(g2i,n1);
          for (int ic=1; ic<=nc; ++ic) {
            float ci = c[ic];
            float* xm = (i2>=ic)?x[i2-ic]:x[0];
            float* xp = (i2<n2-ic)?x[i2+ic]:x[n2m1];
            for (int i1=0; i1<n1; ++i1)
              g2i[i1] += ci*(xp[i1]-xm[i1]);
          }
        }
      }
    }
    
    static void gt1(float* c, float** g1, float** x, int nc, int n1, int n2) {
      for (int i2=0; i2<n2; ++i2)
        gt(c,g1[i2],x[i2],nc,n1);
    }
    static void gt2(float* c, float** g2, float** x, int nc, int n1, int n2) {
      int n2m1 = n2-1, n2nc = n2-nc;
      for (int i2=0; i2<min(2*nc,n2); ++i2) { // rolling on
        float* g2i = g2[i2];
        for (int ic=1; ic<=nc; ++ic) {
          float ci = c[ic];
          int im = i2-ic; if (im<0 ) im = 0;
          int ip = i2+ic; if (ip>n2m1) ip = n2m1;
          if (im<nc) {
            float* x2m = x[im];
            for (int i1=0; i1<n1; ++i1)
              x2m[i1] -= ci*g2i[i1];
          }
          if (ip<nc) {
            float* x2p = x[ip];
            for (int i1=0; i1<n1; ++i1)
              x2p[i1] += ci*g2i[i1];
          }
        }
      }
      if (nc==3 && n1>6) { // middle part optimized for nc = 3
        float c1 = c[1], c2 = c[2], c3 = c[3];
        for (int i2=3; i2<n2-3; ++i2) {
          float *gm3 = g2[i2-3], *gm2 = g2[i2-2], *gm1 = g2[i2-1];
          float *gp1 = g2[i2+1], *gp2 = g2[i2+2], *gp3 = g2[i2+3];
          float * x2 = x[i2];
          for (int i1=0; i1<n1; ++i1) {
            x2[i1] += c1*(gm1[i1]-gp1[i1]) +
                      c2*(gm2[i1]-gp2[i1]) +
                      c3*(gm3[i1]-gp3[i1]);
          }
        }
      } else { // middle part not optimized
        for (int i2=nc; i2<n2-nc; ++i2) {
          float* x2 = x[i2];
          for (int ic=1; ic<=nc; ++ic) {
            float ci = c[ic];
            float* g2m = g2[i2-ic];
            float* g2p = g2[i2+ic];
            for (int i1=0; i1<n1; ++i1)
              x2[i1] += ci*(g2m[i1]-g2p[i1]);
          }
        }
      }
      n2nc = max(n2nc,nc);
      for (int i2=max(n2-2*nc,0); i2<n2; ++i2) { // rolling off
        float* g2i = g2[i2];
        for (int ic=1; ic<=nc; ++ic) {
          float ci = c[ic];
          int im = i2-ic; if (im<0 ) im = 0;
          int ip = i2+ic; if (ip>n2m1) ip = n2m1;
          if (im>=n2nc) {
            float* x2m = x[im];
            for (int i1=0; i1<n1; ++i1)
              x2m[i1] -= ci*g2i[i1];
          }
          if (ip>=n2nc) {
            float* x2p = x[ip];
            for (int i1=0; i1<n1; ++i1)
              x2p[i1] += ci*g2i[i1];
          }
        }
      }
    }

    void gf(float* c, float** x, float** g1, float** g2, 
      int nc, int n1, int n2) {
      gf1(c,x,g1,nc,n1,n2);
      gf2(c,x,g2,nc,n1,n2);
    }
    void gt(float* c, float** g1, float** g2, float** x, 
      int nc, int n1, int n2) {
      gt2(c,g2,x,nc,n1,n2);
      gt1(c,g1,x,nc,n1,n2);
    }
    ////////////////////////////////////////////////////////////////////////
    // D21 (for isotropic diffusion only)
    private:
    void apply21(float c, float** s, float** x, float** y, int n1, int n2) {
      for (int i2=0; i2<n2; ++i2) {
        int m2 = (i2>0)?i2-1:0;
        for (int i1=0; i1<n1; ++i1) {
          int m1 = (i1>0)?i1-1:0;
          float cs1 = c, cs2 = c;
          if (s!=NULL) {
            cs1 *= 0.5f*(s[i2][i1]+s[i2][m1]);
            cs2 *= 0.5f*(s[i2][i1]+s[m2][i1]);
          }
          float x1 = x[i2][i1]-x[i2][m1];
          float x2 = x[i2][i1]-x[m2][i1];
          float y1 = cs1*x1;
          float y2 = cs2*x2;
          y[i2][i1] += y1+y2;
          y[i2][m1] -= y1;
          y[m2][i1] -= y2;
        }
      }
    }
    void apply21(int i3, float c, float*** s, float*** x, float*** y, 
      int n1, int n2) {
      int m3 = (i3>0)?i3-1:0;
      for (int i2=0; i2<n2; ++i2) {
        int m2 = (i2>0)?i2-1:0;
        for (int i1=0; i1<n1; ++i1) {
          int m1 = (i1>0)?i1-1:0;
          float cs1 = c;
          float cs2 = c;
          float cs3 = c;
          if (s!=NULL) {
            cs1 *= 0.5f*(s[i3][i2][i1]+s[i3][i2][m1]);
            cs2 *= 0.5f*(s[i3][i2][i1]+s[i3][m2][i1]);
            cs3 *= 0.5f*(s[i3][i2][i1]+s[m3][i2][i1]);
          }
          float csi = (s!=NULL)?c*s[i3][i2][i1]:c;
          float x1 = x[i3][i2][i1]-x[i3][i2][m1];
          float x2 = x[i3][i2][i1]-x[i3][m2][i1];
          float x3 = x[i3][i2][i1]-x[m3][i2][i1];
          float y1 = cs1*x1;
          float y2 = cs2*x2;
          float y3 = cs3*x3;
          y[i3][i2][i1] += y1+y2+y3;
          y[i3][i2][m1] -= y1;
          y[i3][m2][i1] -= y2;
          y[m3][i2][i1] -= y3;
        }
      }
    }    
    ////////////////////////////////////////////////////////////////////////
    // D22
    void apply22(Tensors2* d, float c, float** s, float** x, float** y, 
      int n1, int n2) {
      c *= 0.25f;
      float di[3];
      for (int i2=1; i2<n2; ++i2) {
        float* x0 = x[i2 ];
        float* xm = x[i2-1];
        float* y0 = y[i2 ];
        float* ym = y[i2-1];
        for (int i1=1,m1=0; i1<n1; ++i1,++m1) {
          d->getTensor(i1,i2,di);
          float csi = (s!=NULL)?c*s[i2][i1]:c;
          float d11 = di[0]*csi;
          float d12 = di[1]*csi;
          float d22 = di[2]*csi;
          float x00 = x0[i1];
          float x0m = x0[m1];
          float xm0 = xm[i1];
          float xmm = xm[m1];
          float xa = x00-xmm;
          float xb = x0m-xm0;
          float x1 = xa-xb;
          float x2 = xa+xb;
          float y1 = d11*x1+d12*x2;
          float y2 = d12*x1+d22*x2;
          float ya = y1+y2;
          float yb = y1-y2;
          y0[i1] += ya;
          y0[m1] -= yb;
          ym[i1] += yb;
          ym[m1] -= ya;
        }
      }
    }

    void apply22(int i3, Tensors3* d, float c, float*** s, float*** x, 
      float*** y, int n1, int n2) {
      c *= 0.0625f;
      float di[6];
      for (int i2=1; i2<n2; ++i2) {
        float* x00 = x[i3 ][i2 ];
        float* x0m = x[i3 ][i2-1];
        float* xm0 = x[i3-1][i2 ];
        float* xmm = x[i3-1][i2-1];
        float* y00 = y[i3 ][i2 ];
        float* y0m = y[i3 ][i2-1];
        float* ym0 = y[i3-1][i2 ];
        float* ymm = y[i3-1][i2-1];
        for (int i1=1,m1=0; i1<n1; ++i1,++m1) {
          d->getTensor(i1,i2,i3,di);
          float csi = (s!=NULL)?c*s[i3][i2][i1]:c;
          float d11 = di[0]*csi;
          float d12 = di[1]*csi;
          float d13 = di[2]*csi;
          float d22 = di[3]*csi;
          float d23 = di[4]*csi;
          float d33 = di[5]*csi;
          float xa = x00[i1]-xmm[m1];
          float xb = x00[m1]-xmm[i1];
          float xc = x0m[i1]-xm0[m1];
          float xd = xm0[i1]-x0m[m1];
          float x1 = xa-xb+xc+xd;
          float x2 = xa+xb-xc+xd;
          float x3 = xa+xb+xc-xd;
          float y1 = d11*x1+d12*x2+d13*x3;
          float y2 = d12*x1+d22*x2+d23*x3;
          float y3 = d13*x1+d23*x2+d33*x3;
          float ya = y1+y2+y3; y00[i1] += ya; ymm[m1] -= ya;
          float yb = y1-y2+y3; y0m[i1] += yb; ym0[m1] -= yb;
          float yc = y1+y2-y3; ym0[i1] += yc; y0m[m1] -= yc;
          float yd = y1-y2-y3; ymm[i1] += yd; y00[m1] -= yd;
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////
    // D24

    void apply24(Tensors2* d, float c, float** s, float** x, float** y, 
      int n1, int n2) {
      float p = 0.18f; // best for high anisotropy
      float a = 0.5f*(1.0f+p);
      float b = 0.5f*( -p);
      b /= a;
      c *= a*a;
      float di[3];
      int i2m2, i2m1 = 0, i2p0 = 0, i2p1 = 1;
      for (int i2=1; i2<n2; ++i2) {
        i2m2 = i2m1; i2m1 = i2p0; i2p0 = i2p1; ++i2p1;
        if (i2p1>=n1) i2p1 = n1-1;
        float *xm2=x[i2m2], *xm1=x[i2m1], *xp0=x[i2p0], *xp1=x[i2p1];
        float *ym2=y[i2m2], *ym1=y[i2m1], *yp0=y[i2p0], *yp1=y[i2p1];
        int m2, m1 = 0, p0 = 0, p1 = 1;
        for (int i1=1; i1<n1; ++i1) {
          m2 = m1; m1 = p0; p0 = p1; ++p1;
          if (p1>=n1) p1 = n1-1;
          d->getTensor(i1,i2,di);
          float csi = (s!=NULL)?c*s[i2][i1]:c;
          float d11 = di[0]*csi;
          float d12 = di[1]*csi;
          float d22 = di[2]*csi;
          float xa = xp0[p0]-xm1[m1];
          float xb = xm1[p0]-xp0[m1];
          float x1 = xa+xb+b*(xp1[p0]+xm2[p0]-xp1[m1]-xm2[m1]);
          float x2 = xa-xb+b*(xp0[p1]+xp0[m2]-xm1[p1]-xm1[m2]);
          float y1 = d11*x1+d12*x2;
          float y2 = d12*x1+d22*x2;
          float ya = y1+y2;
          float yb = y1-y2;
          float yc = b*y1;
          float yd = b*y2;
          yp0[p0] += ya; ym1[m1] -= ya;
          ym1[p0] += yb; yp0[m1] -= yb;
          yp1[p0] += yc; ym2[m1] -= yc;
          ym2[p0] += yc; yp1[m1] -= yc;
          yp0[p1] += yd; ym1[m2] -= yd;
          yp0[m2] += yd; ym1[p1] -= yd;
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////
    // D33

    void apply33(Tensors2* d, float c, float** s, float** x, float** y, 
      int n1, int n2) {
      float p = 0.182962f; // Scharr, best for high anisotropy
      float a = 0.5f-p; // ~ 10/32
      float b = 0.5f*p; // ~ 3/32
      b /= a;
      c *= a*a;
      float di[3];
      for (int i2=1; i2<n2-1; ++i2) {
        float *xm = x[i2-1], *x0 = x[i2], *xp = x[i2+1];
        float *ym = y[i2-1], *y0 = y[i2], *yp = y[i2+1];
        float xmm, xm0 = xm[0], xmp = xm[1];
        float x0m, x00 = x0[0], x0p = x0[1];
        float xpm, xp0 = xp[0], xpp = xp[1];
        float ymm, ym0 = ym[0], ymp = ym[1];
        float y0m, y00 = y0[0], y0p = y0[1];
        float ypm, yp0 = yp[0], ypp = yp[1];
        for (int i1m=0,i1=1,i1p=2; i1p<n1; ++i1m,++i1,++i1p) {
          xmm = xm0; xm0 = xmp; xmp = xm[i1p];
          x0m = x00; x00 = x0p; x0p = x0[i1p];
          xpm = xp0; xp0 = xpp; xpp = xp[i1p];
          ymm = ym0; ym0 = ymp; ymp = ym[i1p];
          y0m = y00; y00 = y0p; y0p = y0[i1p];
          ypm = yp0; yp0 = ypp; ypp = yp[i1p];
          d->getTensor(i1,i2,di);
          float csi = (s!=NULL)?c*s[i2][i1]:c;
          float d11 = di[0]*csi;
          float d12 = di[1]*csi;
          float d22 = di[2]*csi;
          float xa = b*(xpp-xmm);
          float xb = b*(xmp-xpm);
          float x1 = x0p-x0m+xa+xb;
          float x2 = xp0-xm0+xa-xb;
          float y1 = d11*x1+d12*x2;
          float y2 = d12*x1+d22*x2;
          float ya = b*(y1+y2);
          float yb = b*(y1-y2);
          y0p += y1; y0m -= y1;
          ypp += ya; ymm -= ya;
          ymp += yb; ypm -= yb;
          yp0 += y2; ym0 -= y2;
          ym[i1m] = ymm;
          y0[i1m] = y0m;
          yp[i1m] = ypm;
        }
        ym[n1-2] = ym0; ym[n1-1] = ymp;
        y0[n1-2] = y00; y0[n1-1] = y0p;
        yp[n1-2] = yp0; yp[n1-1] = ypp;
      }
    }

    void apply33(int i3, Tensors3* d, float c, float*** s, float*** x, 
      float*** y, int n1, int n2) {
      float p = 0.174654f; // Scharr, best for high anisotropy
      float a = 1.0f-2.0f*p;
      float b = p;
      float aa = 0.5f*a*a;
      float ab = 0.5f*a*b;
      float bb = 0.5f*b*b;
      float di[6];
      for (int i2=1; i2<n2-1; ++i2) {
        float *xmm = x[i3-1][i2-1], *xm0 = x[i3-1][i2 ], *xmp = x[i3-1][i2+1];
        float *x0m = x[i3  ][i2-1], *x00 = x[i3  ][i2 ], *x0p = x[i3  ][i2+1];
        float *xpm = x[i3+1][i2-1], *xp0 = x[i3+1][i2 ], *xpp = x[i3+1][i2+1];
        float *ymm = y[i3-1][i2-1], *ym0 = y[i3-1][i2 ], *ymp = y[i3-1][i2+1];
        float *y0m = y[i3  ][i2-1], *y00 = y[i3  ][i2 ], *y0p = y[i3  ][i2+1];
        float *ypm = y[i3+1][i2-1], *yp0 = y[i3+1][i2 ], *ypp = y[i3+1][i2+1];
        for (int m1=0,i1=1,p1=2; p1<n1; ++m1,++i1,++p1) {
          d->getTensor(i1,i2,i3,di);
          float csi = (s!=NULL)?c*s[i3][i2][i1]:c;
          float d11 = di[0]*csi;
          float d12 = di[1]*csi;
          float d13 = di[2]*csi;
          float d22 = di[3]*csi;
          float d23 = di[4]*csi;
          float d33 = di[5]*csi;
          float xmmm = xmm[m1], xmm0 = xmm[i1], xmmp = xmm[p1];
          float xm0m = xm0[m1], xm00 = xm0[i1], xm0p = xm0[p1];
          float xmpm = xmp[m1], xmp0 = xmp[i1], xmpp = xmp[p1];
          float x0mm = x0m[m1], x0m0 = x0m[i1], x0mp = x0m[p1];
          float x00m = x00[m1], x00p = x00[p1];
          float x0pm = x0p[m1], x0p0 = x0p[i1], x0pp = x0p[p1];
          float xpmm = xpm[m1], xpm0 = xpm[i1], xpmp = xpm[p1];
          float xp0m = xp0[m1], xp00 = xp0[i1], xp0p = xp0[p1];
          float xppm = xpp[m1], xpp0 = xpp[i1], xppp = xpp[p1];
          float x00p00m = x00p-x00m; // aa differences, used once
          float x0p00m0 = x0p0-x0m0;
          float xp00m00 = xp00-xm00;
          float xmp0mm0 = xmp0-xmm0; // ab differences, used twice
          float xpp0pm0 = xpp0-xpm0;
          float xpm0mm0 = xpm0-xmm0;
          float xpp0mp0 = xpp0-xmp0;
          float xm0pm0m = xm0p-xm0m;
          float xp0pp0m = xp0p-xp0m;
          float xp0mm0m = xp0m-xm0m;
          float xp0pm0p = xp0p-xm0p;
          float x0mp0mm = x0mp-x0mm;
          float x0pp0pm = x0pp-x0pm;
          float x0pm0mm = x0pm-x0mm;
          float x0pp0mp = x0pp-x0mp;
          float xpppmmm = xppp-xmmm; // bb differences, used thrice
          float xppmmmp = xppm-xmmp;
          float xpmpmpm = xpmp-xmpm;
          float xmpppmm = xmpp-xpmm;
          float x1 = aa*x00p00m +
                     ab*(x0pp0pm+x0mp0mm+xp0pp0m+xm0pm0m) +
                     bb*(xpppmmm-xppmmmp+xpmpmpm+xmpppmm);
          float x2 = aa*x0p00m0 +
                     ab*(x0pp0mp+x0pm0mm+xpp0pm0+xmp0mm0) +
                     bb*(xpppmmm+xppmmmp-xpmpmpm+xmpppmm);
          float x3 = aa*xp00m00 +
                     ab*(xp0pm0p+xp0mm0m+xpp0mp0+xpm0mm0) +
                     bb*(xpppmmm+xppmmmp+xpmpmpm-xmpppmm);
          float y1 = d11*x1+d12*x2+d13*x3;
          float y2 = d12*x1+d22*x2+d23*x3;
          float y3 = d13*x1+d23*x2+d33*x3;
          float aa00p = aa*y1; y00[p1] += aa00p; y00[m1] -= aa00p;
          float aa0p0 = aa*y2; y0p[i1] += aa0p0; y0m[i1] -= aa0p0;
          float aap00 = aa*y3; yp0[i1] += aap00; ym0[i1] -= aap00;
          float ab0pp = ab*(y1+y2); y0p[p1] += ab0pp; y0m[m1] -= ab0pp;
          float ab0mp = ab*(y1-y2); y0m[p1] += ab0mp; y0p[m1] -= ab0mp;
          float abp0p = ab*(y1+y3); yp0[p1] += abp0p; ym0[m1] -= abp0p;
          float abm0p = ab*(y1-y3); ym0[p1] += abm0p; yp0[m1] -= abm0p;
          float abpp0 = ab*(y2+y3); ypp[i1] += abpp0; ymm[i1] -= abpp0;
          float abmp0 = ab*(y2-y3); ymp[i1] += abmp0; ypm[i1] -= abmp0;
          float bbppp = bb*(y1+y2+y3); ypp[p1] += bbppp; ymm[m1] -= bbppp;
          float bbmmp = bb*(y1-y2-y3); ymm[p1] += bbmmp; ypp[m1] -= bbmmp;
          float bbpmp = bb*(y1-y2+y3); ypm[p1] += bbpmp; ymp[m1] -= bbpmp;
          float bbmpp = bb*(y1+y2-y3); ymp[p1] += bbmpp; ypm[m1] -= bbmpp;
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////
    // D71

    void apply71(Tensors2* d, float c, float** s, float** x, float** y, 
      int n1, int n2) {
      float* C71 = new float[4];
      C71[0] =0.0f; C71[1] = 0.830893f; 
      C71[2] = -0.227266f; C71[3] =  0.042877f;
      const float c1 = C71[1], c2 = C71[2], c3 = C71[3];
      float di[3];
      float* g1 = new float[n1];
      for (int i2=0; i2<n2; ++i2) {
        int i2m3 = max(0,i2-3), i2p3 = min(n2-1,i2+3);
        int i2m2 = max(0,i2-2), i2p2 = min(n2-1,i2+2);
        int i2m1 = max(0,i2-1), i2p1 = min(n2-1,i2+1);
        float *xm1 = x[i2m1], *xm2 = x[i2m2], *xm3 = x[i2m3];
        float *xp1 = x[i2p1], *xp2 = x[i2p2], *xp3 = x[i2p3];
        float *ym1 = y[i2m1], *ym2 = y[i2m2], *ym3 = y[i2m3];
        float *yp1 = y[i2p1], *yp2 = y[i2p2], *yp3 = y[i2p3];
        gf(C71,x[i2],g1,3,n1); // nc=4-1=3
        for (int i1=0; i1<n1; ++i1) {
          d->getTensor(i1,i2,di);
          float csi = (s!=NULL)?c*s[i2][i1]:c;
          float d11 = di[0]*csi;
          float d12 = di[1]*csi;
          float d22 = di[2]*csi;
          float x1 = g1[i1];
          float x2 = c1*(xp1[i1]-xm1[i1]) +
                     c2*(xp2[i1]-xm2[i1]) +
                     c3*(xp3[i1]-xm3[i1]);
          float y1 = d11*x1+d12*x2;
          float y2 = d12*x1+d22*x2;
          g1[i1] = y1;
          float c1y2 = c1*y2; yp1[i1] += c1y2; ym1[i1] -= c1y2;
          float c2y2 = c2*y2; yp2[i1] += c2y2; ym2[i1] -= c2y2;
          float c3y2 = c3*y2; yp3[i1] += c3y2; ym3[i1] -= c3y2;
        }
        gt(C71,g1,y[i2],3,n1);
      }
      delete [] g1; delete [] C71;
    }

    void apply71(int i3, Tensors3* d, float c, float*** s, float*** x, 
      float*** y, int n1, int n2, int n3) {
      float* C71 = new float[4];
      C71[0] =0.0f; C71[1] = 0.830893f; 
      C71[2] = -0.227266f; C71[3] =  0.042877f;
      const float c1 = C71[1], c2 = C71[2], c3 = C71[3];
      float di[6];
      int i3m3 = max(0,i3-3), i3p3 = min(n3-1,i3+3);
      int i3m2 = max(0,i3-2), i3p2 = min(n3-1,i3+2);
      int i3m1 = max(0,i3-1), i3p1 = min(n3-1,i3+1);
      float** g1 = new float*[n2];
      float** g2 = new float*[n2];
      for (int i2=0; i2<n2; ++i2) {
        g1[i2] = new float[n1];
        g2[i2] = new float[n1];
      }
      gf(C71,x[i3],g1,g2,3,n1,n2); // nc=4-1=3
      for (int i2=0; i2<n2; ++i2) {
        float *xm1 = x[i3m1][i2], *xm2 = x[i3m2][i2], *xm3 = x[i3m3][i2];
        float *xp1 = x[i3p1][i2], *xp2 = x[i3p2][i2], *xp3 = x[i3p3][i2];
        float *ym1 = y[i3m1][i2], *ym2 = y[i3m2][i2], *ym3 = y[i3m3][i2];
        float *yp1 = y[i3p1][i2], *yp2 = y[i3p2][i2], *yp3 = y[i3p3][i2];
        float *g1i = g1[i2], *g2i = g2[i2];
        for (int i1=0; i1<n1; ++i1) {
          d->getTensor(i1,i2,i3,di);
          float csi = (s!=NULL)?c*s[i3][i2][i1]:c;
          float d11 = di[0]*csi;
          float d12 = di[1]*csi;
          float d13 = di[2]*csi;
          float d22 = di[3]*csi;
          float d23 = di[4]*csi;
          float d33 = di[5]*csi;
          float x1 = g1i[i1];
          float x2 = g2i[i1];
          float x3 = c1*(xp1[i1]-xm1[i1]) +
                     c2*(xp2[i1]-xm2[i1]) +
                     c3*(xp3[i1]-xm3[i1]);
          float y1 = d11*x1+d12*x2+d13*x3;
          float y2 = d12*x1+d22*x2+d23*x3;
          float y3 = d13*x1+d23*x2+d33*x3;
          g1i[i1] = y1;
          g2i[i1] = y2;
          float c1y3 = c1*y3; yp1[i1] += c1y3; ym1[i1] -= c1y3;
          float c2y3 = c2*y3; yp2[i1] += c2y3; ym2[i1] -= c2y3;
          float c3y3 = c3*y3; yp3[i1] += c3y3; ym3[i1] -= c3y3;
        }
      }
      gt(C71,g1,g2,y[i3],3,n1,n2);
      delete [] g1; delete [] g2; delete [] C71;
    }

    ///////////////////////////////////////////////////////////////////////////
    // D91

    void apply91(Tensors2* d, float c, float** s, float** x, float** y, 
      int n1, int n2) {
      float C91[5] = {0.0f, 0.8947167f, -0.3153471f, 0.1096895f, -0.0259358f};
      float c1 = C91[1], c2 = C91[2], c3 = C91[3], c4 = C91[4];
      float di[3];
      int i2m4,i2m3=0,i2m2=0,i2m1=0,i2p0=0,i2p1=0,i2p2=1,i2p3=2,i2p4=3;
      for (int i2=0; i2<n2; ++i2) {
        i2m4 = i2m3; i2m3 = i2m2; i2m2 = i2m1; i2m1 = i2p0;
        i2p0 = i2p1; i2p1 = i2p2; i2p2 = i2p3; i2p3 = i2p4; ++i2p4;
        if (i2p1>=n2) i2p1 = n2-1;
        if (i2p2>=n2) i2p2 = n2-1;
        if (i2p3>=n2) i2p3 = n2-1;
        if (i2p4>=n2) i2p4 = n2-1;
        float *xm4 = x[i2m4], *xm3 = x[i2m3], *xm2 = x[i2m2], *xm1 = x[i2m1];
        float *xp4 = x[i2p4], *xp3 = x[i2p3], *xp2 = x[i2p2], *xp1 = x[i2p1];
        float *xp0 = x[i2p0];
        float *ym4 = y[i2m4], *ym3 = y[i2m3], *ym2 = y[i2m2], *ym1 = y[i2m1];
        float *yp4 = y[i2p4], *yp3 = y[i2p3], *yp2 = y[i2p2], *yp1 = y[i2p1];
        float *yp0 = y[i2p0];
        int m4,m3=0,m2=0,m1=0,p0=0,p1=0,p2=1,p3=2,p4=3;
        for (int i1=0; i1<n1; ++i1) {
          m4 = m3; m3 = m2; m2 = m1; m1 = p0;
          p0 = p1; p1 = p2; p2 = p3; p3 = p4; ++p4;
          if (p1>=n1) p1 = n1-1;
          if (p2>=n1) p2 = n1-1;
          if (p3>=n1) p3 = n1-1;
          if (p4>=n1) p4 = n1-1;
          d->getTensor(i1,i2,di);
          float csi = (s!=NULL)?c*s[i2][i1]:c;
          float d11 = di[0]*csi;
          float d12 = di[1]*csi;
          float d22 = di[2]*csi;
          float x1 = c1*(xp0[p1]-xp0[m1]) +
                     c2*(xp0[p2]-xp0[m2]) +
                     c3*(xp0[p3]-xp0[m3]) +
                     c4*(xp0[p4]-xp0[m4]);
          float x2 = c1*(xp1[p0]-xm1[p0]) +
                     c2*(xp2[p0]-xm2[p0]) +
                     c3*(xp3[p0]-xm3[p0]) +
                     c4*(xp4[p0]-xm4[p0]);
          float y1 = d11*x1+d12*x2;
          float y2 = d12*x1+d22*x2;
          float c1y1 = c1*y1; yp0[p1] += c1y1; yp0[m1] -= c1y1;
          float c2y1 = c2*y1; yp0[p2] += c2y1; yp0[m2] -= c2y1;
          float c3y1 = c3*y1; yp0[p3] += c3y1; yp0[m3] -= c3y1;
          float c4y1 = c4*y1; yp0[p4] += c4y1; yp0[m4] -= c4y1;
          float c1y2 = c1*y2; yp1[p0] += c1y2; ym1[p0] -= c1y2;
          float c2y2 = c2*y2; yp2[p0] += c2y2; ym2[p0] -= c2y2;
          float c3y2 = c3*y2; yp3[p0] += c3y2; ym3[p0] -= c3y2;
          float c4y2 = c4*y2; yp4[p0] += c4y2; ym4[p0] -= c4y2;
        } 
      }
    }

    void apply(int i3, Tensors3* d, float c, float*** s, float*** x, 
      float*** y, int n1, int n2, int n3) {
      switch (_stencil) {
        case D21: {apply21(i3,c,s,x,y,n1,n2);break;}
        case D22: {apply22(i3,d,c,s,x,y,n1,n2); break;}
        case D24: {
          cout<<"Stencil.D24 not supported for 3D arrays"<<endl;
          return;
        }
        case D33: {apply33(i3,d,c,s,x,y,n1,n2); break;}
        case D71: {apply71(i3,d,c,s,x,y,n1,n2,n3); break;}
        default: {
          cout<<"Stencil.D91 not supported for 3D arrays"<<endl;
          return;
        }
      }
    }

    void applySerial(int i3start, int i3step, int i3stop,
      Tensors3* d, float c, float*** s, float*** x, float*** y, 
      int n1, int n2, int n3) {
      for (int i3=i3start; i3<i3stop; i3+=i3step)
        apply(i3,d,c,s,x,y,n1,n2,n3);
    }

    public:
    LocalDiffusionKernel() {LocalDiffusionKernel(D22);}
    LocalDiffusionKernel(Stencil s) {_stencil=s;_npass=1;_parallel=false;}
    Stencil getStencil() {return _stencil;}

    /**
     * Sets the number of kernel passes in each apply of this filter.
     * For example, if npass = 2, then the output is computed in two
     * passes: (1) y += G'DGx, (2) y += G'DGy.
     * The default is one pass.
     */
    void setNumberOfPasses(int npass) {_npass = npass;}
    /**
     * Applies this filter for specified tensor coefficients and scale factors.
     * @param d tensor coefficients.
     * @param c constant scale factor for tensor coefficients.
     * @param s array of scale factors for tensor coefficients.
     * @param x input array.
     * @param y output array.
     */
    void apply(Tensors2* d, float c, float** s, float** x, float** y, 
      int n1, int n2) {
      for (int ipass=0; ipass<_npass; ++ipass) {
        if (ipass>0) x = copy(y,n1,n2);
        if (d==NULL) d = &IDENTITY_TENSORS2;
        switch (_stencil) {
          case D21: {apply21(c,s,x,y,n1,n2); break;}
          case D22: {apply22(d,c,s,x,y,n1,n2); break;}
          case D24: {apply24(d,c,s,x,y,n1,n2); break;}
          case D33: {apply33(d,c,s,x,y,n1,n2); break;}
          case D71: {apply71(d,c,s,x,y,n1,n2); break;}
          default: {apply91(d,c,s,x,y,n1,n2); break;}
        }
      }
    }
    /**
     * Applies this filter for constant isotropic identity tensor.
     * @param x input array.
     * @param y output array.
     */
    void apply(float** x, float** y, int n1, int n2){
      apply(NULL,1.0f,x,y,n1,n2);
    }
    void apply(Tensors2* d, float** x, float** y, int n1, int n2) {
      apply(d,1.0f,x,y,n1,n2);
    }
    void apply(float c, float** x, float** y, int n1, int n2) {
      apply(NULL,c,NULL,x,y,n1,n2);
    }
    void apply(Tensors2* d, float c, float** x, float** y, int n1, int n2) {
      apply(d,c,NULL,x,y,n1,n2);
    }
    /**
     * Applies this filter for specified scale factors.
     * Uses a constant isotropic identity tensor.
     * @param c constant scale factor for tensor coefficients.
     * @param s array of scale factors for tensor coefficients.
     * @param x input array.
     * @param y output array.
     */
    void apply(float c, float** s, float** x, float** y, int n1, int n2) {
      apply(NULL,c,s,x,y,n1,n2);
    }
    
    // 3D
    void apply(Tensors3* d, float c, float*** s, float*** x, float*** y, 
      int n1, int n2, int n3) {
      int i3start = 0, i3step = 1, i3stop = n3;
      for (int ipass=0; ipass<_npass; ++ipass) {
        if (ipass>0) x = copy(y,n1,n2,n3);
        if (d==NULL) d = &IDENTITY_TENSORS3;
        switch (_stencil) {
          case D21: {i3start=0; i3step=2; i3stop=n3; break;}
          case D22: {i3start=1; i3step=2; i3stop=n3; break;}
          case D24: {i3start=1; i3step=4; i3stop=n3; break;}
          case D33: {i3start=1; i3step=3; i3stop=n3-1; break;}
          case D71: {i3start=0; i3step=7; i3stop=n3; break;}
          default: {break;}
        }
        applySerial(i3start,1,i3stop,d,c,s,x,y,n1,n2,n3);
      }
   }
   /**
    * Applies this filter for a constant isotropic identity tensor.
    * @param x input array.
     * @param y output array.
     */
    void apply(float*** x, float*** y, int n1, int n2, int n3) {
      apply(NULL,1.0f,x,y,n1,n2,n3);
    }
    /**
     * Applies this filter for specified tensor coefficients.
     * @param d tensor coefficients.
     * @param x input array.
     * @param y output array.
     */
    void apply(Tensors3* d, float*** x, float*** y, int n1, int n2, int n3) {
      apply(d,1.0f,x,y,n1,n2,n3);
    }
    void apply(float c, float*** x, float*** y, int n1, int n2, int n3) {
      apply(NULL,c,NULL,x,y,n1,n2,n3);
    }
    void apply(Tensors3* d, float c, float*** x, float*** y, 
      int n1, int n2, int n3) {
      apply(d,c,NULL,x,y,n1,n2,n3);
    }
    void apply(float c, float*** s, float*** x, float*** y, 
      int n1, int n2, int n3) {
      apply(NULL,c,s,x,y,n1,n2,n3);
    }
    
  };
}

#endif

