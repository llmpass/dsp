#ifndef _LOCALSMOOTHINGFILTER_H
#define _LOCALSMOOTHINGFILTER_H

#include <iostream>

#include "Tensors2.h"
#include "Tensors3.h"
#include "BandPassFilter.h"
#include "LocalDiffusionKernel.h"
#include "../util/ArrayMath.h"
using namespace util;

namespace dsp {
  /**
   * Local smoothing of images with tensor filter coefficients.
   * Smoothing is performed by solving a sparse symmetric positive-definite
   * (SPD) system of equations: (I+G'DG)y = x, where G is a gradient operator,
   * D is an SPD tensor field, x is an input image, and y is an output image.
   * <p>
   * The sparse system of filter equations (I+G'DG)y = x is solved iteratively,
   * beginning with y = x. Iterations continue until either the error in the
   * solution y is below a specified threshold or the number of iterations
   * exceeds a specified limit.
   * <p>
   * For low wavenumbers the output of this filter approximates the solution
   * to an anisotropic inhomogeneous diffusion equation, where the filter
   * input x corresponds to the initial condition at time t = 0 and filter
   * output y corresponds to the solution at some later time t.
   * <p>
   * Additional smoothing filters may be applied to the input image x before
   * or after solving the sparse system of equations for the smoothed output
   * image y. These additional filters compensate for deficiencies in the
   * gradient operator G, which is a finite-difference approximation that is
   * poor for high wavenumbers near the Nyquist limit. The extra smoothing
   * filters attenuate these high wavenumbers.
   * <p>
   * The additional smoothing filter S is a simple 3x3 (or, in 3D, 3x3x3)
   * weighted-average filter that zeros Nyquist wavenumbers. This filter
   * is fast and has non-negative coefficients. However, it may smooth too
   * much, as it attenuates all non-zero wavenumbers, not only the highest
   * wavenumbers. Moreover, this filter is not isotropic.
   * <p>
   * The other additional smoothing operator L is an isotropic low-pass
   * filter designed to pass wavenumbers up to a specified maximum.
   * Although slower than S, the cost of applying L to the input image x is
   * likely to be insignificant relative to the cost of solving the sparse
   * system of equations for the output image y.
   *
   * @author Luming Liang, translated from Java Mines Toolkit by Dave Hale
   * @version 2012.06.28
   */

  class LocalSmoothingFilter {

  public:
     /**
     * Constructs a local smoothing filter with default parameters.
     * The default parameter small is 0.01 and the default maximum
     * number of iterations is 100. Uses a default 2x2 stencil for the
     * derivatives in the operator G.
     */
    LocalSmoothingFilter() {
      new (this) LocalSmoothingFilter(0.01,100);
    }

    /**
     * Constructs a local smoothing filter with specified iteration parameters.
     * Uses a default 2x2 stencil for the derivatives in the operator G.
     * @param small stop when norm of residuals is less than this factor
     * times the norm of the input array.
     * @param niter stop when number of iterations exceeds this limit.
     */
    LocalSmoothingFilter(double small, int niter) {
      _small = (float)small;
      _niter = niter;
      _ldk = new LocalDiffusionKernel(LocalDiffusionKernel::D22);
    }

    /**
     * Constructs a local smoothing filter with specified parameters.
     * @param small stop when norm of residuals is less than this factor
     * times the norm of the input array.
     * @param niter stop when number of iterations exceeds this limit.
     * @param ldk the local diffusion kernel that computes y += (I+G'DG)x.
     */
    LocalSmoothingFilter(double small, int niter, LocalDiffusionKernel* ldk) {
      _small = (float)small;
      _niter = niter;
      _ldk = ldk;
    }

    /**
     * Sets the use of a preconditioner in this local smoothing filter.
     * A preconditioner requires extra memory and more computing time
     * per iteration, but may result in fewer iterations.
     * The default is to not use a preconditioner.
     * @param pc true, to use a preconditioner; false, otherwise.
     */
    void setPreconditioner(bool pc) {
      _pc = pc;
    }

    /**
     * Applies this filter for specified constant scale factor.
     * Local smoothing for 1D arrays is a special case that requires no tensors.
     * All tensors are implicitly scalar values equal to one, so that filtering
     * is determined entirely by the specified constant scale factor.
     * @param c constant scale factor.
     * @param x input array.
     * @param y output array.
     */
    void apply(float c, float* x, float* y, int n) {
      apply(c,NULL,x,y,n);
    }

    /**
     * Applies this filter for specified scale factors.
     * Local smoothing for 1D arrays is a special case that requires no tensors.
     * All tensors are implicitly scalar values equal to one, so that filtering
     * is determined entirely by the specified scale factors.
     * @param c constant scale factor.
     * @param s array of scale factors.
     * @param x input array.
     * @param y output array.
     */
    void apply(float c, float* s, float* x, float* y, int n1) {
      // Sub-diagonal e of SPD tridiagonal matrix I+G'DG; e[0] = e[n1] = 0.0.
      float* e = new float[n1+1];
      if (s!=NULL) {
        c = -0.5f*c;
        for (int i1=1; i1<n1; ++i1) e[i1] = c*(s[i1]+s[i1-1]);
      } else {
        c = -c;
        for (int i1=1; i1<n1; ++i1) e[i1] = c;
      }
      // Work array w overwrites sub-diagonal array e.
      float* w = e;
      // Solve tridiagonal system of equations (I+G'DG)y = x.
      float t = 1.0f-e[0]-e[1];
      y[0] = x[0]/t;
      for (int i1=1; i1<n1; ++i1) {
        float di = 1.0f-e[i1]-e[i1+1]; // diagonal element
        float ei = e[i1]; // sub-diagonal element
        w[i1] = ei/t;
        t = di-ei*w[i1];
        y[i1] = (x[i1]-ei*y[i1-1])/t;
      }
      for (int i1=n1-1; i1>0; --i1) y[i1-1] -= w[i1]*y[i1];
    }

    /**
     * Applies this filter for identity tensors.
     * @param x input array.
     * @param y output array.
     */
    void apply(float** x, float** y, int n1, int n2) {
      apply(NULL,1.0f,NULL,x,y,n1,n2);
    }

    /**
     * Applies this filter for identity tensors and specified scale factor.
     * @param c constant scale factor.
     * @param x input array.
     * @param y output array.
     */
    void apply(float c, float** x, float** y, int n1, int n2) {
      apply(NULL,c,NULL,x,y,n1,n2);
    }

    /**
     * Applies this filter for identity tensors and specified scale factors.
     * @param c constant scale factor.
     * @param s array of scale factors.
     * @param x input array.
     * @param y output array.
     */
    void apply(float c, float** s, float** x, float** y, int n1, int n2) {
      apply(NULL,c,s,x,y,n1,n2);
    }

    /**
     * Applies this filter for specified tensors.
     * @param d tensors.
     * @param x input array.
     * @param y output array.
     */
    void apply(Tensors2* d, float** x, float** y, int n1, int n2) {
      apply(d,1.0f,NULL,x,y,n1,n2);
    }

    /**
     * Applies this filter for specified tensors and scale factor.
     * @param d tensors.
     * @param c constant scale factor for tensors.
     * @param x input array.
     * @param y output array.
     */
    void apply(Tensors2* d, float c, float** x, float** y, int n1, int n2) {
      apply(d,c,NULL,x,y,n1,n2);
    }

    /**
     * Applies this filter for specified tensors and scale factors.
     * @param d tensors.
     * @param c constant scale factor for tensors.
     * @param s array of scale factors for tensors.
     * @param x input array.
     * @param y output array.
     */
    void apply(Tensors2* d, float c, float** s, float** x, float** y, 
      int n1, int n2) {
      Operator2* a = new A2(_ldk,d,c,s);
      copy(x,y,n1,n2);
      if (_pc) {
        Operator2* m = new M2(d,c,s,x,n1,n2);
        solve(a,m,x,y,n1,n2);
      } else solve(a,x,y,n1,n2);
    }

    /**
     * Applies this filter for identity tensors.
     * @param x input array.
     * @param y output array.
     */
    void apply(float*** x, float*** y, int n1, int n2, int n3) {
      apply(NULL,1.0f,NULL,x,y,n1,n2,n3);
    }

    /**
     * Applies this filter for identity tensors and specified scale factor.
     * @param c constant scale factor.
     * @param x input array.
     * @param y output array.
     */
    void apply(float c, float*** x, float*** y, int n1, int n2, int n3) {
      apply(NULL,c,NULL,x,y,n1,n2,n3);
    }

    /**
     * Applies this filter for identity tensors and specified scale factors.
     * @param c constant scale factor.
     * @param s array of scale factors.
     * @param x input array.
     * @param y output array.
     */
    void apply(float c, float*** s, float*** x, float*** y, 
      int n1, int n2, int n3) {
      apply(NULL,c,s,x,y,n1,n2,n3);
    }

    /**
     * Applies this filter for specified tensors.
     * @param d tensors.
     * @param x input array.
     * @param y output array.
     */
    void apply(Tensors3* d, float*** x, float*** y, int n1, int n2, int n3) {
      apply(d,1.0f,NULL,x,y,n1,n2,n3);
    }

    /**
     * Applies this filter for specified tensors and scale factor.
     * @param d tensors.
     * @param c constant scale factor for tensors.
     * @param x input array.
     * @param y output array.
     */
    void apply(Tensors3* d, float c, float*** x, float*** y, 
      int n1, int n2, int n3) {
      apply(d,c,NULL,x,y,n1,n2,n3);
    }

    /**
     * Applies this filter for specified tensors and scale factors.
     * @param d tensors.
     * @param c constant scale factor for tensors.
     * @param s array of scale factors for tensors.
     * @param x input array.
     * @param y output array.
     */
    void apply(Tensors3* d, float c, float*** s, float*** x, float*** y,
      int n1, int n2, int n3) {
      Operator3* a = new A3(_ldk,d,c,s);
      copy(x,y,n1,n2,n3);
      if (_pc) {
        Operator3* m = new M3(d,c,s,x,n1,n2,n3);
        solve(a,m,x,y,n1,n2,n3);
      } else solve(a,x,y,n1,n2,n3);
    }

    /**
     * Applies a simple 3x3 weighted-average smoothing filter S.
     * Input and output arrays x and y may be the same array.
     * @param x input array.
     * @param y output array.
     */
    void applySmoothS(float** x, float** y, int n1, int n2) {
      smoothS(x,y,n1,n2);
    }

    /**
     * Applies a simple 3x3x3 weighted-average smoothing filter S.
     * Input and output arrays x and y may be the same array.
     * @param x input array.
     * @param y output array.
     */
    void applySmoothS(float*** x, float*** y, int n1, int n2, int n3) {
      smoothS(x,y,n1,n2,n3);
    }

    /**
     * Applies an isotropic low-pass smoothing filter L.
     * Input and output arrays x and y may be the same array.
     * @param kmax maximum wavenumber not attenuated, in cycles/sample.
     * @param x input array.
     * @param y output array.
     */
    void applySmoothL(double kmax, float** x, float** y, int n1, int n2) {
      smoothL(kmax,x,y,n1,n2);
    }

    /**
     * Applies an isotropic low-pass smoothing filter L.
     * Input and output arrays x and y may be the same array.
     * @param kmax maximum wavenumber not attenuated, in cycles/sample.
     * @param x input array.
     * @param y output array.
     */
    void applySmoothL(double kmax, float*** x, float*** y, 
      int n1, int n2, int n3) {
      smoothL(kmax,x,y,n1,n2,n3);
    }

    ///////////////////////////////////////////////////////////////////////////
    // private

    private: 
    //static const bool PARALLEL = true; // false for single-threaded

    //private static Logger log =
      //Logger.getLogger(LocalSmoothingFilter.class.getName());

    float _small; // stop iterations when residuals are small
    int _niter; // number of iterations
    bool _pc; // true, for preconditioned CG iterations
    LocalDiffusionKernel* _ldk; // computes y += (I+G'DG)x
    BandPassFilter* _lpf; // lowpass filter, null until applied
    double _kmax; // maximum wavenumber for lowpass filter

    /*
     * A symmetric positive-definite operator.
     */
    class Operator2 {
      public: virtual void apply(float** x, float** y, int n1, int n2);
    };
    class Operator3 {
      public: virtual void 
        apply(float*** x, float*** y, int n1, int n2, int n3);
    };

    class A2 : public Operator2 {
      public:
      A2(LocalDiffusionKernel* ldk, Tensors2* d, float c, float** s) {
        _ldk = ldk;
        _d = d;
        _c = c;
        _s = s;
      }
      void apply(float** x, float** y, int n1, int n2) {
        copy(x,y,n1,n2);
        _ldk->apply(_d,_c,_s,x,y,n1,n2);
      }
      LocalDiffusionKernel* _ldk;
      Tensors2* _d;
      float _c;
      float** _s;
    };

    class M2 : public Operator2 {
      public:
      M2(Tensors2* d, float c, float** s, float** x, int n1, int n2) {
        _p = fillfloat(1.0f,n1,n2);
        c *= 0.25f;
        float* di = new float[3];
        for (int i2=1,m2=0; i2<n2; ++i2,++m2) {
          for (int i1=1,m1=0; i1<n1; ++i1,++m1) {
            float si = s!=NULL?s[i2][i1]:1.0f;
            float csi = c*si;
            float d11 = csi;
            float d12 = 0.0f;
            float d22 = csi;
            if (d!=NULL) {
              d->getTensor(i1,i2,di);
              d11 = di[0]*csi;
              d12 = di[1]*csi;
              d22 = di[2]*csi;
            }
            _p[i2][i1] += (d11+d12)+( d12+d22);
            _p[m2][m1] += (d11+d12)+( d12+d22);
            _p[i2][m1] += (d11-d12)+(-d12+d22);
            _p[m2][i1] += (d11-d12)+(-d12+d22);
          }
        }
        div(1.0f,_p,_p,n1,n2);
      }
      void apply(float** x, float** y, int n1, int n2) {
        sxy(_p,x,y,n1,n2);
      }
      private: float** _p;
    };

    class A3 : public Operator3 {
      public:
      A3(LocalDiffusionKernel* ldk, Tensors3* d, float c, float*** s) {
        _ldk = ldk;
        _d = d;
        _c = c;
        _s = s;
      }
      void apply(float*** x, float*** y, int n1, int n2, int n3) {
        copy(x,y,n1,n2,n3);
        _ldk->apply(_d,_c,_s,x,y,n1,n2,n3);
      }
      LocalDiffusionKernel* _ldk;
      Tensors3* _d;
      float _c;
      float*** _s;
    };

    class M3 : public Operator3 {
      public:
      M3(Tensors3* d, float c, float*** s, float*** x, int n1, int n2, int n3) {
        _p = fillfloat(1.0f,n1,n2,n3);
        c *= 0.0625f;
        float* di = new float[6];
        for (int i3=1,m3=0; i3<n3; ++i3,++m3) {
          for (int i2=1,m2=0; i2<n2; ++i2,++m2) {
            for (int i1=1,m1=0; i1<n1; ++i1,++m1) {
              float si = s!=NULL?s[i3][i2][i1]:1.0f;
              float csi = c*si;
              float d11 = csi;
              float d12 = 0.0f;
              float d13 = 0.0f;
              float d22 = csi;
              float d23 = 0.0f;
              float d33 = csi;
              if (d!=NULL) {
                d->getTensor(i1,i2,i3,di);
                d11 = di[0]*csi;
                d12 = di[1]*csi;
                d13 = di[2]*csi;
                d22 = di[3]*csi;
                d23 = di[4]*csi;
                d33 = di[5]*csi;
              }
              _p[i3][i2][i1] += ( d11+d12+d13)+( d12+d22+d23)+( d13+d23+d33);
              _p[m3][m2][m1] += ( d11+d12+d13)+( d12+d22+d23)+( d13+d23+d33);
              _p[i3][m2][i1] += ( d11-d12+d13)+(-d12+d22-d23)+( d13-d23+d33);
              _p[m3][i2][m1] += ( d11-d12+d13)+(-d12+d22-d23)+( d13-d23+d33);
              _p[m3][i2][i1] += ( d11+d12-d13)+( d12+d22-d23)+(-d13-d23+d33);
              _p[i3][m2][m1] += ( d11+d12-d13)+( d12+d22-d23)+(-d13-d23+d33);
              _p[m3][m2][i1] += ( d11-d12-d13)+(-d12+d22+d23)+(-d13+d23+d33);
              _p[i3][i2][m1] += ( d11-d12-d13)+(-d12+d22+d23)+(-d13+d23+d33);
            }
          }
        }
        div(1.0f,_p,_p,n1,n2,n3);
      }
      void apply(float*** x, float*** y, int n1, int n2, int n3) {
        sxy(_p,x,y,n1,n2,n3);
      }
      private: float*** _p;
    };

    /*
     * Computes y = lowpass(x). Arrays x and y may be the same array.
     */
    void smoothL(double kmax, float** x, float** y, int n1, int n2) {
      ensureLowpassFilter(kmax);
      _lpf->apply(x,y,n1,n2);
    }
    void smoothL(double kmax, float*** x, float*** y, int n1, int n2, int n3) {
      ensureLowpassFilter(kmax);
      _lpf->apply(x,y,n1,n2,n3);
    }
    void ensureLowpassFilter(double kmax) {
      if (_lpf==NULL || _kmax!=kmax) {
        _kmax = kmax;
        double kdelta = 0.5-kmax;
        double kupper = kmax+0.5*kdelta;
        _lpf = new BandPassFilter(0.0,kupper,kdelta,0.01);
        _lpf->setExtrapolation(BandPassFilter::ZERO_SLOPE);
        _lpf->setFilterCaching(false);
      }
    }

    /*
     * Computes y = S'Sx. Arrays x and y may be the same array.
     */
    static void smoothS(float** x, float** y, int n1, int n2) {
      int n1m = n1-1;
      int n2m = n2-1;
      float** t = new float*[3];
      for (int i=0; i<3; ++i) t[i] = new float[n1];
      copy(x[0],t[0],n1);
      copy(x[0],t[1],n1);
      for (int i2=0; i2<n2; ++i2) {
        int i2m = (i2>0)?i2-1:0;
        int i2p = (i2<n2m)?i2+1:n2m;
        int j2m = i2m%3;
        int j2 = i2%3;
        int j2p = i2p%3;
        copy(x[i2p],t[j2p],n1);
        float* x2m = t[j2m];
        float* x2p = t[j2p];
        float* x20 = t[j2];
        float* y2 = y[i2];
        for (int i1=0; i1<n1; ++i1) {
          int i1m = (i1>0)?i1-1:0;
          int i1p = (i1<n1m)?i1+1:n1m;
          y2[i1] = 0.2500f*(x20[i1 ]) +
                   0.1250f*(x20[i1m]+x20[i1p]+x2m[i1 ]+x2p[i1 ]) +
                   0.0625f*(x2m[i1m]+x2m[i1p]+x2p[i1m]+x2p[i1p]);
        }
        for (int i=0; i<3; ++i) delete [] t[i];
        delete [] t;
      }
    }
    static void xsmoothS(float** x, float** y, int n1, int n2) {
      float** t = new float*[2];
      for (int i=0; i<2; ++i) t[i] = new float[n1];
      copy(x[0],t[0],n1);
      zero(y[0],n1);
      for (int i2=1,i2m=0; i2<n2; ++i2,++i2m) {
        int j2 = (i2 )%2;
        int j2m = (i2m)%2;
        copy(x[i2],t[j2],n1);
        zero(y[i2],n1);
        float* x0 = t[j2 ];
        float* x1 = t[j2m];
        float* y0 = y[i2 ];
        float* y1 = y[i2m];
        for (int i1=1,i1m=0; i1<n1; ++i1,++i1m) {
          float x00 = x0[i1 ];
          float x01 = x0[i1m];
          float x10 = x1[i1 ];
          float x11 = x1[i1m];
          // 0.0625 = 1/16
          float xs = 0.0625f*(x00+x01+x10+x11);
          y0[i1 ] += xs;
          y0[i1m] += xs;
          y1[i1 ] += xs;
          y1[i1m] += xs;
        }
      }
      for (int i=0; i<2; ++i) delete [] t[i];
      delete [] t;
    }

    /*
     * Computes y = S'Sx. Arrays x and y may be the same array.
     */
    static void smoothS(float*** x, float*** y, int n1, int n2, int n3) {
      int n1m = n1-1;
      int n2m = n2-1;
      int n3m = n3-1;
      float*** t = new float**[3];
      for (int i3=0; i3<n3; ++i3) {
        t[i3] = new float*[n2];
        for (int i2=0; i2<n2; ++i2) t[i3][i2] = new float[n1];
      }
      copy(x[0],t[0],n1,n2);
      copy(x[0],t[1],n1,n2);
      for (int i3=0; i3<n3; ++i3) {
        int i3m = (i3>0)?i3-1:0;
        int i3p = (i3<n3m)?i3+1:n3m;
        int j3m = i3m%3;
        int j3 = i3%3;
        int j3p = i3p%3;
        copy(x[i3p],t[j3p],n1,n2);
        float** x3m = t[j3m];
        float** x3p = t[j3p];
        float** x30 = t[j3];
        float** y30 = y[i3];
        for (int i2=0; i2<n2; ++i2) {
          int i2m = (i2>0)?i2-1:0;
          int i2p = (i2<n2m)?i2+1:n2m;
          float* x3m2m = x3m[i2m];
          float* x3m20 = x3m[i2 ];
          float* x3m2p = x3m[i2p];
          float* x302m = x30[i2m];
          float* x3020 = x30[i2 ];
          float* x302p = x30[i2p];
          float* x3p2m = x3p[i2m];
          float* x3p20 = x3p[i2 ];
          float* x3p2p = x3p[i2p];
          float* y3020 = y30[i2 ];
          for (int i1=0; i1<n1; ++i1) {
            int i1m = (i1>0)?i1-1:0;
            int i1p = (i1<n1m)?i1+1:n1m;
            y3020[i1] = 0.125000f*(x3020[i1 ]) +
                        0.062500f*(x3020[i1m]+x3020[i1p]+
                                   x302m[i1 ]+x302p[i1 ]+
                                   x3m20[i1 ]+x3p20[i1 ]) +
                        0.031250f*(x3m20[i1m]+x3m20[i1p]+
                                   x3m2m[i1 ]+x3m2p[i1 ]+
                                   x302m[i1m]+x302m[i1p]+
                                   x302p[i1m]+x302p[i1p]+
                                   x3p20[i1m]+x3p20[i1p]+
                                   x3p2m[i1 ]+x3p2p[i1 ]) +
                        0.015625f*(x3m2m[i1m]+x3m2m[i1p]+
                                   x3m2p[i1m]+x3m2p[i1p]+
                                   x3p2m[i1m]+x3p2m[i1p]+
                                   x3p2p[i1m]+x3p2p[i1p]);
          }
        }
      }
      for (int i3=0; i3<3; ++i3) {
        for (int i2=0; i2<n2; ++i2) delete [] t[i3][i2];
        delete [] t[i3];
      }
      delete [] t;
    }
    static void xsmoothS(float*** x, float*** y, int n1, int n2, int n3) {
      float*** t = new float**[2];
      for (int i3=0; i3<2; ++i3) {
        t[i3] = new float*[n2];
        for (int i2=0; i2<n2; ++i2) t[i3][i2] = new float[n1];
      }
      copy(x[0],t[0],n1,n2);
      zero(y[0],n1,n2);
      for (int i3=1,i3m=0; i3<n3; ++i3,++i3m) {
        int j3 = (i3 )%2;
        int j3m = (i3-1)%2;
        copy(x[i3],t[j3],n1,n2);
        zero(y[i3],n1,n2);
        for (int i2=1,i2m=0; i2<n2; ++i2,++i2m) {
          float* x00 = t[j3 ][i2 ];
          float* x01 = t[j3 ][i2m];
          float* x10 = t[j3m][i2 ];
          float* x11 = t[j3m][i2m];
          float* y00 = y[i3 ][i2 ];
          float* y01 = y[i3 ][i2m];
          float* y10 = y[i3m][i2 ];
          float* y11 = y[i3m][i2m];
          for (int i1=1,i1m=0; i1<n1; ++i1,++i1m) {
            float x000 = x00[i1 ];
            float x001 = x00[i1m];
            float x010 = x01[i1 ];
            float x011 = x01[i1m];
            float x100 = x10[i1 ];
            float x101 = x10[i1m];
            float x110 = x11[i1 ];
            float x111 = x11[i1m];
            // 0.015625 = 1/64
            float xs = 0.015625f*(x000+x001+x010+x011+x100+x101+x110+x111);
            y00[i1 ] += xs;
            y00[i1m] += xs;
            y01[i1 ] += xs;
            y01[i1m] += xs;
            y10[i1 ] += xs;
            y10[i1m] += xs;
            y11[i1 ] += xs;
            y11[i1m] += xs;
          }
        }  
      }
      for (int i3=0; i3<3; ++i3) {
        for (int i2=0; i2<n2; ++i2) delete [] t[i3][i2];
        delete [] t[i3];
      }
      delete [] t;
    }

    // Conjugate-gradient solution of Ax = b, with no preconditioner.
    // Uses the initial values of x; does not assume they are zero.
    void solve(Operator2* a, float** b, float** x, int n1, int n2) {
      float** d = new float*[n2];
      float** q = new float*[n2];
      float** r = new float*[n2];
      for (int i2=0; i2<n2; ++i2) {
        b[i2] = new float[n1];
        q[i2] = new float[n1];
        r[i2] = new float[n1];
      }
      copy(b,r,n1,n2);
      a->apply(x,q,n1,n2);
      saxpy(-1.0f,q,r,n1,n2); // r = b-Ax
      copy(r,d,n1,n2); // d = r
      float delta = sdot(r,r,n1,n2); // delta = r'r
      float bnorm = sqrt(sdot(b,b,n1,n2));
      float rnorm = sqrt(delta);
      float rnormBegin = rnorm;
      float rnormSmall = bnorm*_small;
      int iter;
      //log.fine("solve: bnorm="+bnorm+" rnorm="+rnorm);
      for (iter=0; iter<_niter && rnorm>rnormSmall; ++iter) {
        //log.finer(" iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
        a->apply(d,q,n1,n2); // q = Ad
        float dq = sdot(d,q,n1,n2); // d'q = d'Ad
        float alpha = delta/dq; // alpha = r'r/d'Ad
        saxpy( alpha,d,x,n1,n2); // x = x+alpha*d
        saxpy(-alpha,q,r,n1,n2); // r = r-alpha*q
        float deltaOld = delta;
        delta = sdot(r,r,n1,n2); // delta = r'r
        float beta = delta/deltaOld;
        sxpay(beta,r,d,n1,n2); // d = r+beta*d
        rnorm = sqrt(delta);
      }
      //log.fine(" iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
      for (int i2=0; i2<n2; ++i2) {
        delete [] d[i2];
        delete [] q[i2];
        delete [] r[i2];
      }
      delete [] d; delete [] q; delete [] r;
    }
    void solve(Operator3* a, float*** b, float*** x, int n1, int n2, int n3) {
      float*** d = new float**[n3];
      float*** q = new float**[n3];
      float*** r = new float**[n3];
      for (int i3=0; i3<n3; ++i3) {
        b[i3] = new float*[n2];
        q[i3] = new float*[n2];
        r[i3] = new float*[n2];
        for (int i2=0; i2<n2; ++i2) {
          b[i3][i2] = new float[n1];
          q[i3][i2] = new float[n1];
          r[i3][i2] = new float[n1];
        }
      }
      copy(b,r,n1,n2,n3); a->apply(x,q,n1,n2,n3); 
      saxpy(-1.0f,q,r,n1,n2,n3); // r = b-Ax
      copy(r,d,n1,n2,n3);
      float delta = sdot(r,r,n1,n2,n3);
      float bnorm = sqrt(sdot(b,b,n1,n2,n3));
      float rnorm = sqrt(delta);
      float rnormBegin = rnorm;
      float rnormSmall = bnorm*_small;
      int iter;
      //log.fine("solve: bnorm="+bnorm+" rnorm="+rnorm);
      for (iter=0; iter<_niter && rnorm>rnormSmall; ++iter) {
        //log.finer(" iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
        a->apply(d,q,n1,n2,n3);
        float dq = sdot(d,q,n1,n2,n3);
        float alpha = delta/dq;
        saxpy( alpha,d,x,n1,n2,n3);
        if (iter%100<99) {
          saxpy(-alpha,q,r,n1,n2,n3);
        } else {
          copy(b,r,n1,n2,n3); a->apply(x,q,n1,n2,n3); saxpy(-1.0f,q,r,n1,n2,n3);
        }
        float deltaOld = delta;
        delta = sdot(r,r,n1,n2,n3);
        float beta = delta/deltaOld;
        sxpay(beta,r,d,n1,n2,n3);
        rnorm = sqrt(delta);
      }
      //log.fine(" iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
      for (int i3=0; i3<n3; ++i3) {
        for (int i2=0; i2<n2; ++i2) {
          delete [] b[i3][i2];
          delete [] q[i3][i2];
          delete [] r[i3][i2];
        }
        delete [] b[i3];
        delete [] q[i3];
        delete [] r[i3];
      }
      delete [] b; delete [] q; delete [] r;
    }

    // Conjugate-gradient solution of Ax = b, with preconditioner M.
    // Uses the initial values of x; does not assume they are zero.
    void solve(Operator2* a, Operator2* m, float** b, float** x, 
      int n1, int n2) {
      float** d = new float*[n2];
      float** q = new float*[n2];
      float** r = new float*[n2];
      float** s = new float*[n2];
      for (int i2=0; i2<n2; ++i2) {
        d[i2] = new float[n1];
        q[i2] = new float[n1];
        r[i2] = new float[n1];
        s[i2] = new float[n1];
      }
      copy(b,r,n1,n2);
      a->apply(x,q,n1,n2);
      saxpy(-1.0f,q,r,n1,n2); // r = b-Ax
      float bnorm = sqrt(sdot(b,b,n1,n2));
      float rnorm = sqrt(sdot(r,r,n1,n2));
      float rnormBegin = rnorm;
      float rnormSmall = bnorm*_small;
      m->apply(r,s,n1,n2); // s = Mr
      copy(s,d,n1,n2); // d = s
      float delta = sdot(r,s,n1,n2); // r's = r'Mr
      int iter;
      //log.fine("msolve: bnorm="+bnorm+" rnorm="+rnorm);
      for (iter=0; iter<_niter && rnorm>rnormSmall; ++iter) {
        //log.finer(" iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
        a->apply(d,q,n1,n2); // q = Ad
        float alpha = delta/sdot(d,q,n1,n2); // alpha = r'Mr/d'Ad
        saxpy( alpha,d,x,n1,n2); // x = x+alpha*d
        saxpy(-alpha,q,r,n1,n2); // r = r-alpha*q
        m->apply(r,s,n1,n2); // s = Mr
        float deltaOld = delta;
        delta = sdot(r,s,n1,n2); // delta = r's = r'Mr
        float beta = delta/deltaOld;
        sxpay(beta,s,d,n1,n2); // d = s+beta*d
        rnorm = sqrt(sdot(r,r,n1,n2));
      }
      //log.fine(" iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
      for (int i2=0; i2<n2; ++i2) {
        delete [] d[i2];
        delete [] q[i2];
        delete [] r[i2];
        delete [] s[i2];
      }
      delete [] d;
      delete [] q;
      delete [] r;
      delete [] s;
    }
    void solve(Operator3* a, Operator3* m, float*** b, float*** x, 
      int n1, int n2, int n3) {
      float*** d = new float**[n3];
      float*** q = new float**[n3];
      float*** r = new float**[n3];
      float*** s = new float**[n3];
      for (int i3=0; i3<n3; ++i3) {
        d[i3] = new float*[n2];
        q[i3] = new float*[n2];
        r[i3] = new float*[n2];
        s[i3] = new float*[n2];
        for (int i2=0; i2<n2; ++i2) {
          d[i3][i2] = new float[n1];
          q[i3][i2] = new float[n1];
          r[i3][i2] = new float[n1];
          s[i3][i2] = new float[n1];
        }
      }
      copy(b,r,n1,n2,n3); a->apply(x,q,n1,n2,n3); 
      saxpy(-1.0f,q,r,n1,n2,n3); // r = b-Ax
      float bnorm = sqrt(sdot(b,b,n1,n2,n3));
      float rnorm = sqrt(sdot(r,r,n1,n2,n3));
      float rnormBegin = rnorm;
      float rnormSmall = bnorm*_small;
      m->apply(r,s,n1,n2,n3); // s = Mr
      copy(s,d,n1,n2,n3); // d = s
      float delta = sdot(r,s,n1,n2,n3); // r's = r'Mr
      int iter;
      //log.fine("msolve: bnorm="+bnorm+" rnorm="+rnorm);
      for (iter=0; iter<_niter && rnorm>rnormSmall; ++iter) {
        //log.finer(" iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
        a->apply(d,q,n1,n2,n3); // q = Ad
        float alpha = delta/sdot(d,q,n1,n2,n3); // alpha = r'Mr/d'Ad
        saxpy( alpha,d,x,n1,n2,n3); // x = x+alpha*d
        if (iter%100<99) {
          saxpy(-alpha,q,r,n1,n2,n3); // r = r-alpha*q
        } else {
          copy(b,r,n1,n2,n3); a->apply(x,q,n1,n2,n3); 
          saxpy(-1.0f,q,r,n1,n2,n3); // r = b-Ax
        }
        m->apply(r,s,n1,n2,n3); // s = Mr
        float deltaOld = delta;
        delta = sdot(r,s,n1,n2,n3); // delta = r's = r'Mr
        float beta = delta/deltaOld;
        sxpay(beta,s,d,n1,n2,n3); // d = s+beta*d
        rnorm = sqrt(sdot(r,r,n1,n2,n3));
      }
      //log.fine(" iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
    }

    // Returns the dot product x'y.
    static float sdot(float** x, float** y, int n1, int n2) {
      float d = 0.0f;
      for (int i2=0; i2<n2; ++i2) {
        float *x2 = x[i2], *y2 = y[i2];
        for (int i1=0; i1<n1; ++i1) d += x2[i1]*y2[i1];
      }
      return d;
    }
    static float sdot(float*** x, float*** y, int n1, int n2, int n3) {
      float d = 0.0f;
      for (int i3=0; i3<n3; ++i3) d += sdot(x[i3],y[i3],n1,n2);
      return d;
    }
 
    // Computes y = y + a*x.
    static void saxpy(float a, float** x, float** y, int n1, int n2) {
      for (int i2=0; i2<n2; ++i2) {
        float *x2 = x[i2], *y2 = y[i2];
        for (int i1=0; i1<n1; ++i1) y2[i1] += a*x2[i1];
      }
    }
    static void saxpy(float a, float*** x, float*** y, int n1, int n2, int n3) {
      for (int i3=0; i3<n3; ++i3)
        saxpy(a,x[i3],y[i3],n1,n2);
    }

    // Computes y = x + a*y.
    static void sxpay(float a, float** x, float** y, int n1, int n2) {
      for (int i2=0; i2<n2; ++i2) {
        float *x2 = x[i2], *y2 = y[i2];
        for (int i1=0; i1<n1; ++i1) y2[i1] = a*y2[i1]+x2[i1];
      }
    }
    static void sxpay(float a, float*** x, float*** y, 
      int n1, int n2, int n3) {
      for (int i3=0; i3<n3; ++i3) sxpay(a,x[i3],y[i3],n1,n2);
    }

    // Computes z = x*y.
    static void sxy(float** x, float** y, float** z, int n1, int n2) {
      for (int i2=0; i2<n2; ++i2) {
        float *x2 = x[i2], *y2 = y[i2], *z2 = z[i2];
        for (int i1=0; i1<n1; ++i1) z2[i1] = x2[i1]*y2[i1];
      }
    }
    static void sxy(float*** x, float*** y, float*** z, 
      int n1, int n2, int n3) {
      for (int i3=0; i3<n3; ++i3) sxy(x[i3],y[i3],z[i3],n1,n2);
    }
  };
}

#endif
