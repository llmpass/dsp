#ifndef _FFTFILTER_H
#define _FFTFILTER_H

/**
* A linear shift-invariant filter implemented by fast Fourier transform.
* This filtering is equivalent to computing the convolution y = h*x,
* where h, x and y are filter, input and output arrays, respectively.
* <p>
* The filter is specified as a 1D, 2D or 3D array of coefficients.
* Filter dimension must match that of arrays to be filtered. For
* example, a filter constructed with a 2D array of coefficients
* cannot be applied to a 1D array.
* <p>
* The linear shift-invariant filtering performed by this class is a
* convolution sum. Each output sample in y is a sum of scaled input
* samples in x. For 1D filters this sum is
* <pre><code>
* nh-1-kh
* y[i] = sum h[kh+j]*x[i-j] ; i = 0, 1, 2, ..., ny-1 = nx-1
* j=-kh
* </code></pre>
* For each output sample y[i], kh is the array index of the filter
* coefficient h[kh] that scales the corresponding input sample x[i].
* For example, in a symmetric filter with odd length nh, the index
* kh = (nh-1)/2 is that of the middle coefficient in the array h.
* In other words, kh is the array index of the filter's origin.
* <p>
* The lengths nx and ny of the input and output arrays x and y are
* assumed to be equal. By default, values beyond the ends of an input
* array x in the convolution sum above are assumed to be zero. That
* is, zero values are used for x[i-j] when i-j &lt; 0 or when
* i-j &gt;= nx. Other methods for defining values beyond the ends of
* the array x may be specified. With any of these methods, the input
* array x is padded with extra values so that x[i-j] is defined for
* any i-j in the range [kh-nh+1:kh+nx-1] required by the convolution sum.
* <p>
* For efficiency, this filter can cache the fast Fourier transform of
* its coefficients h when the filter is first applied to any input
* array x. The filter may then be applied again, without recomputing
* its FFT, to other input arrays x that have the same lengths. The FFT
* of a cached filter is recomputed only when the lengths of the input
* and output arrays have changed. Because this caching consumes memory,
* it is disabled by default.
*
* @author Luming Liang, translated from Java Mines Toolkit by Dave Hale.
* @version 2012.07.16
*/

#include <iostream>

#include "FftReal.h"
#include "FftComplex.h"
#include "../util/ArrayMath.h"

using namespace std;
using namespace util;
 
namespace dsp {
  class FftFilter {
    public: enum Extrapolation {
      ZERO_VALUE,
      ZERO_SLOPE
    };
    /**
     * Constructs an FFT filter for specified filter coefficients.
     * The filter's origin is the center of the array.
     * @param h array of filter coefficients; copied, not referenced.
     */
    FftFilter(float* h, int n) {
      new (this) FftFilter((n-1)/2,h,n);
      //this->FftFilter((n-1)/2,h,n);
    }
    /**
     * Constructs an FFT filter for specified filter coefficients.
     * The coefficient h[kh] corresponds to the filter's origin.
     * @param kh array index of the filter's origin.
     * @param h array of filter coefficients; copied, not referenced.
     */
    FftFilter(int kh, float* h, int n) {
      _nh1 = n;
      _kh1 = kh;
      _h1 = copy(h,n);
      setExtrapolation(ZERO_VALUE);
    }
    FftFilter(float** h, int n1, int n2) {
      new (this) FftFilter((n1-1)/2,(n2-1)/2,h,n1,n2);
    }
    FftFilter(int kh1, int kh2, float** h, int n1, int n2) {
      _nh1 = n1;
      _nh2 = n2;
      _kh1 = kh1;
      _kh2 = kh2;
      _h2 = copy(h,n1,n2);
      setExtrapolation(ZERO_VALUE);
    }
    FftFilter(float*** h, int n1, int n2, int n3) {
      new (this) FftFilter((n1-1)/2,(n2-1)/2,(n3-1)/2,h,n1,n2,n3);
    }
    FftFilter(int kh1, int kh2, int kh3, float*** h, int n1, int n2, int n3) {
    //Check.argument(0<=kh1 && kh1<h[0][0].length,"index kh1 is valid");
    //Check.argument(0<=kh2 && kh2<h[0].length,"index kh2 is valid");
    //Check.argument(0<=kh3 && kh3<h.length,"index kh3 is valid");
      _nh1 = n1;
      _nh2 = n2;
      _nh3 = n3;
      _kh1 = kh1;
      _kh2 = kh2;
      _kh3 = kh3;
      _h3 = copy(h,n1,n2,n3);
      setExtrapolation(ZERO_VALUE);
    }
    void setExtrapolation(Extrapolation extrapolation) {
      _extrapolation = extrapolation;
    }
    /**
     * Enables or disables caching of the Fourier transform of the filter.
     * Caching consumes memory but improves performance by about 50% when
     * the same filter is applied repeatedly to arrays that have the same
     * dimensions.
     * @param filterCaching true, to enable caching; false, to disable.
     */
    void setFilterCaching(bool filterCaching) {
      _filterCaching = filterCaching;
    }
    int getnh1() {
      return _nh1;
    }
    /**
     * Applies this filter.
     * @param x input array.
     * @return filtered array.
     */
    float* apply(float* x, int n) {
      float* y = new float[n];
      apply(x,y,n);
      return y;
    }
    void apply(float* x, float* y, int n) {
    //Check.state(_h1!=null,"1D filter is available");
      int nx1 = n;
      updateFfts(nx1);
      float* xfft = new float[_nfft1+2];
      copy(nx1,x,xfft);
      extrapolate(xfft);
      _fft1->realToComplex(-1,xfft,xfft);
      int nk1 = _nfft1/2+1;
      for (int ik1=0,k1r=0,k1i=1; ik1<nk1; ++ik1,k1r+=2,k1i+=2) {
        float xr = xfft[k1r];
        float xi = xfft[k1i];
        float hr = _h1fft[k1r];
        float hi = _h1fft[k1i];
        xfft[k1r] = xr*hr-xi*hi;
        xfft[k1i] = xr*hi+xi*hr;
      }
      if (!_filterCaching) _h1fft = NULL;
      _fft1->complexToReal(1,xfft,xfft);
      copy(nx1,xfft,y);
      delete [] xfft;
    }
    
    float** apply(float** x, int n1, int n2) {
      float** y = new float*[n2];
      for (int i2=0; i2<n2; ++i2) y[i2] = new float[n1];
      apply(x,y,n1,n2);
      return y;
    }
    void apply(float** x, float** y, int n1, int n2) {
    //Check.state(_h2!=null,"2D filter is valid");
      int nx1 = n1;
      int nx2 = n2;
      updateFfts(nx1,nx2);
      float** xfft = new float*[_nfft2];
      for (int i2=0; i2<_nfft2; ++i2) xfft[i2] = new float[_nfft1+2];
      copy(nx1,nx2,x,xfft);
      extrapolate(xfft);
      _fft1->realToComplex1(-1,_nfft2,xfft,xfft);
      _fft2->complexToComplex2(-1,_nfft1/2+1,xfft,xfft);
      int nk1 = _nfft1/2+1;
      int nk2 = _nfft2;
      float *x2, *h2;
      for (int ik2=0; ik2<nk2; ++ik2) {
        x2 = xfft[ik2];
        h2 = _h2fft[ik2];
        for (int ik1=0,k1r=0,k1i=1; ik1<nk1; ++ik1,k1r+=2,k1i+=2) {
          float xr = x2[k1r];
          float xi = x2[k1i];
          float hr = h2[k1r];
          float hi = h2[k1i];
          x2[k1r] = xr*hr-xi*hi;
          x2[k1i] = xr*hi+xi*hr;
        }
      }  
      if (!_filterCaching) _h2fft = NULL;
      _fft2->complexToComplex2(1,_nfft1/2+1,xfft,xfft);
      _fft1->complexToReal1(1,_nfft2,xfft,xfft);
      copy(nx1,nx2,xfft,y);
      delete [] xfft;
    }

    float*** apply(float*** x, int n1, int n2, int n3) {
      float*** y = new float**[n3];
      for (int i3=0; i3<n3; ++i3) {
        y[i3] = new float*[n2];
        for (int i2=0; i2<n2; ++i2) y[i3][i2] = new float[n1];
      }
      apply(x,y,n1,n2,n3);
      return y;
    }
    void apply(float*** x, float*** y, int n1, int n2, int n3) {
    //Check.state(_h3!=null,"3D filter is valid");
      int nx1 = n1;
      int nx2 = n2;
      int nx3 = n3;
      updateFfts(nx1,nx2,nx3);
      float*** xfft = new float**[_nfft3];
      for (int i3=0; i3<n3; ++i3) {
        xfft[i3] = new float*[_nfft2];
        for (int i2=0; i2<n2; ++i2) xfft[i3][i2] = new float[_nfft1+2];
      }
      copy(nx1,nx2,nx3,x,xfft);
      extrapolate(xfft);
      _fft1->realToComplex1(-1,_nfft2,_nfft3,xfft,xfft);
      _fft2->complexToComplex2(-1,_nfft1/2+1,_nfft3,xfft,xfft);
      _fft3->complexToComplex3(-1,_nfft1/2+1,_nfft2,xfft,xfft);
      int nk1 = _nfft1/2+1;
      int nk2 = _nfft2;
      int nk3 = _nfft3;
      float *x32, *h32;
      for (int ik3=0; ik3<nk3; ++ik3) {
        for (int ik2=0; ik2<nk2; ++ik2) {
          x32 = xfft[ik3][ik2];
          h32 = _h3fft[ik3][ik2];
          for (int ik1=0,k1r=0,k1i=1; ik1<nk1; ++ik1,k1r+=2,k1i+=2) {
            float xr = x32[k1r];
            float xi = x32[k1i];
            float hr = h32[k1r];
            float hi = h32[k1i];
            x32[k1r] = xr*hr-xi*hi;
            x32[k1i] = xr*hi+xi*hr;
          }
        }
      }
      if (!_filterCaching) _h3fft = NULL;
      _fft3->complexToComplex3(1,_nfft1/2+1,_nfft2,xfft,xfft);
      _fft2->complexToComplex2(1,_nfft1/2+1,_nfft3,xfft,xfft);
      _fft1->complexToReal1(1,_nfft2,_nfft3,xfft,xfft);
      copy(nx1,nx2,nx3,xfft,y);
      delete [] xfft; 
    }

    ///////////////////////////////////////////////////////////////////////////
    // private

    private: 
    int _nx1,_nx2,_nx3;
    int _nh1,_nh2,_nh3;
    int _kh1,_kh2,_kh3;
    int _nfft1,_nfft2,_nfft3;
    FftReal* _fft1;
    FftComplex *_fft2,*_fft3;
    float *_h1,*_h1fft;
    float ** _h2,**_h2fft;
    float ***_h3,***_h3fft;
    Extrapolation _extrapolation;
    bool _filterCaching;

    void updateFfts(int nx1) {
      if (_fft1==NULL || _h1fft==NULL || _nx1!=nx1) {
        _nx1 = nx1;
        _nx2 = 0;
        _nx3 = 0;
        _nfft1 = FftReal::nfftFast(_nx1+_nh1);
        _fft1 = new FftReal(_nfft1);
        _fft2 = NULL;
        _fft3 = NULL;
        _h1fft = new float[_nfft1+2];
        _h2fft = NULL;
        _h3fft = NULL;
        float scale = 1.0f/(float)_nfft1;
        for (int ih1=0; ih1<_nh1; ++ih1) {
          int jh1 = ih1-_kh1;
          if (jh1<0) jh1 += _nfft1;
          _h1fft[jh1] = scale*_h1[ih1];
        }
        _fft1->realToComplex(-1,_h1fft,_h1fft);
      }
    }

    void updateFfts(int nx1, int nx2) {
      if (_fft2==NULL || _h2fft==NULL || _nx1!=nx1 || _nx2!=nx2) {
        _nx1 = nx1;
        _nx2 = nx2;
        _nx3 = 0;
        _nfft1 = FftReal::nfftFast(_nx1+_nh1);
        _nfft2 = FftComplex::nfftFast(_nx2+_nh2);
        _fft1 = new FftReal(_nfft1);
        _fft2 = new FftComplex(_nfft2);
        _fft3 = NULL;
        _h1fft = NULL;
        _h2fft = new float*[_nfft2];
        for (int i2=0; i2<_nfft2; ++i2) _h2fft[i2] = new float[_nfft1+2];
        _h3fft = NULL;
        float scale = 1.0f/(float)_nfft1/(float)_nfft2;
        for (int ih2=0; ih2<_nh2; ++ih2) {
          int jh2 = ih2-_kh2;
          if (jh2<0) jh2 += _nfft2;
          for (int ih1=0; ih1<_nh1; ++ih1) {
            int jh1 = ih1-_kh1;
            if (jh1<0) jh1 += _nfft1;
            _h2fft[jh2][jh1] = scale*_h2[ih2][ih1];
          }
        }
        _fft1->realToComplex1(-1,_nfft2,_h2fft,_h2fft);
        _fft2->complexToComplex2(-1,_nfft1/2+1,_h2fft,_h2fft);
      }
    }

    void updateFfts(int nx1, int nx2, int nx3) {
      if (_fft3==NULL || _h3fft==NULL || _nx1!=nx1 || _nx2!=nx2 || _nx3!=nx3) {
        _nx1 = nx1;
        _nx2 = nx2;
        _nx3 = nx3;
        _nfft1 = FftReal::nfftFast(_nx1+_nh1);
        _nfft2 = FftComplex::nfftFast(_nx2+_nh2);
        _nfft3 = FftComplex::nfftFast(_nx3+_nh3);
        _fft1 = new FftReal(_nfft1);
        _fft2 = new FftComplex(_nfft2);
        _fft3 = new FftComplex(_nfft3);
        _h1fft = NULL;
        _h2fft = NULL;
        _h3fft = new float**[_nfft3];
        for (int i3=0; i3<_nfft3; ++i3) {
          _h3fft[i3] = new float*[_nfft2];
          for (int i2=0; i2<_nfft2; ++i2) _h3fft[i3][i2] = new float[_nfft1+2];
        }
        float scale = 1.0f/(float)_nfft1/(float)_nfft2/(float)_nfft3;
        for (int ih3=0; ih3<_nh3; ++ih3) {
          int jh3 = ih3-_kh3;
          if (jh3<0) jh3 += _nfft3;
          for (int ih2=0; ih2<_nh2; ++ih2) {
            int jh2 = ih2-_kh2;
            if (jh2<0) jh2 += _nfft2;
            for (int ih1=0; ih1<_nh1; ++ih1) {
              int jh1 = ih1-_kh1;
              if (jh1<0) jh1 += _nfft1;
              _h3fft[jh3][jh2][jh1] = scale*_h3[ih3][ih2][ih1];
            }
          }
        }
        _fft1->realToComplex1(-1,_nfft2,_nfft3,_h3fft,_h3fft);
        _fft2->complexToComplex2(-1,_nfft1/2+1,_nfft3,_h3fft,_h3fft);
        _fft3->complexToComplex3(-1,_nfft1/2+1,_nfft2,_h3fft,_h3fft);
      }
    }

    void extrapolate(float* xfft) {
      if (_extrapolation==ZERO_SLOPE) {
        int mr1 = _nx1+_kh1;
        float xr1 = xfft[_nx1-1];
        for (int i1=_nx1; i1<mr1; ++i1)
          xfft[i1] = xr1;
        int ml1 = _nfft1+_kh1-_nh1+1;
        float xl1 = xfft[0];
        for (int i1=ml1; i1<_nfft1; ++i1)
          xfft[i1] = xl1;
      }
    }

    void extrapolate(float** xfft) {
      if (_extrapolation==ZERO_SLOPE) {
        for (int i2=0; i2<_nx2; ++i2) extrapolate(xfft[i2]);
        int mr2 = _nx2+_kh2;
        float* xr2 = xfft[_nx2-1];
        for (int i2=_nx2; i2<mr2; ++i2) copy(xr2,xfft[i2],_nx1);
        int ml2 = _nfft2+_kh2-_nh2+1;
        float* xl2 = xfft[0];
        for (int i2=ml2; i2<_nfft2; ++i2) copy(xl2,xfft[i2],_nx1);
      }
    }

    void extrapolate(float*** xfft) {
      if (_extrapolation==ZERO_SLOPE) {
        for (int i3=0; i3<_nx3; ++i3) extrapolate(xfft[i3]);
        int mr3 = _nx3+_kh3;
        float** xr3 = xfft[_nx3-1];
        for (int i3=_nx3; i3<mr3; ++i3) copy(xr3,xfft[i3],_nx2,_nx1);
        int ml3 = _nfft3+_kh3-_nh3+1;
        float** xl3 = xfft[0];
        for (int i3=ml3; i3<_nfft3; ++i3) copy(xl3,xfft[i3],_nx2,_nx1);
      }
    }
  };
}

#endif
