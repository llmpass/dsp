#ifndef _BANDPASSFILTER_H
#define _BANDPASSFILTER_H

#include <iostream>

#include "FftFilter.h"
#include "KaiserWindow.h"
#include "../util/ArrayMath.h"

using namespace std;
using namespace util;
/**
* A multi-dimensional band-pass filter. Filtering is performed using fast
* Fourier transforms. The result is equivalent to convolution with an
* ideal symmetric (zero-phase) band-pass filter that has been smoothly
* tapered to zero.
* <p>
* Filter parameters include lower and upper frequencies that define the
* pass band, the width of the transition from pass band to stop bands,
* and the maximum error for amplitude in both pass and stop bands.
* <p>
* For efficiency the Fourier transform of the filter may be cached for
* repeated application to multiple input arrays. A cached transform
* can be reused while the lengths of input and output arrays do not
* change. Because caching consumes memory, it is disabled by default.
*
* @author Luming Liang, translated from Mines Java Toolkit by Dave Hale
* @version 2009.12.19
*/

namespace dsp {
  static double PIO4 = PI/4.0;
  static double PIO6 = PI/6.0;
  class BandPassFilter {

    /**
    * The method used to extrapolate values beyond the ends of input arrays.
    * The default is extrapolation with zero values.
    */
    public: enum Extrapolation {
      ZERO_VALUE,
      ZERO_SLOPE
    };

    /**
     * Constructs a band-pass filter with specified parameters.
     * @param klower the lower pass band frequency, in cycles per sample.
     * @param kupper the upper pass band frequency, in cycles per sample.
     * @param kwidth width of the transition between pass and stop bands.
     * @param aerror approximate bound on amplitude error, a positive fraction.
    */
    BandPassFilter(double klower, double kupper, double kwidth, double aerror) {
      _klower = klower;
      _kupper = kupper;
      _kwidth = kwidth;
      _aerror = aerror;
      _extrapolation = ZERO_VALUE;
      _ff1 = _ff2 = _ff3 = NULL;
    }

    void setExtrapolation(Extrapolation extrapolation) {
      if (_extrapolation!=extrapolation) {
        _extrapolation = extrapolation;
        _ff1 = _ff2 = _ff3 = NULL;
      }
    }

    /**
     * Enables or disables caching of the Fourier transform of the filter.
     * Caching consumes memory but improves performance by about 50% when
     * the same filter is applied repeatedly to arrays that have the same
     * dimensions.
     * @param filterCaching true, to enable caching; false, to disable.
     */
    void setFilterCaching(bool filterCaching) {
      if (_filterCaching!=filterCaching) {
        _filterCaching = filterCaching;
        _ff1 = _ff2 = _ff3 = NULL;
      }
    }

    /**
     * Gets the 1D array of coefficients for this filter.
     * The origin of the filter is at the center of the array.
     * @return the array of filter coefficients.
     */
    float* getCoefficients1() {
      updateFilter1();
      return copy(_h1,_nh1);
    }

    /**
     * Gets the 2D array of coefficients for this filter.
     * The origin of the filter is at the center of the array.
     * @return the array of filter coefficients.
     */
    float** getCoefficients2() {
      updateFilter2();
      return copy(_h2,_nh1,_nh2);
    }

    /**
     * Gets the 3D array of coefficients for this filter.
     * The origin of the filter is at the center of the array.
     * @return the array of filter coefficients.
     */
    float*** getCoefficients3() {
      updateFilter3();
      return copy(_h3,_nh1,_nh2,_nh3);
    }

    /**
     * Applies this filter.
     * Input and output arrays may be the same array.
     * @param x input array.
     * @param y output filtered array.
     */
    void apply(float* x, float* y, int n) {
      updateFilter1();
      _ff1->apply(x,y,n);
      /*for (int i=0; i<n; ++i) 
        cout<<y[i]<<"  ";
      cout<<endl;
      cout<<"fft filter finished"<<endl;*/
    }

    void apply(float** x, float** y, int n1, int n2) {
      updateFilter2();
      _ff2->apply(x,y,n1,n2);
    }

    /**
     * Applies this filter.
     * Input and output arrays may be the same array.
     * @param x input array.
     * @param y output filtered array.
     */
    void apply(float*** x, float*** y, int n1, int n2, int n3) {
      updateFilter3();
      _ff3->apply(x,y,n1,n2,n3);
    }

    ///////////////////////////////////////////////////////////////////////////
    // private

    private: double _klower, _kupper, _kwidth, _aerror;
    FftFilter *_ff1, *_ff2, *_ff3;
    float* _h1;
    float** _h2;
    float*** _h3;
    int _nh1, _nh2, _nh3;
    Extrapolation _extrapolation;
    bool _filterCaching;

    FftFilter::Extrapolation ffExtrap(Extrapolation e) {
      if (e==ZERO_VALUE) return FftFilter::ZERO_VALUE;
      else return FftFilter::ZERO_SLOPE;
    }

    void updateFilter1() {
      if (_ff1==NULL) {
        KaiserWindow* kw = KaiserWindow::fromErrorAndWidth(_aerror,_kwidth);
        int nh = ((int)kw->getLength()+1)/2*2+1;
        int nh1 = nh;
        _nh1 = nh1;
        int kh1 = (nh1-1)/2;
        _h1 = new float[nh1];
        double kus = 2.0*_kupper;
        double kls = 2.0*_klower;
        for (int i1=0; i1<nh1; ++i1) {
          double x1 = i1-kh1;
          double w1 = kw->evaluate(x1);
          double r = x1;
          double kur = 2.0*_kupper*r;
          double klr = 2.0*_klower*r;
          _h1[i1] = (float)(w1*(kus*h1(kur)-kls*h1(klr)));
        }
        _ff1 = new FftFilter(_h1,nh1);
        _ff1->setExtrapolation(ffExtrap(_extrapolation));
        _ff1->setFilterCaching(_filterCaching);
        delete kw;
      }
    }

    void updateFilter2() {
      if (_ff2==NULL) {
        KaiserWindow* kw = KaiserWindow::fromErrorAndWidth(_aerror,_kwidth);
        int nh = ((int)kw->getLength()+1)/2*2+1;
        int nh1 = nh;
        int nh2 = nh;
        _nh1 = nh1;
        _nh2 = nh2;
        int kh1 = (nh1-1)/2;
        int kh2 = (nh2-1)/2;
        _h2 = new float*[nh2];
        for (int i2=0; i2<nh2; ++i2) 
          _h2[i2] = new float[nh1];
        double kus = 4.0*_kupper*_kupper;
        double kls = 4.0*_klower*_klower;
        for (int i2=0; i2<nh2; ++i2) {
          double x2 = i2-kh2;
          double w2 = kw->evaluate(x2);
          for (int i1=0; i1<nh1; ++i1) {
            double x1 = i1-kh1;
            double w1 = kw->evaluate(x1);
            double r = sqrt(x1*x1+x2*x2);
            double kur = 2.0*_kupper*r;
            double klr = 2.0*_klower*r;
            _h2[i2][i1] = (float)(w1*w2*(kus*h2(kur)-kls*h2(klr)));
          }
        }
        _ff2 = new FftFilter(_h2,nh1,nh2);
        _ff2->setExtrapolation(ffExtrap(_extrapolation));
        _ff2->setFilterCaching(_filterCaching);
        delete kw;
      }
    }

    void updateFilter3() {
      if (_ff3==NULL) {
        KaiserWindow* kw = KaiserWindow::fromErrorAndWidth(_aerror,_kwidth);
        int nh = ((int)kw->getLength()+1)/2*2+1;
        int nh1 = nh;
        int nh2 = nh;
        int nh3 = nh;
        _nh1 = nh1;
        _nh2 = nh2;
        _nh3 = nh3;
        int kh1 = (nh1-1)/2;
        int kh2 = (nh2-1)/2;
        int kh3 = (nh3-1)/2;
        _h3 = new float**[nh3];
        for (int i3=0; i3<nh3; ++i3) {
          _h3[i3] = new float*[nh2];
          for (int i2=0; i2<nh2; ++i2) _h3[i3][i2] = new float[nh1];
        }
        double kus = 8.0*_kupper*_kupper*_kupper;
        double kls = 8.0*_klower*_klower*_klower;
        for (int i3=0; i3<nh3; ++i3) {
          double x3 = i3-kh3;
          double w3 = kw->evaluate(x3);
          for (int i2=0; i2<nh2; ++i2) {
            double x2 = i2-kh2;
            double w2 = kw->evaluate(x2);
            for (int i1=0; i1<nh1; ++i1) {
              double x1 = i1-kh1;
              double w1 = kw->evaluate(x1);
              double r = sqrt(x1*x1+x2*x2+x3*x3);
              double kur = 2.0*_kupper*r;
              double klr = 2.0*_klower*r;
              _h3[i3][i2][i1] = (float)(w1*w2*w3*(kus*h3(kur)-kls*h3(klr)));
            }
          }
        }
        _ff3 = new FftFilter(_h3,nh1,nh2,nh3);
        _ff3->setExtrapolation(ffExtrap(_extrapolation));
        _ff3->setFilterCaching(_filterCaching);
        delete [] kw;
      }
    }

    static double h1(double r) {
      return (r==0.0)?1.0:sin(PI*r)/(PI*r);
    }

    static double h2(double r) {
      return (r==0.0)?PIO4:besselJ1(PI*r)/(2.0*r);
    }
    static double besselJ1(double x) {
      double ax = fabs(x);
      if (ax<8.0) {
        double xx = x*x;
        double num = x*(72362614232.0 +
          xx*(-7895059235.0 +
          xx*(242396853.1 +
          xx*(-2972611.439 +
          xx*(15704.48260+
          xx*(-30.16036606))))));
        double den = 144725228442.0 +
          xx*(2300535178.0 +
          xx*(18583304.74 +
          xx*(99447.43394 +
          xx*(376.9991397 +
          xx))));
        return num/den;
      } else {
        double z = 8.0/ax;
        double zz = z*z;
        double t1 = 1.0 +
          zz*(0.183105e-2 +
          zz*(-0.3516396496e-4 +
          zz*(0.2457520174e-5 +
          zz*(-0.240337019e-6))));
        double t2 = 0.04687499995 +
          zz*(-0.2002690873e-3 +
          zz*(0.8449199096e-5 +
          zz*(-0.88228987e-6 +
          zz*0.105787412e-6)));
        double am = ax-2.356194491;
        double y = sqrt(0.636619772/ax)*(cos(am)*t1-z*sin(am)*t2);
        return (x<0.0)?-y:y;
      }
    }

    static double h3(double r) {
      if (r==0.0) return PIO6;
      else {
        double pir = PI*r;
        return 0.5*PI*(sin(pir)-pir*cos(pir))/(pir*pir*pir);
      }
    }
  };
}

#endif
