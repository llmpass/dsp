#ifndef _KAISERWINDOW_H
#define _KAISERWINDOW_H

#include <cfloat>
#include <math.h>

/**
 * The Kaiser window is often used in FIR filter design. It is easy to
 * use and exhibits near-optimal properties for many filter design
 * problems. The window is defined by any two of three parameters:
 * window length, transition width, and maximum absolute error.
 * <p>
 * For definiteness, assume that the Kaiser window is a function w(x)
 * of argument x. Then, the window length is that range of x, centered
 * about x = 0, for which the Kaiser window is non-zero. In other words,
 * w(x) = 0 for |x| &gt; length/2. When windowing functions of time, both
 * the window length and the argument x of w(x) have dimensions of time.
 * <p>
 * The transition width is the width of the central lobe in the Fourier
 * transform of the window. For band-pass filters, this is the width of
 * the transition between pass and stop bands. When windowing functions
 * of time, the dimensions of transition width are cycles per unit time
 * (frequency). In any case, the product of window length and transition
 * width is dimensionless.
 * <p>
 * The maximum absolute error corresponds to the magnitude of ripples in
 * the passbands and stopbands of windowed filters. For an ideal band-pass
 * filter that has magnitude one in the pass band, the maximum (or minimum)
 * amplitude in the passband of a windowed filter is one plus (or minus)
 * the maximum amplitude error. Likewise, the maximum amplitude in the
 * stopband of such a windowed filter equals the maximum amplitude error.
 * <p>
 * Kaiser windows are based on approximate relationships among the three
 * design parameters. These approximations break down for passbands and
 * stopbands that are narrow relative to the transition width. In such
 * cases, the actual maximum error may exceed a specified maximum error
 * for which a Kaiser window is designed by up to a factor of two.
 * <p>
 * When constructing a Kaiser window for a specified window length and
 * transition width, the product of these two parameters cannot be less
 * than one. When length*width is less than one, a useful upper bound for
 * the maximum absolute error cannot be obtained from the Kaiser window
 * design equations. However, in this case, the lower bound for maximum
 * absolute error is nearly 10%, which is too large for most applications
 * anyway. Therefore, in practice, this restriction seldom matters.
 * @author Luming Liang, translated form Java Mines Toolkit by Dave Hale 
 * @version 2012.07.16
 */

namespace dsp {
  class KaiserWindow {

    /**
     * Returns a Kaiser window with specified error and transition width.
     * @param error the maximum absolute error.
     * @param width the transition width.
     * @return the window.
     */
    public: 
    static KaiserWindow* fromErrorAndWidth(double error, double width) {
      double a = -20.0*log10(error);
      double d = (a>21.0)?(a-7.95)/14.36:0.9222;
      double length = d/width;
      return new KaiserWindow(error,width,length);
    }

    /**
     * Returns a Kaiser window with specified error and window length.
     * @param error the maximum absolute error.
     * @param length the two-sided window length.
     * @return the window.
     */
    static KaiserWindow* fromErrorAndLength(double error, double length) {
      double a = -20.0*log10(error);
      double d = (a>21.0)?(a-7.95)/14.36:0.9222;
      double width = d/length;
      return new KaiserWindow(error,width,length);
    }

    /**
     * Returns a Kaiser window with specified transition width and window 
     * length.
     * The product width*length cannot be less than one.
     * @param width the transition width
     * @param length the two-sided window length.
     * @return the window.
     */
    static KaiserWindow* fromWidthAndLength(double width, double length) {
      double d = width*length;
      double a = 14.36*d+7.95;
      double error = pow(10.0,-a/20.0);
      return new KaiserWindow(error,width,length);
    }

    /**
     * Returns the value of this Kaiser window function w(x) for specified x.
     * @param x the argument for which to evaluate w(x).
     * @return the value w(x).
     */
    double evaluate(double x) {
      double xx = x*x;
      return (xx<=_xxmax)?_scale*ino(_alpha*sqrt(1.0-xx/_xxmax)):0.0;
    }

    /**
     * Gets the maximum absolute error.
     * @return the maximum absolute error.
     */
    double getError() {
      return _error;
    }

    /**
     * Gets the two-sided window length.
     * @return the window length.
     */
    double getLength() {
      return _length;
    }

    /**
     * Gets the transition width.
     * @return the transition width.
     */
    double getWidth() {
      return _width;
    }

    ///////////////////////////////////////////////////////////////////////////
    // private

    private: 
    double _error, _width, _length, _alpha, _scale, _xxmax;

    KaiserWindow(double error, double width, double length) {
      _error = error;
      _width = width;
      _length = length;
      double a = -20.0*log10(_error);
      if (a<=21.0) _alpha = 0.0;
      else { 
        if (a<=50.0) _alpha = 0.5842*pow(a-21.0,0.4)+0.07886*(a-21.0);
        else _alpha = 0.1102*(a-8.7);
      }
      _scale = 1.0/ino(_alpha);
      _xxmax = 0.25*_length*_length;
    }

    double ino(double x) {
      double s = 1.0, ds = 1.0, d = 0.0;
      do {
        d += 2.0;
        ds *= (x*x)/(d*d);
        s += ds;
      } while (ds>s*DBL_EPSILON);
      return s;
    }
  };
}

#endif
