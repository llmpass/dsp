#ifndef _RECURSIVEGAUSSIANFILTER_H
#define _RECURSIVEGAUSSIANFILTER_H

#include <math.h>
#include <iostream>
using namespace std;

namespace dsp {

/**
  * Recursive implementation of a Gaussian filter and derivatives.
  * Translate from Mines Java Toolkit: 
  * https://github.com/dhale/jtk/blob/master/src/main/java/edu/mines/jtk/dsp/RecursiveGaussianFilter.java
  * by Luming Liang, 2012.05.30
  */

class RecursiveGaussianFilter {
  /////////////////////////////////////////////////////////////////////////////////
  // private methods

  private:
  class Filter {
    public:
    virtual void applyN(int nd, float* x, float* y, int n){}
    virtual void applyXN(int nd, float** x, float** y, int n1, int n2){}
    void applyNX(int nd, float** x, float** y, int n1, int n2) {
      for (int i2=0; i2<n2; ++i2) applyN(nd,x[i2],y[i2],n1);
    }
    void applyNXX(int nd, float*** x, float*** y, int n1, int n2, int n3) {
      for (int i3=0; i3<n3; ++i3) applyNX(nd,x[i3],y[i3],n1,n2);
    }
    void applyXNX(int nd, float*** x, float*** y, int n1, int n2, int n3) {
      for (int i3=0; i3<n3; ++i3) applyXN(nd,x[i3],y[i3],n1,n2);
    }
    void applyXXN(int nd, float*** x, float*** y, int n1, int n2, int n3) {
      // temporary n2*n3 arrays
      float** x2 = new float*[n2];
      float** y2 = new float*[n2];
      for (int i2=0; i2<n2; ++i2) 
        for (int i3=0; i3<n3; ++i3) {
          x2[i3] = x[i3][i2];
          y2[i3] = y[i3][i2];
        }
      applyXN(nd,x2,y2,n3,n2);
      x2=0; y2=0; delete x2; delete y2;
    }
  };

  /////////////////////////////////////////////////////////////////////////////////
  // public methods
  public:
  enum Method { 
    DERICHE,
    VAN_VLIET
  };

  Filter* _filter;
  /**
   * Construct a Gaussian filter with specified width and design method.
   * @param sigma the width; must not be less than 1.
   * @param method the method used to design the filter.
   */
  RecursiveGaussianFilter(double sigma, Method m) {
    _filter = (m==DERICHE) ? 
      new DericheFilter(sigma) :
      new DericheFilter(sigma);
    //VanVlietFilter(sigma);
  }

  /**
   * Construct a Gaussian filter with specified width.
   * @param sigma the width; must not be less than 1.
   */
  RecursiveGaussianFilter(double sigma) {
    //Check.argument(sigma>=1.0,"sigma>=1.0");
    _filter = (sigma<32.0) ?
      new DericheFilter(sigma) :
      new DericheFilter(sigma);
    //VanVlietFilter(sigma);
  }

  /**
   * Applies the 0th-derivative filter.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply0(float* x, float* y, int n) {
    _filter->applyN(0,x,y,n);
  }

  /**
   * Applies the 1st-derivative filter.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply1(float* x, float* y, int n) {
    _filter->applyN(1,x,y,n);
  }

  /**
   * Applies the 2nd-derivative filter.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply2(float* x, float* y, int n) {
    _filter->applyN(2,x,y,n);
  }

  /**
   * Applies the 0th-derivative filter along the 1st dimension.
   * Applies no filter along the 2nd dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply0X(float** x, float** y, int n1, int n2) {
    _filter->applyNX(0,x,y,n1,n2);
  }

  /**
   * Applies the 1st-derivative filter along the 1st dimension.
   * Applies no filter along the 2nd dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply1X(float** x, float** y, int n1, int n2) {
    _filter->applyNX(1,x,y,n1,n2);
  }

  /**
   * Applies the 2nd-derivative filter along the 1st dimension.
   * Applies no filter along the 2nd dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply2X(float** x, float** y, int n1, int n2) {
    _filter->applyNX(2,x,y,n1,n2);
  }

  /**
   * Applies the 0th-derivative filter along the 2nd dimension.
   * Applies no filter along the 1st dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void applyX0(float** x, float** y, int n1, int n2) {
    _filter->applyXN(0,x,y,n1,n2);
  }

  /**
   * Applies the 1st-derivative filter along the 2nd dimension.
   * Applies no filter along the 1st dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void applyX1(float** x, float** y, int n1, int n2) {
    _filter->applyXN(1,x,y,n1,n2);
  }

  /**
   * Applies the 2nd-derivative filter along the 2nd dimension.
   * Applies no filter along the 1st dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void applyX2(float** x, float** y, int n1, int n2) {
    _filter->applyXN(2,x,y,n1,n2);
  }

  /**
   * Applies the 0th-derivative filter along the 1st dimension.
   * Applies no filter along the 2nd or 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply0XX(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyNXX(0,x,y,n1,n2,n3);
  }

  /**
   * Applies the 1st-derivative filter along the 1st dimension.
   * Applies no filter along the 2nd or 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply1XX(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyNXX(1,x,y,n1,n2,n3);
  }

  /**
   * Applies the 2nd-derivative filter along the 1st dimension.
   * Applies no filter along the 2nd or 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply2XX(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyNXX(2,x,y,n1,n2,n3);
  }

  /**
   * Applies the 0th-derivative filter along the 2nd dimension.
   * Applies no filter along the 1st or 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void applyX0X(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXNX(0,x,y,n1,n2,n3);
  }

  /**
   * Applies the 1st-derivative filter along the 2nd dimension.
   * Applies no filter along the 1st or 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void applyX1X(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXNX(1,x,y,n1,n2,n3);
  }

  /**
   * Applies the 2nd-derivative filter along the 2nd dimension.
   * Applies no filter along the 1st or 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void applyX2X(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXNX(2,x,y,n1,n2,n3);
  }

  /**
   * Applies the 0th-derivative filter along the 3rd dimension.
   * Applies no filter along the 1st or 2nd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void applyXX0(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(0,x,y,n1,n2,n3);
  }

  /**
   * Applies the 1st-derivative filter along the 3rd dimension.
   * Applies no filter along the 1st or 2nd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void applyXX1(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(1,x,y,n1,n2,n3);
  }

  /**
   * Applies the 2nd-derivative filter along the 3rd dimension.
   * Applies no filter along the 1st or 2nd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void applyXX2(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(2,x,y,n1,n2,n3);
  }

  /**
   * Applies the 0th-derivative filter along the 1st and 2nd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply00(float** x, float** y, int n1, int n2) {
    float** temp = zerofloat(n1,n2);
    _filter->applyXN(0,x,temp,n1,n2);
    _filter->applyNX(0,temp,y,n1,n2);
    for (int i2=0; i2<n2; ++i2) delete [] temp[i2];
    delete [] temp;
  }

  /**
   * Applies the 1st-derivative filter along the 1st dimension
   * and the 0th-derivative filter along the 2nd dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply10(float** x, float** y, int n1, int n2) {
    _filter->applyXN(0,x,y,n1,n2);
    _filter->applyNX(1,y,y,n1,n2);
  }

  /**
   * Applies the 0th-derivative filter along the 1st dimension
   * and the 1st-derivative filter along the 2nd dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply01(float** x, float** y, int n1, int n2) {
    _filter->applyXN(1,x,y,n1,n2);
    _filter->applyNX(0,y,y,n1,n2);
  }

  /**
   * Applies the 1st-derivative filter along the 1st and 2nd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply11(float** x, float** y, int n1, int n2) {
    float** temp = zerofloat(n1,n2);
    _filter->applyXN(1,x,temp,n1,n2);
    _filter->applyNX(1,temp,y,n1,n2);
    for (int i2=0; i2<n2; ++i2) delete [] temp[i2];
    delete [] temp;
  }

  /**
   * Applies the 2nd-derivative filter along the 1st dimension
   * and the 0th-derivative filter along the 2nd dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply20(float** x, float** y, int n1, int n2) {
    _filter->applyXN(0,x,y,n1,n2);
    _filter->applyNX(2,y,y,n1,n2);
  }

  /**
   * Applies the 0th-derivative filter along the 1st dimension
   * and the 2nd-derivative filter along the 2nd dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply02(float** x, float** y, int n1, int n2) {
    _filter->applyXN(2,x,y,n1,n2);
    _filter->applyNX(0,y,y,n1,n2);
  }

  /**
   * Applies the 0th-derivative filter along the 1st, 2nd and 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply000(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(0,x,y,n1,n2,n3);
    _filter->applyXNX(0,y,y,n1,n2,n3);
    _filter->applyNXX(0,y,y,n1,n2,n3);
  }

  /**
   * Applies the 1st-derivative filter along the 1st dimension
   * and the 0th-derivative filter along the 2nd and 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply100(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(0,x,y,n1,n2,n3);
    _filter->applyXNX(0,y,y,n1,n2,n3);
    _filter->applyNXX(1,y,y,n1,n2,n3);
  }

  /**
   * Applies the 1st-derivative filter along the 2nd dimension
   * and the 0th-derivative filter along the 1st and 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply010(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(0,x,y,n1,n2,n3);
    _filter->applyXNX(1,y,y,n1,n2,n3);
    _filter->applyNXX(0,y,y,n1,n2,n3);
  }

  /**
   * Applies the 1st-derivative filter along the 3rd dimension
   * and the 0th-derivative filter along the 1st and 2nd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply001(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(1,x,y,n1,n2,n3);
    _filter->applyXNX(0,y,y,n1,n2,n3);
    _filter->applyNXX(0,y,y,n1,n2,n3);
  }

  /**
   * Applies the 1st-derivative filter along the 1st and 2nd dimensions
   * and the 0th-derivative filter along the 3rd dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply110(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(0,x,y,n1,n2,n3);
    _filter->applyXNX(1,y,y,n1,n2,n3);
    _filter->applyNXX(1,y,y,n1,n2,n3);
  }

  /**
   * Applies the 1st-derivative filter along the 1st and 3rd dimensions
   * and the 0th-derivative filter along the 2nd dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply101(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(1,x,y,n1,n2,n3);
    _filter->applyXNX(0,y,y,n1,n2,n3);
    _filter->applyNXX(1,y,y,n1,n2,n3);
  }

  /**
   * Applies the 1st-derivative filter along the 2nd and 3rd dimensions
   * and the 0th-derivative filter along the 1st dimension.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply011(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(1,x,y,n1,n2,n3);
    _filter->applyXNX(1,y,y,n1,n2,n3);
    _filter->applyNXX(0,y,y,n1,n2,n3);
  }

  /**
   * Applies the 2nd-derivative filter along the 1st dimension
   * and the 0th-derivative filter along the 2nd and 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply200(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(0,x,y,n1,n2,n3);
    _filter->applyXNX(0,y,y,n1,n2,n3);
    _filter->applyNXX(2,y,y,n1,n2,n3);
  }

  /**
   * Applies the 2nd-derivative filter along the 2nd dimension
   * and the 0th-derivative filter along the 1st and 3rd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply020(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(0,x,y,n1,n2,n3);
    _filter->applyXNX(2,y,y,n1,n2,n3);
    _filter->applyNXX(0,y,y,n1,n2,n3);
  }

  /**
   * Applies the 2nd-derivative filter along the 3rd dimension
   * and the 0th-derivative filter along the 1st and 2nd dimensions.
   * @param x the filter input.
   * @param y the filter output.
   */
  void apply002(float*** x, float*** y, int n1, int n2, int n3) {
    _filter->applyXXN(2,x,y,n1,n2,n3);
    _filter->applyXNX(0,y,y,n1,n2,n3);
    _filter->applyNXX(0,y,y,n1,n2,n3);
  }
  

  /////////////////////////////////////////////////////////////////////////////////
  private:
  class DericheFilter: public Filter {
    public:
    DericheFilter(double sigma) {
      cout<<"DericheFilter constructor"<<endl;
      initialization();
      makeND(sigma);
    }
    ~DericheFilter() {
      _n0=0; _n1=0; _n2=0; _n3=0;
      _d1=0; _d2=0; _d3=0; _d4=0;
      delete _n0; delete _n1; delete _n2; delete _n3;
      delete _d1; delete _d2; delete _d3; delete _d4;
    }
    void initialization() {
      a0[0]=a00; a0[1]=a01; a0[2]=a02;
      a1[0]=a01; a1[1]=a11; a1[2]=a12;
      b0[0]=b00; b0[1]=b01; b0[2]=b02;
      b1[0]=b10; b1[1]=b11; b1[2]=b12;
      c0[0]=c00; c0[1]=c01; c0[2]=c02;
      c1[0]=c10; c1[1]=c11; c1[2]=c12;
      w0[0]=w00; w0[1]=w01; w0[2]=w02;
      w1[0]=w10; w1[1]=w11; w1[2]=w12;
    }
    void applyN(int nd, float* x, float* y, int n) {
      //checkArrays(x,y);
      //if (sameArrays(x,y)) x = copy(x);
      float n0 = _n0[nd], n1 = _n1[nd], n2 = _n2[nd], n3 = _n3[nd];
      float d1 = _d1[nd], d2 = _d2[nd], d3 = _d3[nd], d4 = _d4[nd];
      float yim4 = 0.0f, yim3 = 0.0f, yim2 = 0.0f, yim1 = 0.0f;
      float xim3 = 0.0f, xim2 = 0.0f, xim1 = 0.0f;
      for (int i=0; i<n; ++i) {
        float xi = x[i];
        float yi = n0*xi+n1*xim1+n2*xim2+n3*xim3 -
                         d1*yim1-d2*yim2-d3*yim3-d4*yim4;
        y[i] = yi;
        yim4 = yim3; yim3 = yim2; yim2 = yim1; yim1 = yi;
                     xim3 = xim2; xim2 = xim1; xim1 = xi;
      }
      n1 = n1-d1*n0;
      n2 = n2-d2*n0;
      n3 = n3-d3*n0;
      float n4 = -d4*n0;
      if (nd%2!=0) {
        n1 = -n1; n2 = -n2; n3 = -n3; n4 = -n4;
      }
      float yip4 = 0.0f, yip3 = 0.0f, yip2 = 0.0f, yip1 = 0.0f;
      float xip4 = 0.0f, xip3 = 0.0f, xip2 = 0.0f, xip1 = 0.0f;
      for (int i=n-1; i>=0; --i) {
        float xi = x[i];
        float yi = n1*xip1+n2*xip2+n3*xip3+n4*xip4 -
                   d1*yip1-d2*yip2-d3*yip3-d4*yip4;
        y[i] += yi;
        yip4 = yip3; yip3 = yip2; yip2 = yip1; yip1 = yi;
        xip4 = xip3; xip3 = xip2; xip2 = xip1; xip1 = xi;
      }
    }

    void applyXN(int nd, float** x, float** y, int nn1, int nn2) {
      cout<<"go smoothing "<<endl;
      //checkArrays(x,y);
      //if (sameArrays(x,y)) x = copy(x);
      float n0 = _n0[nd], n1 = _n1[nd], n2 = _n2[nd], n3 = _n3[nd];
      float d1 = _d1[nd], d2 = _d2[nd], d3 = _d3[nd], d4 = _d4[nd];
      float* yim4 = zerofloat(nn1);
      float* yim3 = zerofloat(nn1);
      float* yim2 = zerofloat(nn1);
      float* yim1 = zerofloat(nn1);
      float* xim4 = zerofloat(nn1);
      float* xim3 = zerofloat(nn1);
      float* xim2 = zerofloat(nn1);
      float* xim1 = zerofloat(nn1);
      float* yi = zerofloat(nn1);
      float* xi = zerofloat(nn1);
      float* yt4 = yim4;
      float* yt3 = yim3;
      float* yt2 = yim2;
      float* yt1 = yim1;
      float* xt4 = xim4;
      float* xt3 = xim3;
      float* xt2 = xim2;
      float* xt1 = xim1;
      float* yt0 = yi;
      float* xt0 = xi;
      float *x2, *y2, *xt, *yt;
      for (int i2=0; i2<nn2; ++i2) {
        x2 = x[i2];
        y2 = y[i2];
        for (int i1=0; i1<nn1; ++i1) {
          xi[i1] = x2[i1];
          yi[i1] = n0*xi[i1]+n1*xim1[i1]+n2*xim2[i1]+n3*xim3[i1]
                            -d1*yim1[i1]-d2*yim2[i1]-d3*yim3[i1]-d4*yim4[i1];
          y2[i1] = yi[i1];
        }
        yt = yim4; yim4 = yim3; yim3 = yim2; yim2 = yim1; yim1 = yi; yi = yt;
        xt = xim3; xim3 = xim2; xim2 = xim1; xim1 = xi; xi = xt;
      }
      n1 -= d1*n0; n2 -= d2*n0; n3 -= d3*n0;
      float n4 = -d4*n0;
      if (nd%2!=0) {n1 = -n1; n2 = -n2; n3 = -n3; n4 = -n4;}
      float* yip4 = yim4;
      float* yip3 = yim3;
      float* yip2 = yim2;
      float* yip1 = yim1;
      float* xip4 = xim4;
      float* xip3 = xim3;
      float* xip2 = xim2;
      float* xip1 = xim1;
      for (int i1=0; i1<nn1; ++i1) {
        yip4[i1] = yip3[i1] = yip2[i1] = yip1[i1] = 0.0f;
        xip4[i1] = xip3[i1] = xip2[i1] = xip1[i1] = 0.0f;
      }
      for (int i2=nn2-1; i2>=0; --i2) {
        x2 = x[i2];
        y2 = y[i2];
        for (int i1=0; i1<n1; ++i1) {
          xi[i1] = x2[i1];
          yi[i1] = n1*xip1[i1]+n2*xip2[i1]+n3*xip3[i1]+n4*xip4[i1] -
                   d1*yip1[i1]-d2*yip2[i1]-d3*yip3[i1]-d4*yip4[i1];
          y2[i1] += yi[i1];
        }
        yt = yip4; yip4 = yip3; yip3 = yip2; yip2 = yip1; yip1 = yi; yi = yt;
        xt = xip4; xip4 = xip3; xip3 = xip2; xip2 = xip1; xip1 = xi; xi = xt;
      }
      delete [] yt4; delete [] yt3; delete [] yt2; delete [] yt1; 
      delete [] xt4; delete [] xt3; delete [] xt2; delete [] xt1; 
      delete [] yt0; delete [] xt0; 
    }

    // Coefficients computed using Deriche's method. These coefficients
    // were computed for sigma = 100 and 0 <= x <= 10*sigma = 1000,
    // using the Mathematica function FindFit. The coefficients have
    // roughly 10 digits of precision.
    // 0th derivative.
    private:
    static const double a00 =  1.6797292232361107;
    static const double a10 =  3.7348298269103580;
    static const double b00 =  1.7831906544515104;
    static const double b10 =  1.7228297663338028;
    static const double c00 = -0.6802783501806897;
    static const double c10 = -0.2598300478959625;
    static const double w00 =  0.6318113174569493;
    static const double w10 =  1.9969276832487770;
    // 1st derivative.
    static const double a01 =  0.6494024008440620;
    static const double a11 =  0.9557370760729773;
    static const double b01 =  1.5159726670750566;
    static const double b11 =  1.5267608734791140;
    static const double c01 = -0.6472105276644291;
    static const double c11 = -4.5306923044570760;
    static const double w01 =  2.0718953658782650;
    static const double w11 =  0.6719055957689513;
    // 2nd derivative.
    static const double a02 =  0.3224570510072559;
    static const double a12 = -1.7382843963561239;
    static const double b02 =  1.3138054926516880;
    static const double b12 =  1.2402181393295362;
    static const double c02 = -1.3312275593739595;
    static const double c12 =  3.6607035671974897;
    static const double w02 =  2.1656041357418863;
    static const double w12 =  0.7479888745408682;
    double a0[3], a1[3], b0[3], b1[3], c0[3], c1[3], w0[3], w1[3];
    //
    float *_n0, *_n1, *_n2, *_n3; // numerator coefficients
    float *_d1, *_d2, *_d3, *_d4; // denominator coefficient
    /**
     * Makes Deriche's numerator and denominator coefficients.
     */
    private:
    void makeND(double sigma) {
      _n0 = new float[3];
      _n1 = new float[3];
      _n2 = new float[3];
      _n3 = new float[3];
      _d1 = new float[3];
      _d2 = new float[3];
      _d3 = new float[3];
      _d4 = new float[3];

      // For 0th, 1st, and 2nd derivatives, ...
      for (int i=0; i<3; ++i) {
        double n0 = (i%2==0)?a0[i]+c0[i]:0.0;
        double n1 = exp(-b1[i]/sigma) * (
                      c1[i]*sin(w1[i]/sigma) -
                      (c0[i]+2.0*a0[i])*cos(w1[i]/sigma)) +
                    exp(-b0[i]/sigma) * (
                      a1[i]*sin(w0[i]/sigma) -
                      (2.0*c0[i]+a0[i])*cos(w0[i]/sigma));
        double n2 = 2.0*exp(-(b0[i]+b1[i])/sigma) * (
                      (a0[i]+c0[i])*cos(w1[i]/sigma)*cos(w0[i]/sigma) -
                      a1[i]*cos(w1[i]/sigma)*sin(w0[i]/sigma) -
                      c1[i]*cos(w0[i]/sigma)*sin(w1[i]/sigma)) +
                    c0[i]*exp(-2.0*b0[i]/sigma) +
                    a0[i]*exp(-2.0*b1[i]/sigma);
        double n3 = exp(-(b1[i]+2.0*b0[i])/sigma) * (
                      c1[i]*sin(w1[i]/sigma) -
                      c0[i]*cos(w1[i]/sigma)) +
                    exp(-(b0[i]+2.0*b1[i])/sigma) * (
                      a1[i]*sin(w0[i]/sigma) -
                      a0[i]*cos(w0[i]/sigma));
        double d1 = -2.0*exp(-b0[i]/sigma)*cos(w0[i]/sigma) -
                     2.0*exp(-b1[i]/sigma)*cos(w1[i]/sigma);
        double d2 = 4.0*exp(-(b0[i]+b1[i])/sigma) *
                      cos(w0[i]/sigma)*cos(w1[i]/sigma) +
                    exp(-2.0*b0[i]/sigma) +
                    exp(-2.0*b1[i]/sigma);
        double d3 = -2.0*exp(-(b0[i]+2.0*b1[i])/sigma)*cos(w0[i]/sigma) -
                     2.0*exp(-(b1[i]+2.0*b0[i])/sigma)*cos(w1[i]/sigma);
        double d4 = exp(-2.0*(b0[i]+b1[i])/sigma);
        _n0[i] = (float)n0;
        _n1[i] = (float)n1;
        _n2[i] = (float)n2;
        _n3[i] = (float)n3;
        _d1[i] = (float)d1;
        _d2[i] = (float)d2;
        _d3[i] = (float)d3;
        _d4[i] = (float)d4;
      }
      scaleN(sigma);
    }
    /**
     * Scales numerator filter coefficients to normalize the filters.
     * For example, the sum of the 0th-derivative filter coefficients
     * should be 1.0. The scale factors are computed from finite-length
     * approximations to the impulse responses of the three filters.
     */
    void scaleN(double sigma) {
      int n = 1+2*(int)(10.0*sigma);
      float* x = new float[n];
      float* y0 = new float[n];
      float* y1 = new float[n];
      float* y2 = new float[n];
      for (int i=0; i<n; ++i) x[i]=0;
      int m = (n-1)/2;
      x[m] = 1.0f;
      applyN(0,x,y0,n);
      applyN(1,x,y1,n);
      applyN(2,x,y2,n);
      double s[3];
      for (int i=0; i<3; ++i) s[i]=0;
      for (int i=0,j=n-1; i<j; ++i,--j) {
        double t = i-m;
        s[0] += y0[j]+y0[i];
        s[1] += sin(t/sigma)*(y1[j]-y1[i]);
        s[2] += cos(t*sqrt(2.0)/sigma)*(y2[j]+y2[i]);
      }
      s[0] += y0[m];
      s[2] += y2[m];
      s[1] *= sigma*exp(0.5);
      s[2] *= -(sigma*sigma)/2.0*exp(1.0);
      for (int i=0; i<3; ++i) {
        _n0[i] /= s[i];
        _n1[i] /= s[i];
        _n2[i] /= s[i];
        _n3[i] /= s[i];
      }
      x=0; y0=0; y1=0; y2=0;
      delete x; delete y0; delete y1; delete y2;
    }
  };
  
  /////////////////////////////////////////////////////////////////////////////////
  class VanVlietFilter: public Filter {
    public:
    VanVlietFilter(double sigma) {}//makeG(sigma);}
  };  
};
}

#endif
