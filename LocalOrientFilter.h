#ifndef _LOCALORIENTFILTER_H
#define _LOCALORIENTFILTER_H

#include <math.h>
#include <float.h>
#include <iostream>

#include "EigenTensors2.h"
#include "EigenTensors3.h"
#include "RecursiveGaussianFilter.h"
#include "../util/ArrayMath.h"

using namespace std;
using namespace util;

namespace dsp {
  
  /**
   * Local estimates of orientations of features in images.
   * Methods of this class can compute for each image sample numerous
   * parameters related to orientation. All orientation information
   * is derived from eigenvectors and eigenvalues of the structure tensor
   * (also called the "gradient-squared tensor"). This tensor is equivalent
   * to a matrix of 2nd partial derivatives of an autocorrelation evaluated
   * at zero lag. In other words, orientation is here determined by the
   * (2-D) ellipse or (3-D) ellipsoid that best fits the peak of the
   * autocorrelation of image samples in a local window.
   * <p>
   * The coordinate system for a 2-D image has two orthogonal axes 1 and 2,
   * which correspond to the 1st and 2nd indices of the array containing
   * image samples. For 2-D images, the eigenvectors are the unit vectors
   * u = (u1,u2) and v = (v1,v2). The 1st eigenvector u is perpendicular
   * to the best fitting line, and the 1st component u1 of u is always
   * non-negative. The 2nd eigenvector v is perpendicular to u such that
   * the cross product u1*v2-u2*v1 = 1; that is, v1 = -u2 and v2 = u1.
   * The angle theta = asin(u2) is the angle measured counter-clockwise
   * between the 1st eigenvector u and axis 1; -pi/2 &lt;= theta &lt;= pi/2.
   * <p>
   * The coordinate system for a 3-D image has three orthogonal axes 1, 2
   * and 3, which correspond to the 1st, 2nd and 3rd indices of the array
   * containing image samples. For 3-D images, the eigenvectors are unit
   * vectors u = (u1,u2,u3), v = (v1,v2,v3), and w = (w1,w2,w3). The 1st
   * eigenvector u is orthogonal to the best fitting plane, and the 1st
   * component u1 of u is always non-negative. The 2nd eigenvector v is
   * orthogonal to the best fitting line within the best fitting plane.
   * The 3rd eigenvector w is orthogonal to both u and v and is aligned
   * with the direction in which the images changes least. The dip angle
   * theta = acos(u1) is the angle between the 1st eigenvector u and axis 1;
   * 0 &lt;= theta &lt;= pi/2. The azimuthal angle phi = atan2(u3,u2)
   * is well-defined for only non-zero theta; -pi &lt;= phi &lt;= pi.
   * <p>
   * The local linearity or planarity of features is determined by the
   * eigenvalues. For 2-D images with eigenvalues eu and ev (corresponding
   * to the eigenvectors u and v), linearity is (eu-ev)/eu. For 3-D
   * images with eigenvalues eu, ev, and ew, planarity is (eu-ev)/eu
   * and linearity is (ev-ew)/eu. Both linearity and planarity are
   * in the range [0,1].
   *
   * @author Luming Liang, translated from Mines Java Toolkit
   * @version 2012.06.07
   */

  class LocalOrientFilter {
    
    ///////////////////////////////////////////////////////////////////////////
    // private

    private: 
    RecursiveGaussianFilter* _rgfGradient1;
    RecursiveGaussianFilter* _rgfGradient2;
    RecursiveGaussianFilter* _rgfGradient3;
    RecursiveGaussianFilter* _rgfSmoother1;
    RecursiveGaussianFilter* _rgfSmoother2;
    RecursiveGaussianFilter* _rgfSmoother3;

    void computeGradientProducts(float*** g1, float*** g2, float*** g3,
      float*** g11, float*** g12, float*** g13, float*** g22, float*** g23, 
      float*** g33, int n1, int n2, int n3) {
        for (int i3=0; i3<n3; ++i3) 
          for (int i2=0; i2<n2; ++i2) {
            float *g1i = g1[i3][i2], *g2i = g2[i3][i2], *g3i = g3[i3][i2];
            float *g11i = g11[i3][i2], *g12i = g12[i3][i2], *g13i = g13[i3][i2];
            float *g22i = g22[i3][i2], *g23i = g23[i3][i2], *g33i = g33[i3][i2];
            for (int i1=0; i1<n1; ++i1) {
              float g1ii = g1i[i1], g2ii = g2i[i1], g3ii = g3i[i1];
              g11i[i1] = g1ii*g1ii;
              g22i[i1] = g2ii*g2ii;
              g33i[i1] = g3ii*g3ii;
              g12i[i1] = g1ii*g2ii;
              g13i[i1] = g1ii*g3ii;
              g23i[i1] = g2ii*g3ii;
            }
          }
     }
     
    void solveEigenproblems(
      float*** g11, float*** g12, float*** g13,
      float*** g22, float*** g23, float*** g33,
      float*** theta, float*** phi,
      float*** u1, float*** u2, float*** u3,
      float*** v1, float*** v2, float*** v3,
      float*** w1, float*** w2, float*** w3,
      float*** eu, float*** ev, float*** ew,
      float*** ep, float*** el, int n1, int n2, int n3) {
      double** a = new double*[3];
      double** z = new double*[3];
      for (int i=0; i<3; ++i) {
        a[i] = new double[3];
        z[i] = new double[3];
      }
      double* e = new double[3];
      for (int i3=0; i3<n3; ++i3)
        for (int i2=0; i2<n2; ++i2) {
          for (int i1=0; i1<n1; ++i1) {
            a[0][0] = g11[i3][i2][i1]; 
            a[0][1] = g12[i3][i2][i1];
            a[0][2] = g13[i3][i2][i1];
            a[1][0] = g12[i3][i2][i1];
            a[1][1] = g22[i3][i2][i1];
            a[1][2] = g23[i3][i2][i1];
            a[2][0] = g13[i3][i2][i1];
            a[2][1] = g23[i3][i2][i1];
            a[2][2] = g33[i3][i2][i1];
            Eigen::solveSymmetric33(a,z,e);
            float u1i = (float)z[0][0];
            float u2i = (float)z[0][1];
            float u3i = (float)z[0][2];
            float v1i = (float)z[1][0];
            float v2i = (float)z[1][1];
            float v3i = (float)z[1][2];
            float w1i = (float)z[2][0];
            float w2i = (float)z[2][1];
            float w3i = (float)z[2][2];
            if (u1i<0.0f) {u1i = -u1i; u2i = -u2i; u3i = -u3i;}
            if (v2i<0.0f) {v1i = -v1i; v2i = -v2i; v3i = -v3i;}
            if (w3i<0.0f) {w1i = -w1i; w2i = -w2i; w3i = -w3i;}
            float eui = (float)e[0], evi = (float)e[1], ewi = (float)e[2];
            if (ewi<0.0f) ewi = 0.0f;
            if (evi<ewi) evi = ewi;
            if (eui<evi) eui = evi;
            if (theta!=NULL) theta[i3][i2][i1] = acos(u1i);
            if (phi!=NULL) phi[i3][i2][i1] = atan2(u3i,u2i);
            if (u1!=NULL) u1[i3][i2][i1] = u1i;
            if (u2!=NULL) u2[i3][i2][i1] = u2i;
            if (u3!=NULL) u3[i3][i2][i1] = u3i;
            if (v1!=NULL) v1[i3][i2][i1] = v1i;
            if (v2!=NULL) v2[i3][i2][i1] = v2i;
            if (v3!=NULL) v3[i3][i2][i1] = v3i;
            if (w1!=NULL) w1[i3][i2][i1] = w1i;
            if (w2!=NULL) w2[i3][i2][i1] = w2i;
            if (w3!=NULL) w3[i3][i2][i1] = w3i;
            if (eu!=NULL) eu[i3][i2][i1] = eui;
            if (ev!=NULL) ev[i3][i2][i1] = evi;
            if (ew!=NULL) ew[i3][i2][i1] = ewi;
            if (ep!=NULL || el!=NULL) {
              float esi = (eui>0.0f)?1.0f/eui:1.0f;
              if (ep!=NULL) ep[i3][i2][i1] = (eui-evi)*esi;
              if (el!=NULL) el[i3][i2][i1] = (evi-ewi)*esi;
            }
          }
        }
      delete [] a; delete [] z; delete [] e;
    }
  
    //////////////////////////////////////////////////////////////////////////
    // public
    public:
    /**
     * Constructs a filter with a possibly anisotropic Gaussian window.
     * @param sigma1 half-width of window in 1st dimension.
     * @param sigma2 half-width of window in 2nd dimension.
     * @param sigma3 half-width of window in 3rd and higher dimensions.
     */
    LocalOrientFilter(double sigma1, double sigma2, double sigma3) {
      _rgfSmoother1 = (sigma1>=1.0)?new RecursiveGaussianFilter(sigma1):NULL;
      if (sigma2==sigma1) _rgfSmoother2 = _rgfSmoother1;
      else 
        _rgfSmoother2 = (sigma2>=1.0)?new RecursiveGaussianFilter(sigma2):NULL;
      if (sigma3==sigma2) _rgfSmoother3 = _rgfSmoother2;
      else 
        _rgfSmoother3 = (sigma3>=1.0)?new RecursiveGaussianFilter(sigma3):NULL;
      setGradientSmoothing(1.0);
    }
    LocalOrientFilter(double sigma) {
      LocalOrientFilter(sigma,sigma,sigma);
    }
    LocalOrientFilter(double sigma1, double sigma2) {
      LocalOrientFilter(sigma1,sigma2,sigma2);
    }
    /**
     * Sets half-widths of Gaussian derivative filters used to compute 
     * gradients.
     * Typically, these half-widths should not exceed one-fourth those of the
     * the corresponding Gaussian windows used to compute local averages of
     * gradient-squared tensors.
     * The default half-widths for Gaussian derivatives is 1.0.
     * @param sigma1 half-width of derivative in 1st dimension.
     * @param sigma2 half-width of derivative in 2nd dimension.
     * @param sigma3 half-width of derivatives in 3rd and higher dimensions.
     */
    void setGradientSmoothing(double sigma1, double sigma2, double sigma3){
      _rgfGradient1 = new RecursiveGaussianFilter(sigma1);
      if (sigma2==sigma1) _rgfGradient2 = _rgfGradient1;
      else _rgfGradient2 = new RecursiveGaussianFilter(sigma2);
      if (sigma3==sigma2) _rgfGradient3 = _rgfGradient2;
      else _rgfGradient3 = new RecursiveGaussianFilter(sigma3);
    }
    void setGradientSmoothing(double sigma) {
      setGradientSmoothing(sigma,sigma,sigma);
    }
    void setGradientSmoothing(double sigma1, double sigma2) {
      setGradientSmoothing(sigma1,sigma2,sigma2);
    }

    /**
     * Applies this filter for the specified image and outputs. All
     * outputs are optional and are computed for only non-null arrays.
     * @param x input array for 2-D image
     * @param theta orientation angle = asin(u2); -pi &lt;= theta &lt;= pi
     * @param u1 1st component of 1st eigenvector.
     * @param u2 2nd component of 1st eigenvector.
     * @param v1 1st component of 2nd eigenvector.
     * @param v2 2nd component of 2nd eigenvector.
     * @param eu largest eigenvalue corresponding to the eigenvector u.
     * @param ev smallest eigenvalue corresponding to the eigenvector v.
     * @param el (eu-ev)/eu, a measure of linearity.
     */
    void apply(float** x,
      float** theta,
      float** u1, float** u2,
      float** v1, float** v2,
      float** eu, float** ev,
      float** el, int n1, int n2) {
      // Where possible, use output arrays for workspace.
      float*** t = new float**[8];
      int nt = 0;
      if (theta!=NULL) t[nt++] = theta;
      if (u1!=NULL) t[nt++] = u1;
      if (u2!=NULL) t[nt++] = u2;
      if (v1!=NULL) t[nt++] = v1;
      if (v2!=NULL) t[nt++] = v2;
      if (eu!=NULL) t[nt++] = eu;
      if (ev!=NULL) t[nt++] = ev;
      if (el!=NULL) t[nt++] = el;
      // Gradient.
      float** g1 = new float*[n2];
      float** g2 = new float*[n2];
      for (int i2=0; i2<n2; ++i2) {
        g1[i2] = new float[n1];
        g2[i2] = new float[n1];
      }
      _rgfGradient1->apply10(x,g1,n1,n2);
      _rgfGradient2->apply01(x,g2,n1,n2);
      if (nt>0) t[0] = copy(g1,n1,n2);
      if (nt>1) t[1] = copy(g2,n1,n2);

      // Gradient products.
      float** g11 = g1;
      float** g22 = g2;
      float** g12 = new float*[n2];
      for (int i2=0; i2<n2; ++i2) {
        g12[i2] = new float[n1];
        for (int i1=0; i1<n1; ++i1) {
          float g1i = g1[i2][i1];
          float g2i = g2[i2][i1];
          g11[i2][i1] = g1i*g1i;
          g22[i2][i1] = g2i*g2i;
          g12[i2][i1] = g1i*g2i;
        }
      }
      if (nt>2) t[2] = copy(g12,n1,n2);
    
      // Smoothed gradient products comprise the structure tensor.
      if (_rgfSmoother1!=NULL || _rgfSmoother2!=NULL) {
        float** h = new float*[n2];
        for (int i2=0; i2<n2; ++i2) h[i2] = new float[n1];
        float*** gs = new float**[3];
        gs[0] = g11; gs[1] = g22; gs[2] = g12;
        for (int i=0; i<3; ++i) {
          if (_rgfSmoother1!=NULL) _rgfSmoother1->apply0X(gs[i],h,n1,n2);
          else copy(gs[i],h,n1,n2);
          if (_rgfSmoother2!=NULL) _rgfSmoother2->applyX0(h,gs[i],n1,n2);
          else copy(h,gs[i],n1,n2);
        }
        if (nt>3) t[3] = copy(h,n1,n2);
        delete [] gs; delete [] h;
      }

      // Compute eigenvectors, eigenvalues, and outputs that depend on them.
      float a[2][2];
      float z[2][2];
      float e[2];
      for (int i2=0; i2<n2; ++i2) {
        for (int i1=0; i1<n1; ++i1) {
          a[0][0] = g11[i2][i1];
          a[0][1] = g12[i2][i1];
          a[1][0] = g12[i2][i1];
          a[1][1] = g22[i2][i1];
          Eigen::solveSymmetric22(a,z,e);
          float u1i = z[0][0];
          float u2i = z[0][1];
          if (u1i<0.0f) {
            u1i = -u1i;
            u2i = -u2i;
          }
          float v1i = -u2i;
          float v2i = u1i;
          float eui = e[0];
          float evi = e[1];
          if (evi<0.0f) evi = 0.0f;
          if (eui<evi) eui = evi;
          if (theta!=NULL) theta[i2][i1] = asin(u2i);
          if (u1!=NULL) u1[i2][i1] = u1i;
          if (u2!=NULL) u2[i2][i1] = u2i;
          if (v1!=NULL) v1[i2][i1] = v1i;
          if (v2!=NULL) v2[i2][i1] = v2i;
          if (eu!=NULL) eu[i2][i1] = eui;
          if (ev!=NULL) ev[i2][i1] = evi;
          if (el!=NULL) el[i2][i1] = (eui-evi)/eui;
        }
      }
    }

    /**
     * Applies this filter to estimate orientation angles.
     * @param x input array for 2-D image.
     * @param theta orientation angle; -pi &lt;= theta &lt;= pi
     */
    void applyForTheta(float** x, float** theta, int n1, int n2) {
      apply(x,
        theta,
        NULL,NULL,
        NULL,NULL,
        NULL,NULL,
        NULL,n1,n2);
    }
    /**
     * Applies this filter to estimate normal vectors (1st eigenvectors).
     * @param x input array for 2-D image.
     * @param u1 1st component of normal vector.
     * @param u2 2nd component of normal vector.
     */
    void applyForNormal(float** x, float** u1, float** u2, int n1, int n2) {
      apply(x,
        NULL,
        u1,u2,
        NULL,NULL,
        NULL,NULL,
        NULL,n1,n2);
    }
    /**
     * Applies this filter to estimate normal vectors and linearities.
     * @param x input array for 2-D image.
     * @param u1 1st component of normal vector.
     * @param u2 2nd component of normal vector.
     * @param el linearity in range [0,1].
     */
    void applyForNormalLinear(float** x,float** u1, float** u2, float** el, 
      int n1, int n2) {
      apply(x,
        NULL,
        u1,u2,
        NULL,NULL,
        NULL,NULL,
        el,n1,n2);
    }

    /**
     * Applies this filter to estimate 2-D structure tensors.
     * @param x input array for 2-D image.
     * @return structure tensors.
     */
    EigenTensors2* applyForTensors(float** x, int n1, int n2) {
      float** u1 = new float*[n2];
      float** u2 = new float*[n2];
      float** eu = new float*[n2];
      float** ev = new float*[n2];
      for (int i2=0; i2<n2; ++i2) {
        u1[i2] = new float[n1];
        u2[i2] = new float[n1];
        eu[i2] = new float[n1];
        ev[i2] = new float[n1];
      }
      apply(x,
        NULL,
        u1,u2,
        NULL,NULL,
        eu,ev,
        NULL,n1,n2);
      return new EigenTensors2(u1,u2,eu,ev,n1,n2);
      delete [] u1; delete [] u2; delete [] eu; delete [] ev;
    }

    // ready for 3d...
  };

}

#endif
