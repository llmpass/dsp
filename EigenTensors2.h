#ifndef _EIGENTENSOR_H
#define _EIGENTENSOR_H

#include <math.h>
#include <float.h>

#include "../util/ArrayMath.h"
#include "Eigen.h"
#include "Tensors2.h"

using namespace util;

/**
* An array of eigen-decompositions of tensors for 2D image processing.
* Each tensor is a symmetric positive-semidefinite 2-by-2 matrix:
* <pre><code>
* A = |a11 a12|
* |a12 a22|
* </code></pre>
* Such tensors can be used to parameterize anisotropic image processing.
* <p>
* The eigen-decomposition of the matrix A is
* <pre><code>
* A = au*u*u' + av*v*v'
* = (au-av)*u*u' + av*I
* </code></pre>
* where u and v are orthogonal unit eigenvectors of A. (The notation u'
* denotes the transpose of u.) The outer products of eigenvectors are
* scaled by the non-negative eigenvalues au and av. The second equation
* exploits the identity u*u' + v*v' = I, and makes apparent the redundancy
* of the vector v.
*
* @author Luming Liang
* translated from Mines Java Toolkit written by Dave Hale.
*/

namespace dsp {
  class EigenTensors2 : public Tensors2 {
    public:
    /**
     * Constructs tensors for specified array dimensions. All eigenvalues
     * and eigenvectors are not set and are initially zero.
     * @param n1 number of tensors in 1st dimension.
     * @param n2 number of tensors in 2nd dimension.
     */
    EigenTensors2(int n1, int n2) {
      _n1 = n1; _n2 = n2;
      _au = new float*[n2];
      _av = new float*[n2];
      _u1 = new float*[n2];
      _u2 = new float*[n2];
      for (int i2=0; i2<n2; ++i2) {
        _au[i2] = new float[n1];
        _av[i2] = new float[n1];
        _u1[i2] = new float[n1];
        _u2[i2] = new float[n1];
      }
    }     
    /**
     * Constructs tensors for specified array dimensions and eigenvalues.
     * @param u1 array of 1st components of u.
     * @param u2 array of 2nd components of u.
     * @param au array of 1D eigenvalues.
     * @param av array of 2D eigenvalues.
     */
    EigenTensors2(float** u1, float** u2, float** au, float** av, 
      int n1, int n2) {
      EigenTensors2(n1,n2);
      float aui, avi, u1i, u2i;
      for (int i2=0; i2<_n2; ++i2) 
        for (int i1=0; i1<_n1; ++i1) {
          aui = au[i2][i1]; avi = av[i2][i1];
          u1i = u1[i2][i1]; u2i = u2[i2][i1];
          setEigenvalues(i1,i2,aui,avi);
          setEigenvectorU(i1,i2,u1i,u2i);
        }
    }
    /**
     * Constructs tensors from the specified tensors.
     * @param t the tensors from which to copy eigenvectors and eigenvalues.
     */
    EigenTensors2(EigenTensors2& t) {
      EigenTensors2(t._u1,t._u2,t._au,t._av,t._n1,t._n2);
    }
    ~EigenTensors2() {
      delete [] _au; delete [] _av; delete [] _u1; delete [] _u2;
    }
    int getN1() {return _n1;} 
    int getN2() {return _n2;}
    /**
     * Gets tensor elements for specified indices.
     * @param i1 index for 1st dimension.
     * @param i2 index for 2nd dimension.
     * @param a array {a11,a12,a22} of tensor elements.
     */
    void getTensor(int i1, int i2, float* a) {
      float au = _au[i2][i1];
      float av = _av[i2][i1];
      float u1 = _u1[i2][i1];
      float u2 = _u2[i2][i1];
      au -= av;
      a[0] = au*u1*u1+av; // a11
      a[1] = au*u1*u2 ; // a12
      a[2] = au*u2*u2+av; // a22
    }
    float* getTensor(int i1, int i2) {
      float* a = new float[3];
      getTensor(i1,i2,a);
      return a;
    }
    void getEigenvalues(int i1, int i2, float* a) {
      a[0] = _au[i2][i1];
      a[1] = _av[i2][i1];
    }
    float* getEigenvalues(int i1, int i2) {
      float* a = new float[2];
      getEigenvalues(i1,i2,a);
      return a;
    }
    void getEigenvalues(float** au, float** av) {
      copy(_au,au,_n1,_n2);
      copy(_av,av,_n1,_n2);
    }
    void getEigenvectorU(int i1, int i2, float* u) {
      u[0] = _u1[i2][i1];
      u[1] = _u2[i2][i1];
    }
    float* getEigenvectorU(int i1, int i2) {
      float* u = new float[2];
      getEigenvectorU(i1,i2,u);
      return u;
    }
    void getEigenvectorV(int i1, int i2, float* v) {
      v[0] = _u2[i2][i1];
      v[1] = -_u1[i2][i1];
    }
    float* getEigenvectorV(int i1, int i2) {
      float* v = new float[2];
      getEigenvectorV(i1,i2,v);
      return v;
    }
    void setTensor(int i1, int i2, float* a) {
      setTensor(i1,i2,a[0],a[1],a[2]);
    }
    /**
     * Sets tensor elements for specified indices.
     * This method first computes an eigen-decomposition of the specified
     * tensor, and then stores the computed eigenvectors and eigenvalues.
     * The eigenvalues are ordered such that au &gt;= av &gt;= 0.
     */
    void setTensor(int i1, int i2, float a11, float a12, float a22) {
      float aa[2][2] = {
        {a11,a12},
        {a12,a22}
      };
      float vv[2][2], ev[2];
      Eigen::solveSymmetric22(aa,vv,ev);
      float u[2]; 
      u[0] = vv[0][0]; u[1] = vv[0][1];
      float au = ev[0]; if (au<0.0f) au = 0.0f;
      float av = ev[1]; if (av<0.0f) av = 0.0f;
      setEigenvectorU(i1,i2,u);
      setEigenvalues(i1,i2,au,av);
    }
    void setEigenvalues(float au, float av) {
      fill(au,_au,_n1,_n2);
      fill(av,_av,_n1,_n2);
    }
    void setEigenvalues(int i1, int i2, float au, float av) {
      _au[i2][i1] = au;
      _av[i2][i1] = av;
    }
    void setEigenvalues(int i1, int i2, float* a) {
      setEigenvalues(i1,i2,a[0],a[1]);
    } 
    void setEigenvalues(float** au, float** av) {
      copy(au,_au,_n1,_n2);
      copy(av,_av,_n1,_n2);
    }
    void setEigenvectorU(int i1, int i2, float u1, float u2) {
      _u1[i2][i1] = u1;
      _u2[i2][i1] = u2;
    }
    void setEigenvectorU(int i1, int i2, float* u) {
      setEigenvectorU(i1,i2,u[0],u[1]);
    }
    void scale(float** s) {
      for (int i2=0; i2<_n2; ++i2) 
        for (int i1=0; i1<_n1; ++i1) {
          float si = s[i2][i1];
          _au[i2][i1] *= si;
          _av[i2][i1] *= si;
        }
    }
    void invert() {
      for (int i2=0; i2<_n2; ++i2) 
        for (int i1=0; i1<_n1; ++i1) {
          _au[i2][i1] = 1.0f/_au[i2][i1];
          _av[i2][i1] = 1.0f/_av[i2][i1];
        }
    }
    /**
     * Inverts these tensors, assumed to be structure tensors.
     * After inversion, all eigenvalues are in the range (0,1].
     * Specifically, after inversion, 0 &lt; au &lt;= av &lt;= 1.
     * <p>
     * Before inversion, tensors are assumed to be structure tensors,
     * for which eigenvalues au are not less than their corresponding
     * eigenvalues av. (Any eigenvalues au for which this condition is
     * not satisfied are set equal to the corresponding eigenvalue av.)
     * Structure tensors can, for example, be computed using
     * {@link LocalOrientFilter}.
     * <p>
     * Then, if any eigenvalues are equal to zero, this method adds a
     * small fraction of the largest eigenvalue au to all eigenvalues.
     * If am is the minimum of the eigenvalues av after this perturbation,
     * then the parameter p0 is used to compute a0 = pow(am/av,p0) and
     * the parameter p1 is used to compute a1 = pow(av/au,p1). Inverted
     * eigenvalues are then au = a0*a1 and av = a0.
     * <p>
     * In this way, p0 emphasizes overall amplitude and p1 emphasizes
     * linearity. For amplitude-independent tensors with all eigenvalues
     * av equal to one, set p0 = 0.0. To enhance linearity, set p1 &gt; 1.0.
     * To simply invert (and normalize) these tensors, set p0 = p1 = 1.0.
     * @param p0 power for amplitude.
     * @param p1 power for linearity.
     */
    void invertStructure(double p0, double p1) {
      float amax = 0.0f;
      float amin = FLT_MAX;
      for (int i2=0; i2<_n2; ++i2) 
        for (int i1=0; i1<_n1; ++i1) {
          float aui = _au[i2][i1];
          float avi = _av[i2][i1];
          if (avi<0.0f) avi = 0.0f;
          if (aui< avi) aui = avi;
          if (avi<amin) amin = avi;
          if (aui>amax) amax = aui;
          _au[i2][i1] = aui;
          _av[i2][i1] = avi;
        }
      float aeps = max(FLT_MIN*100.0f,FLT_EPSILON*amax);
      amin += aeps; amax += aeps;
      float fp0 = (float)p0;
      float fp1 = (float)p1;
      for (int i2=0; i2<_n2; ++i2) 
        for (int i1=0; i1<_n1; ++i1) {
          float aui = _au[i2][i1]+aeps;
          float avi = _av[i2][i1]+aeps;
          float a0i = pow(amin/avi,fp0);
          float a1i = pow( avi/aui,fp1);
          _au[i2][i1] = a0i*a1i;
          _av[i2][i1] = a0i;
        }
    }

  ///////////////////////////////////////////////////////////////////////////
  // private

  private: 
  int _n1,_n2;
  float **_au, **_av, **_u1, **_u2;
  };
}

#endif
