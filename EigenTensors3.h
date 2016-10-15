#ifndef _EIGENTENSORS3_H 
#define _EIGENTENSORS3_H 

#include <limits.h>
#include <float.h>

#include "Tensors3.h"
#include "../util/ArrayMath.h"

using namespace std;
using namespace util;

/**
* An array of eigen-decompositions of tensors for 3D image processing.
* Each tensor is a symmetric positive-semidefinite 3-by-3 matrix:
* <pre><code>
* |a11 a12 a13|
* A = |a12 a22 a23|
* |a13 a23 a33|
* </code></pre>
* Such tensors can be used to parameterize anisotropic image processing.
* <p>
* The eigen-decomposition of the matrix A is
* <pre><code>
* A = au*u*u' + av*v*v' + aw*w*w'
* = (au-av)*u*u' + (aw-av)*w*w' + av*I
* </code></pre>
* where u, v, and w are orthogonal unit eigenvectors of A. (The notation
* u' denotes the transpose of u.) The outer products of eigenvectors are
* scaled by the non-negative eigenvalues au, av, and aw. The second
* equation exploits the identity u*u' + v*v' + w*w' = I, and makes
* apparent the redundancy of the vector v.
* <p>
* Only the 1st and 2nd components of the eigenvectors u and w are stored.
* Except for a sign, the 3rd components may be computed from the 1st and
* 2nd. Because the tensors are independent of the choice of sign, the
* eigenvectors u and w are stored with an implied non-negative 3rd
* component.
* <p>
* Storage may be further reduced by compression, whereby eigenvalues
* and eigenvectors are quantized. Quantization errors for eigenvalues
* (au,av,aw) are less than 0.001*(au+av+aw). Quantization errors for
* eigenvectors are less than one degree of arc on the unit sphere.
* Memory required to store each tensor is 12 bytes if compressed, and
* 28 bytes if not compressed.
*
* @author Luming Liang, translated from Mines Java Toolkit written 
* by Dave Hale
* @version 2012.06.21
*/

namespace dsp {

  class EigenTensors3 : public Tensors3 {
    private:
      float AS_SET, AS_GET;
      void initConst() {
        AS_SET = (float)(SHRT_MAX);
        AS_GET = 1.0f/AS_SET;
      }
    public:
      EigenTensors3() {
        initConst();
      }
  };

}
#endif
