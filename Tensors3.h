#ifndef _TENSORS3_H
#define _TENSORS3_H

namespace dsp {
  class Tensors3 {
    public:
    virtual void getTensor(int i1, int i2, int i3, float* a);
  };
}
#endif
