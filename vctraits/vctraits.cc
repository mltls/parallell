#include <iostream>
#include <Vc/Vc>

enum CtEnum { kScalar, kVc };

template <CtEnum ct>
struct CtTraits {};

template <>
struct CtTraits<kScalar> {
  typedef int    int_t;
  typedef float  float_t;
  typedef double double_t;
};

template <>
struct CtTraits<kVc> {
  typedef Vc::Memory<Vc::int_v>    int_t;
  typedef Vc::Memory<Vc::float_v>  float_t;
  typedef Vc::Memory<Vc::double_v> double_t;
};

template <CtEnum ct>
inline void AddVector(
    const size_t count,
    const typename CtTraits<ct>::double_t *a,
    const typename CtTraits<ct>::double_t *b,
          typename CtTraits<ct>::double_t *c) {

  // Use specific implementations

}

template <>
inline void AddVector<kScalar>(
    const size_t elements,
    const CtTraits<kScalar>::double_t *a_arr,
    const CtTraits<kScalar>::double_t *b_arr,
          CtTraits<kScalar>::double_t *c_arr) {

  for (size_t i=0; i<elements; i++) {
    c_arr[i] = a_arr[i] + b_arr[i];
  }

}

template <>
inline void AddVector<kVc>(
    const size_t vectors,
    const CtTraits<kVc>::double_t *a_mem,
    const CtTraits<kVc>::double_t *b_mem,
          CtTraits<kVc>::double_t *c_mem) {

  for (size_t i=0; i<vectors; i++) {
    const Vc::double_v a_vec = a_mem->vector(i);
    const Vc::double_v b_vec = b_mem->vector(i);
    c_mem->vector(i) = a_vec + b_vec;
  }

}

int main(void) {

  const double a[] = { 1,  2,  3,  4,  5,  6,  7,  8};
  const double b[] = {10, 20, 30, 40, 50, 60, 70, 80};
  double c[8];

  Vc::Memory<Vc::double_v> a_vc(8);
  Vc::Memory<Vc::double_v> b_vc(8);
  for (size_t i=0; i<8; i++) {
    a_vc[i] = a[i];
    b_vc[i] = b[i];
  }
  Vc::Memory<Vc::double_v> c_vc(8);

  AddVector<kScalar>(8,a,b,c);
  AddVector<kVc>(a_vc.vectorsCount(),&a_vc,&b_vc,&c_vc);

  std::cout << "Output vectors:" << std::endl;
  for (size_t i=0; i<8; i++) {
    std::cout << c[i] << " " << c_vc[i] << std::endl;
  }

  return 0;
}