#include <iostream>
#include <iomanip>
#include <Vc/Vc>

#define N_COORDINATES 1000

using namespace std;

int main(void) {
  Vc::Memory<Vc::float_v, N_COORDINATES> x_mem;
  Vc::Memory<Vc::float_v, N_COORDINATES> y_mem;
  Vc::Memory<Vc::float_v, N_COORDINATES> r_mem;
  Vc::Memory<Vc::float_v, N_COORDINATES> phi_mem;

  // Initialize cartesian arrays with random numbers
  for (size_t i=0;i<x_mem.vectorsCount();i++) {
    x_mem.vector(i) = Vc::float_v::Random() * 2.f - 1.f;
    y_mem.vector(i) = Vc::float_v::Random() * 2.f - 1.f;
  }

  // Convert to polar coordinates
  for (size_t i=0;i<x_mem.vectorsCount();i++) {
    const Vc::float_v x = x_mem.vector(i);
    const Vc::float_v y = y_mem.vector(i);

    r_mem.vector(i) = Vc::sqrt(x * x + y * y);
    Vc::float_v phi = Vc::atan2(y, x) * 57.295780181884765625f;
    phi(phi < 0.5f) += 360.f;
    phi_mem.vector(i) = phi;
  }

  // Print results
  for (size_t i=0;i<x_mem.entriesCount();i++) {
    cout << setw(3) << i << ": ";
    cout << setw(10) << x_mem[i] << ", " << setw(10) << y_mem[i] << " -> ";
    cout << setw(10) << r_mem[i] << ", " << setw(10) << phi_mem[i] << endl;
  }

  return 0;
}