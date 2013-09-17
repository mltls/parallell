#include <stdio.h>
#include <iomanip>
#include <curand_kernel.h>

#define RUNS_PER_THREAD 1000000

#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

__host__ timespec GetTime();
__host__ double InSeconds(const timespec ts);

__global__
void ThrowDice(const long seed, int *sum_device, curandState* rng_states) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int *sum = &sum_device[index];
  curandState *rng_state = &rng_states[index];
  curand_init(seed,index,0,rng_state);
  int sum2 = 0;
  for (int i=0;i<RUNS_PER_THREAD;i++) {
    float u1 = curand_uniform(rng_state);
    float u2 = curand_uniform(rng_state);
    float r = sqrt(u1*u1 + u2*u2);
    sum2 += 1 - (int)r;
  }
  *sum = sum2;
}

__host__
double FindPi() {
  int threads = 65536 / 2;
  int threadsPerBlock = 256;
  int blocksPerGrid = (threads - 1) / threadsPerBlock + 1;
  int *sum_host = new int[threads];
  for (int i=0;i<threads;i++) sum_host[i] = 0;
  int *sum_device = 0;
  curandState *rng_states = 0;
  cudaMalloc((void**)&sum_device,sizeof(int)*threads);
  cudaMalloc((void**)&rng_states,sizeof(curandState)*threads);
  long int seed = (long int)InSeconds(GetTime());
  ThrowDice<<<blocksPerGrid,threadsPerBlock>>>(seed,sum_device,rng_states);
  cudaMemcpy((void*)sum_host,(void*)sum_device,threads*sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(sum_device);
  cudaFree(rng_states);
  long int sum = 0;
  for (int i=0;i<threads;i++) {
    sum += sum_host[i];
  }
  double runs = ((double)threads) * RUNS_PER_THREAD;
  double result = 4.0 * (double)sum / runs;
  printf("%li / %e hit inside the circle.\n",sum,runs);
  printf("Result after %.2e runs: %.10f\n",runs,result);
  return result;
}

__host__ main() {
  int iterations = 1;
  double final = 0;
  for (int i=0;i<iterations;i++) {
    final += FindPi();
  }
  final /= (double)iterations;
  if (iterations > 1) printf("Sum of results: %.10f\n",final);
  return 0;
}

__host__
timespec GetTime() {

  struct timespec ts;

  #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts.tv_sec = mts.tv_sec;
  ts.tv_nsec = mts.tv_nsec;

  #else
  clock_gettime(CLOCK_REALTIME, &ts);
  #endif

  return ts;

}

__host__
double InSeconds(const timespec ts) {
  double s(ts.tv_sec);
  return s + (1e-9*(double)ts.tv_nsec);
}