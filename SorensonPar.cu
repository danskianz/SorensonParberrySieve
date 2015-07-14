/* SorensonPar.cu
   Parallel Implementation of Algorithm 4.1
   as discussed in Sorenson and Parberry's
   1994 paper "Two Fast Parallel Prime Number
   Sieves".

   Authors:
   Daniel Anzaldo
   David Frank
   Antonio Lanfranchi
*/

// Visual Studio Dependencies (Can be commented out)
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

// C dependencies
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// C++ dependencies
#include <algorithm>

typedef unsigned long long big;

// GLOBAL VARIABLES--------------------------------------

typedef struct Wheel_t	// Struct-of-Arrays Wheel
{
	bool * rp;	// Numbers relatively prime to m
	big * dist; // D s.t. x + d is the smallest integer >dist[x] relatively prime to m
} Wheel_k;

bool * S;	// Global shared bit array of numbers up to N
int P;		// Global number of processors

// HOST FUNCTION HEADERS---------------------------------

/*	gcd
	Host version of the Euclidean Method
*/
__host__ big gcd(big a, big b);

/*	EratosthenesSieve
	HELPER: for Algorithm 4.1 Sequential Portion
	The most basic form of generating primes.
	Used to help find the first k primes.
	Returns the k-th prime.
*/
big EratosthenesSieve(long double x);

/*	Algorithm 4.1 Sequential Portion
	Running Time: O(sqrt(n))
	Space: O(sqrt(n)) up to O(sqrt(n)/log log n)
*/
cudaError_t algorithm4_1(big n);

/*	Algorithm 4.1 Helper: Parallel Sieve
	All CUDA-related functionality goes here.
	This code will change for different kernel versions.
*/
cudaError_t parallelSieve(
	big n, big k, big m, const Wheel_k &wheel, big range);

/* Frees the memory allocated on the device and returns any errors*/
cudaError_t cleanup(bool *d_S, Wheel_k &wheel, cudaError_t cudaStatus);


// DEVICE MATH FUNCTIONS---------------------------------

/*	gcd_d
	Device version of the Euclidean Method
	find number c such that: a = sc, b = tc
*/
__device__ big gcd_d(big a, big b)
{
   big tmp;
   
   while (b!=0)
   {
      tmp = a;
      a = b;
      b = tmp%b;
   }
   return a;
}

/*	sqrt_d
	Device version of the Square Root Function
	Babylonian Method
*/
__device__ big sqrt_d(big a)
{
   big root = a/2;
   
   for (big n = 0; n < 10; n++)
   {
      root = 0.5 * (root + (a/root));
   }
   
   return root;
}

__device__ big min_d(big a, big b)
{
   return (a < b) ? a : b;
}

__device__ big max_d(big a, big b)
{
	return (a > b) ? a : b;
}


// ALGORITHM 4.1 KERNEL VERSIONS-------------------------

/*	Algorithm 4.1: Parallel Sieve Kernel version 1
	Parallelization: O(sqrt(n)) processors
	Space: O(sqrt(n)) up to O(sqrt(n)/log log n)
	PRAM Mode: Exclusive Read, Exclusive Write (EREW)
	Remarks: No optimizations yet performed.
	For n = 1 billion, it uses 31623 threads
*/
__global__ void parallelSieveKernel(
	big n, big k, big m, Wheel_k d_wheel, big range, bool *d_S)
{
	big sqrt_N = sqrt_d(n);

	// Express the sieve in thread mode.
	big i = threadIdx.x + blockIdx.x * blockDim.x;

	big L = range * i + 1;
	big R = min_d(range * (i + 1), n);

	/* Range Sieving */
	for (big x = L; x < R; x++)
		d_S[x] = d_wheel.rp[x % m];

	/* For every prime from prime[k] up to sqrt(N) */
	for (big q = k; q < sqrt_N; q++)
	{
		if (d_S[q])
		{
			/* Compute smallest f s.t.
			gcd_d(qf, m) == 1,
			qf >= max_d(L, q^2) */
			big f = max_d(q - 1, (big)( (L / q) - 1));

			/* f = f + W_k[f mod m].dist */
			f += d_wheel.dist[f % m];

			/* Remove the multiples of current prime */
			while ((q * f) <= R)
			{
				d_S[q * f] = false;
				f += d_wheel.dist[f % m];
			}
		}
	}
}

/*	TODO: Algorithm 4.1: Parallel Sieve Kernel version 2
	Remarks: Prime table S within [0, sqrt(n)] migrated to const memory
			 Wheel completely migrated to const memory
	Beware that const memory is only 64kB.
	Benchmark with the Profiler first before creating this!
*/
__global__ void parallelSieveKernel2(
	big n, big k, big m, Wheel_k d_wheel, big range, bool *d_S);

/*	TODO: Algorithm 4.1: Parallel Sieve Kernel version 3
	Remarks: Prime table S within [0, sqrt(n)] migrated to const memory
			 Wheel completely migrated to const memory
			 Probable use of the shared memory
			 Probable use of registers
	Beware that register is only 4B or 32b.
	Beware that const memory is only 64kB.
	Benchmark with the Profiler first before creating this!
*/
__global__ void parallelSieveKernel3(
	big n, big k, big m, Wheel_k d_wheel, big range, bool *d_S);

/*	MAIN
	To run this add the ff. args:
	1. N = the number up to which you're sieving
*/
int main(int argc, char **argv)
{
	big N = (big)strtoull(argv[1], NULL, 10);
	S = new bool[N]; //(bool*)malloc(N * sizeof(bool));

	printf("Find primes up to: %llu", N);

	cudaError_t x = algorithm4_1(N);
	if (x != cudaSuccess) {
		printf("Algorithm 4.1 failed to execute!");
		return 1;
	}

	// Display the primes.
	for (int i = 0; i < N; i++)
		if (S[i]) printf("%llu ", i);

    return 0;
}


// HOST FUNCTION DEFINITIONS-----------------------------

__host__ big gcd(big a, big b)
{
	big tmp;
   
	while (b != 0)
	{
		tmp = a;
		a = b;
		b = tmp%b;
	}
	return a;
}

big EratosthenesSieve(long double k, big n)
{
	big kthPrime = 0;

	// 0 and 1 are non-primes.
	S[0] = S[1] = false;
	for (big i = 2; i < n; i++)
		S[i] = true;

	// Simple Sieving Operation.
	for (big i = 2; i < (big)sqrtl(n); i++)
		if (S[i])
		{
			int j;
			for (j = i*i; j < n; j += i)
				S[j] = false;
		}

	// Find the k-th prime.
	for (big i = k; i > 2; i--)
		if (S[i]) kthPrime = i;
      
   return kthPrime;
}

cudaError_t algorithm4_1(big n)
{
	/* VARIABLES */
	big range;
	big sqrt_N = (big)sqrtl((long double)n);
	Wheel_k wheel;

	/* Allocation of wheel */
	wheel.rp = new bool[n];
	wheel.dist = new big[n];

	/* Find the first k primes
	   K = maximal s.t. S[K] <= (log N) / 4
	   Find primes up to sqrt(N) */
	big k = EratosthenesSieve(log10l((long double)n) / 4, n);

	/* Find the product of the first k primes m */
	big m = 1;
	for (big ii = 0; ii < k; ii++)
		if (S[ii]) m *= ii;

	/* Compute k-th wheel W_k
	   FUTURE OPTIMIZATION: Delegate kernel for computation */
	for (big x = 0; x < n; x++)
	{
		// True if rp[x] is relatively prime to m
		wheel.rp[x] = (gcd(x, m) == 1);

		/* This is d s.t. x + d is
		the smallest integer >dist[x]
		relatively prime to m */
		int d = 0;
		while (gcd(x + d, m) != 1)
			d++;

		wheel.dist[x] = d;
	}

	/* Delta = ceil(n/p) */
	range = (big)ceill(n / (long double)P);

	/* PARALLEL PART */
	cudaError_t parallelStatus = parallelSieve(n, k, m, wheel, range);
	if (parallelStatus != cudaSuccess) {
		fprintf(stderr, "parallelSieve() failed!");
	}

	/* FREE */
	delete[] wheel.rp;
	delete[] wheel.dist;

	return parallelStatus;
}

cudaError_t parallelSieve(
	big n, big k, big m, const Wheel_k &wheel, big range)
{
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* The Number Field S
	   will be migrated to GLOBAL memory
	   OPTIMIZATION: ranges will be migrated to SHARED memory
	   OPTIMIZATION: [0, sqrt(n)] will be migrated to CONSTANT memory
	*/
	bool * d_S = NULL;

	// The Wheel Precomputed Table
	// will be migrated to GLOBAL memory
	// OPTIMIZATION: may be migrated to CONSTANT memory as well
	Wheel_k d_wheel;
	d_wheel.rp = NULL;
	d_wheel.dist = NULL;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		return cudaStatus;
	}

	// Measure start time for CUDA portion
	cudaEventRecord(start, 0);

	// CUDA Memory Allocations.
	cudaStatus = cudaMalloc((void**)&d_S, n * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on number field S!\n");
		return cleanup(d_S, d_wheel, cudaStatus);
	}

	cudaStatus = cudaMalloc((void**)&(d_wheel.rp), n * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on wheel.rp!\n");
		return cleanup(d_S, d_wheel, cudaStatus);
	}

	cudaStatus = cudaMalloc((void**)&(d_wheel.dist), n * sizeof(big));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on wheel.dist!\n");
		return cleanup(d_S, d_wheel, cudaStatus);
	}

	//  cudaMemCpy -> Device
	cudaStatus = cudaMemcpy(d_S, S, n * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! S->d_S.\n");
		return cleanup(d_S, d_wheel, cudaStatus);
	}

	cudaStatus = cudaMemcpy(d_wheel.rp, wheel.rp, n * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! wheel.rp->d_wheel.rp\n");
		return cleanup(d_S, d_wheel, cudaStatus);
	}

	cudaStatus = cudaMemcpy(d_wheel.dist, wheel.dist, n * sizeof(big), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! wheel.dist->d_wheel.dist\n");
		return cleanup(d_S, d_wheel, cudaStatus);
	}

	// Kernel Call
	dim3 gridSize(ceill(ceill(sqrt(n))/256), 1, 1);
	dim3 blockSize(256, 1, 1);

	parallelSieveKernel<<<gridSize, blockSize>>>(n, k, m, wheel, range, d_S);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "parallelSieveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cleanup(d_S, d_wheel, cudaStatus);
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return cleanup(d_S, d_wheel, cudaStatus);
	}

	// cudaMemCpy -> Host
	cudaStatus = cudaMemcpy(S, d_S, n * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! d_S->S.\n");
		return cleanup(d_S, d_wheel, cudaStatus);
	}

	// Measure stop time for CUDA portion
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %0.5f ms\n", elapsedTime);

	// cudaFree
	return cleanup(d_S, d_wheel, cudaStatus);
}

cudaError_t cleanup(bool *d_S, Wheel_k &wheel, cudaError_t cudaStatus)
{
	cudaFree(d_S);
	cudaFree(wheel.rp);
	cudaFree(wheel.dist);
	return cudaStatus;
}
