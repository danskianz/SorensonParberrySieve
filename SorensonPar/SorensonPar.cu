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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// C dependencies
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// C++ dependencies
#include <algorithm>

using std::min;
using std::max;

typedef unsigned long long big;

// Struct-of-Arrays Wheel
typedef struct Wheel_t
{
	bool * rp;	// Numbers relatively prime to m
	big * dist; // D s.t. x + d is the smallest integer >dist[x] relatively prime to m
} Wheel_k;

bool * S;	// Global shared bit array of numbers up to N
int P;		// Global number of processors

/*	Custom GCD implementation for C++ */
big gcd(big a, big b);

/*	EratosthenesSieve
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
	big sqrt_N = (big)ceill(sqrtl(n));

	// Express the sieve in thread mode.
	big i = threadIdx.x + blockIdx.x * blockDim.x;

	big L = range * i + 1;
	big R = std::min(range * (i + 1), n);

	/* Range Sieving */
	for (big x = L; x < R; x++)
		d_S[x] = d_wheel.rp[x % m];

	/* For every prime from prime[k] up to sqrt(N) */
	for (big q = k; q < sqrt_N; q++)
	{
		if (d_S[q])
		{
			/* Compute smallest f s.t.
			gcd(qf, m) == 1,
			qf >= max(L, q^2) */
			big f = std::max(q - 1, (big)ceill( (L / (long double)q) - 1));

			/* f = f + W_k[f mod m].dist */
			f += d_wheel.dist[f % m];

			/* Remove the multiples of current prime */
			while ((q * f) <= R)
			{
				S[q * f] = false;
				f += d_wheel.dist[f % m];
			}
		}
	}
}

/*	TODO: Algorithm 4.1: Parallel Sieve Kernel version 2
	Remarks: Prime table S within [0, sqrt(n)] migrated to const memory
			 Wheel completely migrated to const memory	
*/
__global__ void parallelSieveKernel2(
	big n, big k, big m, Wheel_k d_wheel, big range, bool *d_S);

/*	TODO: Algorithm 4.1: Parallel Sieve Kernel version 3
	Remarks: Prime table S within [0, sqrt(n)] migrated to const memory
			 Wheel completely migrated to const memory
			 Probable use of the shared memory
			 Probable use of registers
*/
__global__ void parallelSieveKernel3(
	big n, big k, big m, Wheel_k d_wheel, big range, bool *d_S);

/*	MAIN
	To run this add the ff. args:
	1. N = the number up to which you're sieving
*/
int main(int argc, char **argv)
{
	big N = (big)argv[1];
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

big gcd(big a, big b)
{
	int tmp;
	while (a != 0)
	{
		tmp = a;
		a = b % a;
		b = tmp;
	}
	return b;
}

big EratosthenesSieve(long double k, big n)
{
	big kthPrime;

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
		if (S[i]) return i;
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

	// The Number Field S
	// will be migrated to GLOBAL memory
	// OPTIMIZATION: ranges will be migrated to SHARED memory
	bool * d_S;

	// The list of primes up to sqrt(n)
	// OPTIMIZATION: will be migrated to CONSTANT memory

	// The Wheel Precomputed Table
	// will be migrated to GLOBAL memory
	// OPTIMIZATION: may be migrated to CONSTANT memory as well
	Wheel_k d_wheel;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Measure start time for CUDA portion
	cudaEventRecord(start, 0);

	// CUDA Memory Allocations.
	cudaStatus = cudaMalloc((void**)&d_S, n * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on number field S!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_wheel.rp, n * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on wheel.rp!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_wheel.dist, n * sizeof(big));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on wheel.dist!");
		goto Error;
	}

	//  cudaMemCpy -> Device
	cudaStatus = cudaMemcpy(d_S, S, n * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! S->d_S.");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_wheel.rp, wheel.rp, n * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! wheel.rp->d_wheel.rp");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_wheel.dist, wheel.dist, n * sizeof(big), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! wheel.dist->d_wheel.dist");
		goto Error;
	}

	// Kernel Call
	dim3 gridSize(ceill(ceill(sqrt(n))/256), 1, 1);
	dim3 blockSize(256, 1, 1);

	parallelSieveKernel<<<gridSize, blockSize>>>(n, k, m, wheel, range, d_S);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "parallelSieveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// cudaMemCpy -> Host
	cudaStatus = cudaMemcpy(S, d_S, n * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! d_S->S.");
		goto Error;
	}

	// Measure stop time for CUDA portion
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %0.5f ms\n", elapsedTime);

	// cudaFree
Error:
	cudaFree(d_S);
	cudaFree(d_wheel.rp);
	cudaFree(d_wheel.dist);

	return cudaStatus;
}