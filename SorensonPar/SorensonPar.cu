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
void algorithm4_1(big n);

/*	Algorithm 4.1 Helper: Parallel Sieve
	All CUDA-related functionality goes here.
*/
cudaError_t parallelSieve(
	big n, big k, big m, const Wheel_k &wheel, big range, big *&tinyPrimes);

/*	Algorithm 4.1: Parallel Sieve Kernel
	Parallelization: O(sqrt(n)) processors
	Space: O(sqrt(n)) up to O(sqrt(n)/log log n)
	PRAM Mode: Exclusive Read, Exclusive Write (EREW)
*/
__global__ void parallelSieveKernel(
	big n, big k, big m, Wheel_k d_wheel, big range, big *d_tinyPrimes, bool *d_S)
{
	// TODO: Express the sieve in thread mode.
	big i = threadIdx.x + blockIdx.x * blockDim.x;

	big L = range * i + 1;
	big R = std::min(range * (i + 1), n);

	/* Range Sieving */
	for (big x = L; x < R; x++)
		d_S[x] = d_wheel.rp[x % m];

	/* For every prime from prime[k] up to sqrt(N) */
	for (big q = k; q < (big)ceill(sqrt(n)); q++)
	{
		if (d_S[q])
		{
			/* Compute smallest f s.t.
			gcd(qf, m) == 1,
			qf >= max(L, q^2) */
			big f = std::max(d_tinyPrimes[q] - 1, (big)ceill((L / (long double)(d_tinyPrimes[q] - 1))));

			/* f = f + W_k[f mod m].dist */
			f += d_wheel.dist[f % m];

			/* Remove the multiples of current prime */
			while ((d_tinyPrimes[q] * f) <= R)
			{
				S[d_tinyPrimes[q] * f] = false;
				f += d_wheel.dist[f % m];
			}
		}
	}
}

/*	MAIN
	To run this add the ff. args:
	1. N = the number up to which you're sieving
	2. P = the number of threads you'll be using
*/
int main(int argc, char **argv)
{
	big N = (big)argv[1];
	P = (int)argv[2];
	S = new bool[N]; //(bool*)malloc(N * sizeof(bool));

	printf("Find primes up to: %llu", N);

	algorithm4_1(N);

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

big EratosthenesSieve(long double k)
{
	big kthPrime;

	// 0 and 1 are non-primes.
	S[0] = S[1] = false;
	for (int i = 2; i < k; i++)
		S[i] = true;

	// Simple Sieving Operation.
	for (int i = 2; i < sqrtl(k); i++)
		if (S[i])
		{
			int j;
			for (j = i*i; j < k; j += i)
				S[j] = false;
		}

	// Find the k-th prime.
	for (int i = 2; i < k; i++)
		if (S[i]) kthPrime = i;

	return kthPrime;
}

void algorithm4_1(big n)
{
	/* VARIABLES */
	big range;
	big sqrt_N = (big)sqrtl((long double)n);
	Wheel_k wheel;

	/* Allocation of wheel */
	wheel.rp = new bool[n];
	wheel.dist = new big[n];

	/* Find the first k primes
	   K = maximal s.t. S[K] <= (log N) / 4 */
	big k = EratosthenesSieve(log10l((long double)n) / 4);

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

	/* TODO: Find primes up to sqrt(N) */
	big * tinyPrimes;

	/* Delta = ceil(n/p) */
	range = (big)ceill(n / (long double)P);

	/* PARALLEL PART */
	cudaError_t parallelStatus = parallelSieve(n, k, wheel, range, tinyPrimes);
	if (parallelStatus != cudaSuccess) {
		fprintf(stderr, "parallelSieve() failed!");
		exit(EXIT_FAILURE);
	}

	/* SEQUENTIAL CLEANUP */
	for (big i = 2; i < k; i++)
		S[tinyPrimes[i]] = true;

	/* FREE */
	delete[] wheel.rp;
	delete[] wheel.dist;
}

cudaError_t parallelSieve(
	big n, big k, big m, const Wheel_k &wheel, big range, const big *&tinyPrimes)
{
	cudaError_t cudaStatus;

	// The Number Field S
	// will be migrated to GLOBAL memory
	// OPTIMIZATION: ranges will be migrated to SHARED memory
	bool * d_S;

	// The list of primes up to sqrt(n)
	// OPTIMIZATION: will be migrated to CONSTANT memory
	big * d_tinyPrimes;

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

	// TODO: SECONDARY - measure start time

	// CUDA Memory Allocations.
	cudaStatus = cudaMalloc((void**)&d_S, n * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on number field S!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_tinyPrimes, n * sizeof(big));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on tinyPrimes!");
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

	// TODO: cudaMemCpy -> Device

	// TODO: Kernel Call
	dim3 gridSize(ceill(ceill(sqrt(n))/256), 1, 1);
	dim3 blockSize(256, 1, 1);

	parallelSieveKernel<<<gridSize, blockSize>>>(n, k, m, wheel, range, d_tinyPrimes, d_S);

	// TODO: cudaMemCpy -> Host

	// TODO: SECONDARY - measure stop time

	// TODO: cudaFree
Error:
	cudaFree(d_S);
	cudaFree(d_tinyPrimes);
	cudaFree(d_wheel.rp);
	cudaFree(d_wheel.dist);

	return cudaStatus;
}