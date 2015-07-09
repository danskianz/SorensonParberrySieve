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
	big * pos;	// pos[x] = # of numbers relatively prime to m up to x
	long long * inv; // x-th number relatively prime to m
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

/*	EulerPhi
	Computes the Euler Totient/Phi Function
*/
big EulerPhi(big n);

/*	Algorithm 4.1 Sequential Portion
Running Time: O(sqrt(n))
Space: O(sqrt(n)) up to O(sqrt(n)/log log n)
*/
void algorithm4_1(big n);

/*	Algorithm 4.1 Helper: Parallel Sieve
	All CUDA-related functionality goes here.
*/
cudaError_t parallelSieve(
	big n, big k, const Wheel_k &wheel, big range, const big *&tinyPrimes);

/*	Algorithm 4.1: Parallel Sieve Kernel
	Parallelization: O(sqrt(n)) processors
	Space: O(sqrt(n)) up to O(sqrt(n)/log log n)
	PRAM Mode: Exclusive Read, Exclusive Write (EREW)
*/
__global__ void parallelSieveKernel(
	big n, big k, Wheel_k wheel, big range, big *d_tinyPrimes, bool *d_S)
{
	// TODO: Express the sieve in thread mode.
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

big EulerPhi(big n)
{
	big i = 0;
	big j;

	for (j = 1; j <= n; j++)
		if (gcd(n, j) == 1) i++;

	return i;
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
	wheel.pos = new big[n];
	wheel.inv = new long long[n];

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

		/* If gcd(x, m) == 1,
		pos[x] = # of numbers relatively prime to m
		up to x */
		wheel.pos[x] = 0;
		if (gcd(x, m) == 1)
			for (d = 0; d <= x; d++)
				if (wheel.rp[d]) wheel.pos[x]++;

		/* If (x == 0), inv[x] = -1
		Else if (x < phi(m)), inv[x] = rp[x]
		Else inv[x] = 0
		*/
		if (x == 0) wheel.inv[x] = 1;
		else if (x < EulerPhi(m))
			for (d = 0; d <= x; d++)
				if (wheel.rp[d]) wheel.inv[x] = d;
		else wheel.inv[x] = 0;
	}

	/* TODO: Find primes up to sqrt(N) */
	big * tinyPrimes;

	/* Delta = ceil(n/p) */
	range = (big)ceill(n / (long double)P);

	/* PARALLEL PART */
	

	/* SEQUENTIAL CLEANUP */
	for (big i = 2; i < k; i++)
		S[tinyPrimes[i]] = true;

	/* FREE */
	delete[] wheel.rp;
	delete[] wheel.dist;
	delete[] wheel.pos;
	delete[] wheel.inv;
}

cudaError_t parallelSieve(
	big n, big k, const Wheel_k &wheel, big range, const big *&tinyPrimes)
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
	cudaStatus = cudaMalloc((void**)&d_wheel.pos, n * sizeof(big));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on wheel.pos!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_wheel.inv, n * sizeof(long long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on wheel!");
		goto Error;
	}

	// TODO: cudaMemCpy -> Device
	// TODO: Kernel Call
	parallelSieveKernel(n, k, wheel, range, d_tinyPrimes, d_S);
	// TODO: cudaMemCpy -> Host

	// TODO: SECONDARY - measure stop time

	// TODO: cudaFree
Error:
	cudaFree(d_S);
	cudaFree(d_tinyPrimes);
	cudaFree(d_wheel.rp);
	cudaFree(d_wheel.dist);
	cudaFree(d_wheel.pos);
	cudaFree(d_wheel.inv);

	return cudaStatus;
}