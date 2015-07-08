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

// Visual Studio Dependencies
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

bool * S;	// Global shared bit array of numbers up to N
int P;		// Global number of processors


/*	Algorithm 4.1 Sequential Portion
	Running Time: O(sqrt(n))
	Space: O(sqrt(n)) up to O(sqrt(n)/log log n)
*/
void algorithm4_1(big n);

/*	EratosthenesSieve
The most basic form of generating primes.
Used to help find the first k primes.
Returns the k-th prime.
*/
big EratosthenesSieve(long double x);

/*	Algorithm 4.1 Helper: Parallel Sieve
	All CUDA-related functionality goes here.
*/
cudaError_t parallelSieve(big n, big k, bool * W_k_rp, big * W_k_dist);

/*	Algorithm 4.1: Parallel Sieve Kernel
	Parallelization: O(sqrt(n)) processors
	Space: O(sqrt(n)) up to O(sqrt(n)/log log n)
	PRAM Mode: Exclusive Read, Exclusive Write (EREW)
*/
__global__ void parallelSieveKernel(big n, big k)
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
	big range;
	int p_id;
	big sqrt_N = (big)sqrtl((long double)n);

	/* TODO: Find the first k primes*/
	// K = maximal s.t. S[K] <= (log N) / 4
	big k = EratosthenesSieve(log10l((long double)n) / 4);

	/* Find the product of the first k primes m */
	big m = 1;
	for (big ii = 0; ii < k; ii++)
		if (S[ii]) m *= ii;

	/* TODO: Compute k-th wheel W[k] */
	bool * Wheel_k_rp;
	big * Wheel_k_dist;

	/* TODO: Find primes up to sqrt(N) */
	big * tinyPrimes;

	/* Delta = ceil(n/p) */
	range = (big)ceill(n / (long double)P);

	/* PARALLEL PART */
	

	/* SEQUENTIAL CLEANUP */
	for (big ii = 2; ii < k; ii++)
		S[tinyPrimes[ii]] = true;
}

cudaError_t parallelSieve(
	big n,
	big k,
	bool * W_k_rp,
	big * W_k_dist)
{
	cudaError_t status;

	// TODO: Work on the parallel portion. Migrations and all.

	return status;
}