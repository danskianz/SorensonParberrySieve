/* SorensonSeq.c
   Sequential Implementation of Algorithm 4.1
   as discussed in Sorenson and Parberry's
   1994 paper "Two Fast Parallel Prime Number
   Sieves".

   Authors:
   Daniel Anzaldo
   David Frank
   Antonio Lanfranchi
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

typedef unsigned long long big;

bool * S;	// Global shared bit array of numbers up to N
int P;		// Global number of processors


/*	Algorithm 4.1 Fully Sequential Version
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

/*	MAIN
	To run this add the ff. args:
	1. N = the number up to which you're sieving
	2. P = the number of threads you'll be using
*/
int main(int argc, char *argv[])
{
	int i;
	big N = argv[1];	
	P = argv[2];
	S = (big)malloc(N * sizeof(big));

	printf("Find primes up to: %llu", N);

	algorithm4_1(N);

	// Display the primes.
	for (i = 0; i < N; i++)
	{
		if (S[i]) printf("%llu ", i);
	}

	return 0;
}

big EratosthenesSieve(long double k)
{
	big kthPrime;
	int i;

	// 0 and 1 are non-primes.
	S[0] = S[1] = false;
	for (i = 2; i < k; i++)
	{
		S[i] = true;
	}

	// Simple Sieving Operation.
	for (i = 2; i < sqrtl(k); i++)
	{
		if (S[i])
		{
			int j;
			for (j = i*i; j < k; j += i)
			{
				S[j] = false;
			}
		}
	}

	// Find the k-th prime.
	for (i = 2; i < k; i++)
	{
		if (S[i]) kthPrime = i;
	}

	return kthPrime;
}

void algorithm4_1(big n)
{
	big range;
	int p_id;
	big sqrt_N = (big)sqrtl((long double) n);
	big ii;

	/* TODO: Find the first k primes*/
	// K = maximal s.t. S[K] <= (log N) / 4
	big k = EratosthenesSieve(log10l((long double) n) / 4);

	/* Find the product of the first k primes m */
	big m = 1;
	for (ii = 0; ii < k; ii++)
	{
		if (S[ii]) m *= ii;
	}

	/* TODO: Compute k-th wheel W[k] */
	bool * Wheel_k_rp;
	big * Wheel_k_dist;

	/* TODO: Find primes up to sqrt(N) */
	big * tinyPrimes;

	/* Delta = ceil(n/p) */
	range = (big)ceill(n / (long double) P);

	/* PARALLEL PART */
	for (p_id = 0; p_id < P; p_id++)
	{
		big L = range * p_id + 1;
		big R = min(range * (p_id + 1), n);
		big x, i;

		/* Range Sieving */
		for (x = L; x < R; x++)
		{
			S[x] = Wheel_k_rp[x % m];
		}

		/* For every prime from prime[k] up to sqrt(N)*/
		for (i = k; i < (sizeof(tinyPrimes) / sizeof(big)); i++)
		{
			/* Compute smallest f s.t.
			gcd(qf, m) == 1,
			qf >= max(L, q^2) */
			big f = max(tinyPrimes[i] - 1, ceill(L / (long double)(tinyPrimes[i] - 1)));

			/* f = f + W_k[f mod m].dist */
			f += Wheel_k_dist[f % m];

			/* Remove the multiples of current prime */
			while ((tinyPrimes[i]*f) <= R)
			{
				S[tinyPrimes[i] * f] = false;
				f += Wheel_k_dist[f % m];
			}
		}
	}

	/* SEQUENTIAL CLEANUP */
	for (ii = 0; ii < k; ii++)
	{
		S[tinyPrimes[ii]] = true;
	}
	//S[1] = false;
}