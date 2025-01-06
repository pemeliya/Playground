
#include "common/mersenne.h"

namespace mersenne {

/* initializes mt[N] with a seed */
void init_genrand(uint32_t s)
{
    mt[0] = s & 0xffffffff;
    for (mti = 1; mti < static_cast<int>(N); mti++)
    {
        mt[mti] = (1812433253 * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);

        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for mtiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */

        mt[mti] &= 0xffffffff;
        /* for >32 bit machines */
    }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(uint32_t init_key[], int key_length)
{
    int i, j, k;
    init_genrand(19650218);
    i = 1;
    j = 0;
    k = (static_cast<int>(N) > key_length
	 ? static_cast<int>(N)
	 : key_length);
    for (; k; k--)
    {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525))
            + init_key[j] + j;  /* non linear */
        mt[i] &= 0xffffffff;    /* for WORDSIZE > 32 machines */
        i++;
        j++;
        if (i >= static_cast<int>(N))
        {
            mt[0] = mt[N - 1];
            i = 1;
        }
        if (j >= key_length) j = 0;
    }
    for (k = N - 1; k; k--)
    {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i; /* non linear */
        mt[i] &= 0xffffffff; /* for WORDSIZE > 32 machines */
        i++;
        if (i >= static_cast<int>(N))
        {
            mt[0] = mt[N - 1];
            i = 1;
        }
    }

    mt[0] = 0x80000000; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0,0xffffffff]-interval */
uint32_t genrand_int32(void)
{
    uint32_t y;
    static uint32_t mag01[2] = { 0x0, MATRIX_A };

    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= static_cast<int>(N))
    { /* generate N words at one time */
        int kk;

        if (mti == N + 1) /* if init_genrand() has not been called, */
        init_genrand(5489); /* a defat initial seed is used */

        for (kk = 0; kk < static_cast<int>(N - M); kk++)
        {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (; kk < static_cast<int>(N - 1); kk++)
        {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

        mti = 0;
    }

    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
}

} // namespace mersenne 
