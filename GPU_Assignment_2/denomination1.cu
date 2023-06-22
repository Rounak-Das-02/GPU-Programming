#include <stdio.h>

#define NUM_DENOMINATIONS 9
#define TARGET_AMOUNT 10000

__global__ void countChange(int* result)
{
    int denominations[NUM_DENOMINATIONS] = {1, 2, 5, 10, 20, 50, 100, 500, 2000};
    int dp[TARGET_AMOUNT + 1] = {0};
    dp[0] = 1;

    for (int i = 0; i < NUM_DENOMINATIONS; i++)
    {
        for (int j = denominations[i]; j <= TARGET_AMOUNT; j++)
        {
            dp[j] += dp[j - denominations[i]];
        }
    }

    *result = dp[TARGET_AMOUNT];
}

int main()
{
    int* result;
    cudaMallocManaged(&result, sizeof(int));

    countChange<<<1, 1>>>(result);
    cudaDeviceSynchronize();

    printf("Total number of distinct ways: %d\n", *result);

    cudaFree(result);

    return 0;
}

