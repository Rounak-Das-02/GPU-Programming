#include <stdio.h>
#include <sys/time.h>

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

double getCurrentTimestamp()
{
    struct timeval te;
    gettimeofday(&te, NULL);
    double milliseconds = te.tv_sec * 1000.0 + te.tv_usec / 1000.0;
    return milliseconds;
}

int main()
{
    int* result;
    cudaMallocManaged(&result, sizeof(int));

    for (int numBlocks = 1; numBlocks <= 80; numBlocks++)
    {
        for (int numThreads = 32; numThreads <= 1024; numThreads *= 2)
        {
            double startTime = getCurrentTimestamp();

            countChange<<<numBlocks, numThreads>>>(result);
            cudaDeviceSynchronize();

            double endTime = getCurrentTimestamp();
            double executionTime = endTime - startTime;

            printf("Blocks: %d, Threads: %d, Time: %.2f ms\n", numBlocks, numThreads, executionTime);
        }
    }

    cudaFree(result);

    return 0;
}
