#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void initializeUsingCPU(int *c, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += i + 1;
    }
    printf("\n Sum of %d elememts is %d \n", n, sum);
}

void display(int *ar, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", ar[i]);
    }
}

float float_rand()
{
    float min = 1.0, max = 2.0;
    float scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

int main()
{
    unsigned int n = pow(10, 9);
    srand(time(NULL));

    // printf("%d ", n);

    float *a, *b, c;
    a = (float *)malloc(sizeof(float) * n);
    b = (float *)malloc(sizeof(float) * n);
    c = 0.0;

    printf("%f", float_rand());
    return 0;

    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            a[i * n + j] = float_rand();
            b[i * n + j] = float_rand();
        }
    }

    // int *ar = (int *)malloc(sizeof(int) * n);
    // initializeUsingCPU(ar, n);

    // display(ar, n);

    return 0;
}