// TASK 1

#include <stdio.h>
#include <stdlib.h>


int main (int argc, char *argv[]) {
	int i, j, N, T, iterations;

    printf("Enter the number of points in each dimension, that is the N value for N x N:\n");
	scanf("%d", &N);

	printf("Enter the maximum number of iterations:\n");
	scanf("%d", &T);


	double g[N][N];
    double h[N][N];

	for (i = 0; i < N; i++ ) {	//initialize array
		for (j = 0; j < N; j++) {
			h[i][j] = 0;
			g[i][j] = 0;
		}
	}

	//initialize all walls to temperature of 20C
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			h[0][i] = 20.0;
			h[i][0] = 20.0;
			h[N - 1][i] = 20.0;
			h[i][N - 1] = 20.0;
		}
	}

	//define fireplace area
	double fire_start, fire_end;
	fire_start = 0.3 * N;
	fire_end = 0.7 * N;

	//declare temperature of fireplace
	for (i = fire_start; i < fire_end; i++) {
		h[0][i] = 100.0;
	}

	printf("\n");
	printf("Initial Temperatures: \n");
	for (i = 0; i < N; i+=N/10) {
		for (j = 0; j < N; j+=N/10) {
			printf("%-.2f\t", h[i][j]);
		}
		printf("\n");
	}


	for (iterations = 0; iterations < T; iterations++) {
		for (i = 1; i < N - 1; i++) {
			for (j = 1; j < N - 1; j++) {
				g[i][j] = 0.25 * (h[i - 1][j] + h[i + 1][j] + h[i][j - 1] + h[i][j + 1]);
			}
		}
        for (i = 1; i < N - 1; i++) {
			for (j = 1; j < N - 1; j++) {
				h[i][j] = g[i][j];
			}
		}
	}



	//finally print the final temperatures after calculations
	printf("\nFinal Temperatures: \n");
	for (i = 0; i < N; i += N / 10) {
		for (j = 0; j < N; j += N / 10) {
			printf("%-.2f\t ", h[i][j]);
		}
		printf("\n");
	}
}


