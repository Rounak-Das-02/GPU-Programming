// __device__ void multiplyArrays(int result[], int num1[], int size1, int num2[], int size2) {
//     int i, j;

//     // Initialize the result array with zeros
//     memset(result, 0, sizeof(int) * MAX_DIGITS);

//     // Multiply each digit of num2 with num1 and accumulate the results in the result array
//     for (i = 0; i < size2; i++) {
//         int carry = 0;

//         for (j = 0; j < size1; j++) {
//             int product = num2[size2 - 1 - i] * num1[size1 - 1 - j] + carry + result[MAX_DIGITS - 1 - i - j];

//             result[MAX_DIGITS - 1 - i - j] = product % 10;
//             carry = product / 10;
//         }

//         result[MAX_DIGITS - 1 - i - j] += carry;
//     }
// }