## Assignment 1

You can run all the programs using `nvcc`
Just type `nvcc <filename>.cu`
After that type in `./a.out <number of blocks> <number of threads>`

Only exception is normal matrix multiplication program where we decided to experiment with dimensions of blocks and grids. So, to multiple 8192 x 8192 matrices, we divided the number of blocks to be of 256 x 256 and threads to be of 32 x 32.

Total is your 8192 x 8192. Each thread computes something.

Over there, you can type in `./a.out` and you get your output ready.

Team Members:
Rounak Das - SE20UCSE149
Abhinav Chaudhry - SE20UCSE006
Rohith Gunelly - SE20UCSE244
Maneesh Kolli - SE20UCSE091
Shashank Mutyam - SE20UARI139
