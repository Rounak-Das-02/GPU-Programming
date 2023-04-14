## Assignment 2

Problem 1 -

We took the code of vector dot product and modified it to work for generation of the desired product. It was pretty easy and we just had to modify in two areas. Mainly instead of multiplication of two vectors, we multiplied the (googol - i) and then inside the cache index, we multiplied the elements again.

For 80 blocks and 1024 threads, it took 12.447508 seconds.
For 40 blocks and 1024 threads, it took 12.212572 seconds.
For 20 blocks and 1024 threads, it took 12.062388 seconds.
For 1 block and 1024 threads, it took 15.131390 seconds.

For 1 block and 512 threads, it took 27.901493 seconds.

The code is run in `Nvidia MX330` on an `Intel i5 processor`

Code is contributed by `Rounak` and `Abhinav`

Problem 2 -

The code is pretty much the same as what was taught in the class. Separate blocks are created and their particular prefix sums are found. After that, for calculating final prefix values, the value (the last value of the array in a block) from the previous block is added to all the elements in the current block to get final prefix sum.

Code is contributed by `Maneesh` and `Rohit` .

Problem 3 -

The code uses dynamic programming to compute the number of denominations. For creation of the table, GPU cores are used.

Code is contributed by `Shashank`

Team Members:
Rounak Das - SE20UCSE149
Abhinav Chaudhry - SE20UCSE006
Rohith Gunelly - SE20UCSE244
Maneesh Kolli - SE20UCSE091
Shashank Mutyam - SE20UARI139
