# Graphics on GPU

Run test.cu to get an output image.

`nvcc test.cu -lglut -lGLU -lGL && ./a.out 32`

The outputs are also provided as screenshots.

First we have rendered a the face with random colors.

Next we have tried using dot product to get illumination.

All codes are parallelized using CUDA.
