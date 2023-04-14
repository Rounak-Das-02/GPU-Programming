NVCC=nvcc
NVCC_FLAGS=-g -G -Xcompiler -Wall

hello1: hello.cu
	$(NVCC) $(NVCC_FLAGS)  $< -o $@
clean:
	rm *.o hello1
