
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include `pkg-confing --cflags opencv`
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 -lopencv_core -lopencv_imgproc -lopencv_highgui `pkg-confing --libs opencv`
EXE	        = device-query
OBJ	        = main.o

default: $(EXE)

main.o: main.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
