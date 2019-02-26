
CPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cpp')
GPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cu')

dataset:
	mkdir -p data
	curl -o data/train-images.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
	curl -o data/train-labels.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz 
	curl -o data/test-images.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	curl -o data/test-labels.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	gunzip data/train-images.gz
	gunzip data/train-labels.gz
	gunzip data/test-images.gz
	gunzip data/test-labels.gz

build: 
	mkdir -p build
	nvcc ${CPU_SOURCE_FILES} ${GPU_SOURCE_FILES} -lineinfo -o build/run

run:
	./build/run

# run_experiments:
# 	mkdir -p ${LOGS_DIR}
# 	python3 run_experiments.py

clean:
	rm -rf build
