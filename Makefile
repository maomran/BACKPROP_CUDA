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

task1: 
	echo "Running with Shared Memory"
	mkdir -p build
	nvcc -G -g -std=c++11 tensor.cu fclayer.cu sigmoidlayer.cu sgd.cpp funobj.cu model.cu mnist.cpp main.cu -o build/run   
	./build/run

task2:
	echo "Running without Shared Memory"
	mkdir -p build
	nvcc -G -g -std=c++11 tensorshared.cu fclayer.cu sigmoidlayer.cu sgd.cpp funobj.cu model.cu mnist.cpp main.cu -o build/run   
	./build/run

clean:
	rm -rf build
