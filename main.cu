#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "fclayer.h"
#include "sigmoidlayer.h"
#include "sgd.h"
#include "funobj.h"
#include "model.h"
#include "mnist.h"

int main() {

    // Prepare events for measuring time on CUDA
    float elapsedTime = 0.0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Read both training and test dataset
    MNISTDataSet* trainDataset = new MNISTDataSet(true);
    MNISTDataSet* testDataset = new MNISTDataSet(false);

    // Prepare optimizer and loss function
    float learningRate = 1e-03;
    SGD* optimizer = new SGD(learningRate);
    LossFunction* loss = new LossFunction();

    // Prepare model
    Model* model = new Model(optimizer, loss);
    model->addLayer(new FCLayer(28*28, 100));
    model->addLayer(new SigmoidLayer(100));
    model->addLayer(new FCLayer(100, 10));


    int epochs = 100;
    int batchSize = 100;
    int numberOfTrainBatches = trainDataset->getSize() / batchSize;
    int numberOfTestBatches = testDataset->getSize() / batchSize;
    float  trainingTime = 0.0;
    for (int e = 0; e < epochs; e++) {
        float trainingAccuracy = 0.0;
        for (int batch = 0; batch < numberOfTrainBatches; batch++) {
            // Fetch batch from dataset
            tensor* images = trainDataset->getBatchOfImages(batch, batchSize);
            tensor* labels = trainDataset->getBatchOfLabels(batch, batchSize);

            // Forward pass
            cudaEventRecord(start, 0);
            tensor* output = model->forward(images);
            model->backward(output, labels);
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsedTime, start, end);
            trainingTime += elapsedTime;
            trainingAccuracy += loss->TrainingAccuracy(output, labels);



            // Clean data for this batch
            delete images;
            delete labels;
        }

        // Calculate mean training metrics
        trainingAccuracy /= numberOfTrainBatches;
        if (e % 10 == 0){
            printf("Epoch %d:\n", e);
            printf("  Train Accuracy=%.5f\n", trainingAccuracy);
        }
        trainDataset->shuffle();
    }
    printf("Average Training Batch Time=%.5fms\n", trainingTime / numberOfTrainBatches/epochs);
    float testAccuracy = 0.0;
    float testForwardTime = 0.0;

    for (int batch = 0; batch < numberOfTestBatches; batch++) {
        // Fetch batch from dataset
        tensor* images = testDataset->getBatchOfImages(batch, batchSize);
        tensor* labels = testDataset->getBatchOfLabels(batch, batchSize);

        // Forward pass
        cudaEventRecord(start, 0);
        tensor* output = model->forward(images);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsedTime, start, end);
        testForwardTime += elapsedTime;

        // Print error
        testAccuracy += loss->TestAccuracy(output, labels);
        testAccuracy /= numberOfTestBatches;
        if (batch % 10 == 0){
            printf("Batch %d:\n", batch);
            printf("  Test Accuracy=%.5f\n", testAccuracy);
            printf("\n");
        }
        delete images;
        delete labels;
        testDataset->shuffle();
    } 
    printf("Average Test Batch Time=%.5fms\n", testForwardTime / numberOfTestBatches);

    return 0;
}
