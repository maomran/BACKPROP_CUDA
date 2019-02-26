#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "fclayer.h"
#include "relylayer.h"
#include "sgd.h"
#include "funobj.h"
#include "model.h"
#include "mnist.h"
#include "csv_logger.h"
#include "utils.h"


int main() {
    // Always initialize seed to some random value
    srand(static_cast<unsigned>(time(0)));

    // Prepare events for measuring time on CUDA
    float elapsedTime = 0.0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Print our current configuration for this training
    // Configuration::printCurrentConfiguration();
    // Configuration::printCUDAConfiguration();

    // Read both training and test dataset
    MNISTDataSet* trainDataset = new MNISTDataSet(TRAIN);
    MNISTDataSet* testDataset = new MNISTDataSet(TEST);

    // Prepare optimizer and loss function
    float learningRate = LR;
    SGD* optimizer = new SGD(learningRate);
    CrossEntropyLoss* loss = new CrossEntropyLoss();

    // Prepare model
    Model* model = new Model(optimizer, loss);
    model->addLayer(new DenseLayer(28*28, 100));
    model->addLayer(new ReluLayer(100));
    model->addLayer(new DenseLayer(100, 10));

    // Prepare logger that will help us gather timings from experiments
    CSVLogger* logger = new CSVLogger(LOG_FILE_NAME);

    // Run some epochs
    int epochs = EPOCHS;
    int batchSize = BATCHSIZE;
    int numberOfTrainBatches = trainDataset->getSize() / batchSize;
    int numberOfTestBatches = testDataset->getSize() / batchSize;
    for (int e = 0; e < epochs; e++) {
        float trainingLoss = 0.0, trainingAccuracy = 0.0;
        double trainingForwardTime = 0.0, trainingBackwardTime = 0.0;
        printf("Epoch %d:\n", e);
        for (int batch = 0; batch < numberOfTrainBatches; batch++) {
            // Fetch batch from dataset
            Tensor2D* images = trainDataset->getBatchOfImages(batch, batchSize);
            Tensor2D* labels = trainDataset->getBatchOfLabels(batch, batchSize);

            // Forward pass
            cudaEventRecord(start, 0);
            Tensor2D* output = model->forward(images);
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);

            // Save statistics
            trainingLoss += loss->getLoss(output, labels);
            trainingAccuracy += loss->getAccuracy(output, labels);
            cudaEventElapsedTime(&elapsedTime, start, end);
            trainingForwardTime += elapsedTime;

            // Backward pass
            cudaEventRecord(start, 0);
            model->backward(output, labels);
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);

            // Save statistics
            cudaEventElapsedTime(&elapsedTime, start, end);
            trainingBackwardTime += elapsedTime;

            // Clean data for this batch
            delete images;
            delete labels;
        }

        // Calculate mean training metrics
        trainingLoss /= numberOfTrainBatches;
        trainingAccuracy /= numberOfTrainBatches;
        printf("  - [Train] Loss=%.5f\n", trainingLoss);
        printf("  - [Train] Accuracy=%.5f%%\n", trainingAccuracy);
        printf("  - [Train] Total Forward Time=%.5fms\n", trainingForwardTime);
        printf("  - [Train] Total Backward Time=%.5fms\n", trainingBackwardTime);
        printf("  - [Train] Batch Forward Time=%.5fms\n", trainingForwardTime / numberOfTrainBatches);
        printf("  - [Train] Batch Backward Time=%.5fms\n", trainingBackwardTime / numberOfTrainBatches);

        // Check model performance on test set
        float testLoss = 0.0, testAccuracy = 0.0;
        for (int batch = 0; batch < numberOfTestBatches; batch++) {
            // Fetch batch from dataset
            Tensor2D* images = testDataset->getBatchOfImages(batch, batchSize);
            Tensor2D* labels = testDataset->getBatchOfLabels(batch, batchSize);

            // Forward pass
            Tensor2D* output = model->forward(images);

            // Print error
            testLoss += loss->getLoss(output, labels);
            testAccuracy += loss->getAccuracy(output, labels);

            // Clean data for this batch
            delete images;
            delete labels;
        }

        // Calculate mean testing metrics
        testLoss /= numberOfTestBatches;
        testAccuracy /= numberOfTestBatches;
        printf("  - [Test] Loss=%.5f\n", testLoss);
        printf("  - [Test] Accuracy=%.5f%%\n", testAccuracy);
        printf("\n");

        // Save times to the logger
        logger->logEpoch(trainingLoss, trainingAccuracy,
                         testLoss, testAccuracy,
                         trainingForwardTime, trainingBackwardTime,
                         trainingForwardTime / numberOfTrainBatches,
                         trainingBackwardTime / numberOfTrainBatches);

        // Shuffle both datasets before next e!
        trainDataset->shuffle();
        testDataset->shuffle();
    }
    delete logger;
    return 0;
}
