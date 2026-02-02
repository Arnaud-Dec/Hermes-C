#define INPUT_SIZE 60
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 1

struct NeuralNetwork
{
    float min_val;
    float max_val;
    float W1[INPUT_SIZE * HIDDEN_SIZE];
    float B1[HIDDEN_SIZE];
    float W2[OUTPUT_SIZE * HIDDEN_SIZE];
    float B2[OUTPUT_SIZE];
};

