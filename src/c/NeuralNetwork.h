#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

// --- Hyperparameters ---
#define INPUT_SIZE 60
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 1

// --- Data Structure ---
// Represents the "Brain" of the AI + Scaling configuration
struct NeuralNetwork {
    float min_val;                      // Scaling Min (from training)
    float max_val;                      // Scaling Max (from training)
    float W1[INPUT_SIZE * HIDDEN_SIZE]; // Weights Layer 1
    float B1[HIDDEN_SIZE];              // Bias Layer 1
    float W2[OUTPUT_SIZE * HIDDEN_SIZE];// Weights Layer 2 (Output)
    float B2[OUTPUT_SIZE];              // Bias Layer 2 (Output)
};

#endif