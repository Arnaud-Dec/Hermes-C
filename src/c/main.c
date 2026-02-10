#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "NeuralNetwork.h"

// --- Constants ---
#define MAX_DATA_ROWS 2000
#define MODEL_PATH "models/model.bin"
#define DATA_PATH "data/data.csv"

// --- Function Prototypes ---
float forward_pass(struct NeuralNetwork *nn, float input[INPUT_SIZE]);
int load_csv_data(const char *filename, float *output_array);

// --- Main Application ---
int main() {
    printf("[INFO] --- Start Hermes-C Engine ---\n");

    // 1. Load Model (Brain)
    struct NeuralNetwork nn;
    FILE *f = fopen(MODEL_PATH, "rb");
    
    if (f == NULL) { 
        printf("[ERROR] Could not open model file at %s\n", MODEL_PATH); 
        return 1; 
    }

    // Read Scaler Config
    fread(&nn.min_val, sizeof(float), 1, f);
    fread(&nn.max_val, sizeof(float), 1, f);
    
    // Read Weights and Biases
    fread(nn.W1, sizeof(float), INPUT_SIZE * HIDDEN_SIZE, f);
    fread(nn.B1, sizeof(float), HIDDEN_SIZE, f);
    fread(nn.W2, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fread(nn.B2, sizeof(float), OUTPUT_SIZE, f);
    
    fclose(f);

    printf("[INFO] Model loaded successfully.\n");
    printf("[INFO] Scaler Config -> Min: %.2f, Max: %.2f\n", nn.min_val, nn.max_val);

    // 2. Load Real Data (CSV)
    float prices[MAX_DATA_ROWS];
    int count = load_csv_data(DATA_PATH, prices);
    
    if (count == 0) return 1; // Error already printed in function

    printf("[INFO] Data loaded: %d days found.\n", count);

    if (count < INPUT_SIZE) {
        printf("[ERROR] Not enough data (needs %d, found %d).\n", INPUT_SIZE, count);
        return 1;
    }

    // 3. Prepare Input (Normalize last 60 days)
    float input[INPUT_SIZE];
    int start_index = count - INPUT_SIZE;

    // printf("[DEBUG] Input Data (Last %d days):\n", INPUT_SIZE);
    for (int i = 0; i < INPUT_SIZE; i++) {
        float raw_price = prices[start_index + i];
        
        // Normalization Formula: (Val - Min) / (Max - Min)
        float range = nn.max_val - nn.min_val;
        if (range == 0) range = 1.0f; // Prevent division by zero
        
        input[i] = (raw_price - nn.min_val) / range;
    }
    
    printf("[INFO] Last Raw Price: %.2f -> Normalized: %.4f\n", prices[count-1], input[INPUT_SIZE-1]);

    // 4. Inference (Prediction)
    float prediction_norm = forward_pass(&nn, input);

    // 5. De-normalization
    // Formula: Val * (Max - Min) + Min
    float prediction_price = prediction_norm * (nn.max_val - nn.min_val) + nn.min_val;

    printf("\n--------------------------------\n");
    printf("PREDICTED PRICE (Next Day): %.2f $\n", prediction_price);
    printf("--------------------------------\n");

    return 0;
}

// --- Function Implementations ---

/**
 * Performs the forward pass (inference) of the Neural Network.
 * Structure: Input (60) -> Linear -> ReLU -> Hidden (32) -> Linear -> Output (1)
 */
float forward_pass(struct NeuralNetwork *nn, float input[INPUT_SIZE]) {
    // 1. Hidden Layer
    float hidden[HIDDEN_SIZE];
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float val = nn->B1[j];
        
        for (int k = 0; k < INPUT_SIZE; k++) {
            // Indexing: W1 is flat, we jump INPUT_SIZE for each neuron
            val += input[k] * nn->W1[j * INPUT_SIZE + k];
        }
        
        // ReLU Activation Function
        if (val < 0) val = 0;
        
        hidden[j] = val;
    }

    // 2. Output Layer
    float final_val = nn->B2[0];
    
    for (int k = 0; k < HIDDEN_SIZE; k++) {
        final_val += hidden[k] * nn->W2[k];
    }

    return final_val;
}

/**
 * Loads the "Close" column from a CSV file into an array.
 * Assumes format: Date,Close
 */
int load_csv_data(const char *filename, float *output_array) {
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        printf("[ERROR] Could not open CSV file: %s\n", filename);
        return 0;
    }

    char line[1024];
    int count = 0;

    // Skip Header
    fgets(line, sizeof(line), f);
    
    while (fgets(line, sizeof(line), f) && count < MAX_DATA_ROWS) {
        // Parse CSV line
        char *token = strtok(line, ","); // First token (Date) - unused
        token = strtok(NULL, ",");       // Second token (Close)
        
        if (token != NULL) {
            output_array[count] = atof(token);
            count++;
        }
    }
    
    fclose(f);
    return count;
}