#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "NeuralNetwork.h"

/**
 * Helper function: Returns a random float between -1.0f and 1.0f
 */
float random_float() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

/**
 * Frees the memory allocated for the population.
 */
int free_population(int nbIndividual, struct NeuralNetwork **pointerIndividual) {
    for (int i = 0; i < nbIndividual; i++) {
        free(pointerIndividual[i]);
    }
    free(pointerIndividual);
    return 0;
}

/**
 * Initializes all weights and biases with random float values.
 */
int randomize_weights(int nbIndividual, struct NeuralNetwork **neural_network) {
    for (int i = 0; i < nbIndividual; i++) {

        // Randomize B1
        for (int j = 0; j < sizeof(neural_network[i]->B1) / sizeof(float); j++) {
            neural_network[i]->B1[j] = random_float();
        }

        // Randomize B2
        for (int j = 0; j < sizeof(neural_network[i]->B2) / sizeof(float); j++) {
            neural_network[i]->B2[j] = random_float();
        }

        // Randomize W1
        for (int j = 0; j < sizeof(neural_network[i]->W1) / sizeof(float); j++) {
            neural_network[i]->W1[j] = random_float();
        }

        // Randomize W2
        for (int j = 0; j < sizeof(neural_network[i]->W2) / sizeof(float); j++) {
            neural_network[i]->W2[j] = random_float();
        }
    }
    return 0;
}

// Make sure your forward_pass prototype is declared at the top of test_memory.c!
float forward_pass(const struct NeuralNetwork *nn, const float input[INPUT_SIZE]) {
    float hidden[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float value = nn->B1[j];

        for (int k = 0; k < INPUT_SIZE; k++) {
            value += input[k] * nn->W1[(j * INPUT_SIZE) + k];
        }

        // ReLU
        if (value < 0) {
            value = 0;
        }
        hidden[j] = value;
    }

    float final_val = nn->B2[0];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        final_val += hidden[i] * nn->W2[i];
    }

    return final_val;
}

int compare_fitness(const void *a, const void *b) {
    const struct NeuralNetwork *brainA = *(const struct NeuralNetwork **)a;
    const struct NeuralNetwork *brainB = *(const struct NeuralNetwork **)b;

    if (brainA->fitness > brainB->fitness) {
        return -1;
    }else {
        if (brainA->fitness < brainB->fitness) {
            return 1;
        }
        else {
            return 0;
        }
    }
}

void sort_population(int nbIndividual, struct NeuralNetwork **population) {
    qsort(population, nbIndividual, sizeof(struct NeuralNetwork *), compare_fitness);
}

/**
 * Evaluates each brain in the population and assigns a fitness score.
 */
int evaluate_population(int nbIndividual, struct NeuralNetwork **population) {

    // 1. Create fake data for the test (60 days of fake prices)
    float fake_inputs[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        fake_inputs[i] = 0.5f; // Just a fake normalized price
    }

    // 2. This is the fake "Real Price" of tomorrow that they need to guess
    float target_price = 0.8f;

    // 3. Loop through all individuals
    for (int i = 0; i < nbIndividual; i++) {

        // A. Get the prediction using forward_pass()
        float prediction = forward_pass(population[i],fake_inputs);

        // B. Calculate the error: absolute value of (prediction - target_price)
        float error = fabs(prediction - target_price);

        // C. Calculate and save the fitness in the struct
        population[i]->fitness = 1.0f / (error + 0.00001f);

    }

    return 0;
}

void mutate_array(float *array, int size , float mutation_rate) {
    for (int i = 0; i < size ; i++) {
        float chance = (float)rand() / (float)RAND_MAX;

        if (chance < mutation_rate) {
            array[i] += random_float() * 0.1f;
        }
    }
}

void breed_and_mutate(int nbIndividual, struct NeuralNetwork **population, float mutation_rate) {
    int elitecount = nbIndividual/10;

    for (int i = elitecount; i < nbIndividual ; i++) {
        int PIndex = rand()%elitecount;
        *population[i] = *population[PIndex];

        mutate_array(population[i]->B1,HIDDEN_SIZE,mutation_rate);
        mutate_array(population[i]->B2,OUTPUT_SIZE,mutation_rate);
        mutate_array(population[i]->W1,INPUT_SIZE * HIDDEN_SIZE,mutation_rate);
        mutate_array(population[i]->W2,OUTPUT_SIZE * HIDDEN_SIZE,mutation_rate);

    }
}

int main(void) {
    // 1. Setup random seed
    srand(time(NULL));

    int generations = 1000;

    const int nbIndividual = 10000;
    float mutation_rate = 0.05f;

    // 2. Allocate memory for the army
    struct NeuralNetwork **pointerIndividual = malloc(nbIndividual * sizeof(struct NeuralNetwork *));
    if (pointerIndividual == NULL) {
        printf("[ERROR] No memory left for the pointer array!\n");
        return 1;
    }

    for (int i = 0; i < nbIndividual; i++) {
        pointerIndividual[i] = malloc(sizeof(struct NeuralNetwork));
        if (pointerIndividual[i] == NULL) {
            printf("[ERROR] No memory left for instance of NeuralNetwork %d!\n", i);
            free_population(i, pointerIndividual);
            return 1;
        }
    }
    printf("[INFO] Population of %d networks successfully created.\n", nbIndividual);

    // 3. Initialize brains with random weights
    randomize_weights(nbIndividual, pointerIndividual);
    printf("[INFO] Brains initialized with random weights.\n");

    for (int i =0 ; i < generations ; i++) {
        evaluate_population(nbIndividual,pointerIndividual);
        sort_population(nbIndividual,pointerIndividual);
        printf("Generation %d | Best Fitness: %f\n", i, pointerIndividual[0]->fitness);
        breed_and_mutate(nbIndividual,pointerIndividual,mutation_rate);
    }

    printf("\n--- WORST BRAIN ---\n");
    printf("Rank %d - Fitness Score: %f\n\n", nbIndividual, pointerIndividual[nbIndividual - 1]->fitness);

    // 7. Clean up memory before exiting (ONLY ONCE AT THE END!)
    free_population(nbIndividual, pointerIndividual);
    printf("[INFO] Memory successfully freed. No leaks!\n");

    return 0;
}