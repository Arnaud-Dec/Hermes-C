#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include "NeuralNetwork.h"
#include "ThreadArgs.h"

// ==========================================
// 1. OUTILS
// ==========================================

float random_float() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

int save_network(const struct NeuralNetwork *brain, const char *filename) {

    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("[ERROR] Impossible de creer le fichier de sauvegarde %s\n", filename);
        return -1;
    }

    size_t written = fwrite(brain, sizeof(struct NeuralNetwork), 1, file);

    fclose(file);

    if (written == 1) {
        printf("[SUCCESS] Cerveau sauvegarde avec succes dans : %s\n", filename);
        return 0;
    } else {
        printf("[ERROR] Probleme d'ecriture pendant la sauvegarde.\n");
        return -1;
    }
}

// ==========================================
// 2. DONNEES ET NORMALISATION
// ==========================================

int load_prices(const char *filename, float *prices_array, int max_size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("[ERROR] Impossible d'ouvrir le fichier %s\n", filename);
        return -1;
    }

    char line[256];
    int count = 0;

    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return 0;
    }

    while (fgets(line, sizeof(line), file) != NULL && count < max_size) {
        char *comma = strchr(line, ',');
        if (comma == NULL) continue;

        prices_array[count] = atof(comma + 1);
        count++;
    }

    fclose(file);
    return count;
}

void compute_returns(float *prices, int total_days) {
    for (int i = total_days - 1; i >= 1; i--) {
        if (prices[i-1] == 0.0f) {
            prices[i] = 0.0f;
        } else {
            prices[i] = ((prices[i] - prices[i-1]) / prices[i-1]) * 10.0f;
        }
    }
    prices[0] = 0.0f;
}

// ==========================================
// 3. LE CERVEAU (IA)
// ==========================================

int randomize_weights(int nbIndividual, struct NeuralNetwork *neural_network) {
    for (int i = 0; i < nbIndividual; i++) {
        for (size_t j = 0; j < sizeof(neural_network[i].B1) / sizeof(float); j++) neural_network[i].B1[j] = random_float();
        for (size_t j = 0; j < sizeof(neural_network[i].B2) / sizeof(float); j++) neural_network[i].B2[j] = random_float();
        for (size_t j = 0; j < sizeof(neural_network[i].W1) / sizeof(float); j++) neural_network[i].W1[j] = random_float();
        for (size_t j = 0; j < sizeof(neural_network[i].W2) / sizeof(float); j++) neural_network[i].W2[j] = random_float();
    }
    return 0;
}

float forward_pass(const struct NeuralNetwork *nn, const float input[INPUT_SIZE]) {
    float hidden[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float value = nn->B1[j];
        for (int k = 0; k < INPUT_SIZE; k++) {
            value += input[k] * nn->W1[(j * INPUT_SIZE) + k];
        }
        if (value < 0) value = 0;
        hidden[j] = value;
    }

    float final_val = nn->B2[0];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        final_val += hidden[i] * nn->W2[i];
    }
    return final_val;
}

// ==========================================
// 4. EVOLUTION ET SELECTION
// ==========================================

int compare_fitness(const void *a, const void *b) {
    const struct NeuralNetwork *brainA = (const struct NeuralNetwork *)a;
    const struct NeuralNetwork *brainB = (const struct NeuralNetwork *)b;

    if (brainA->fitness > brainB->fitness) return -1;
    if (brainA->fitness < brainB->fitness) return 1;
    return 0;
}

void sort_population(int nbIndividual, struct NeuralNetwork *population) {
    qsort(population, nbIndividual, sizeof(struct NeuralNetwork), compare_fitness);
}

void *evaluate_worker(void *arg) {
    struct ThreadArgs *my_args = (struct ThreadArgs *)arg;

    // my_args est un pointeur, on utilise ->
    int start = my_args->start_index;
    int end = my_args->end_index;
    struct NeuralNetwork *pop = my_args->pop;
    const float *prices = my_args->prices;
    int days = my_args->total_days;

    int days_tested = days - INPUT_SIZE;

    for (int i = start; i < end; i++) {
        float total_error = 0.0f;

        for (int day = INPUT_SIZE; day < days; day++) {
            float prediction = forward_pass(&pop[i], &prices[day - INPUT_SIZE]);
            float target = prices[day];
            total_error += fabs(prediction - target);
        }

        float avg_error = total_error / (float)days_tested;

        if (isnan(avg_error) || isinf(avg_error)) {
            pop[i].fitness = 0.0f;
        } else {
            pop[i].fitness = 1.0f / (avg_error + 0.00001f);
        }
    }

    return NULL;
}

int evaluate_population(int nbIndividual, struct NeuralNetwork *population, const float *historical_prices, int total_days ,int num_threads) {

    if (total_days <= INPUT_SIZE) {
        printf("[FATAL] Pas assez de donnees (%d jours) pour une IA qui demande %d jours.\n", total_days, INPUT_SIZE);
        return -1;
    }

    pthread_t threads[num_threads];
    struct ThreadArgs args[num_threads];

    int chunk_size = nbIndividual / num_threads;

    for (int i = 0; i < num_threads; i++) {
        args[i].pop = population;
        args[i].prices = historical_prices;
        args[i].total_days = total_days;
        args[i].start_index = i * chunk_size;

        if (i == num_threads - 1) {
            args[i].end_index = nbIndividual;
        } else {
            args[i].end_index = (i + 1) * chunk_size;
        }

        pthread_create(&threads[i], NULL, evaluate_worker, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
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

void breed_and_mutate(int nbIndividual, struct NeuralNetwork *population, float mutation_rate) {
    int elitecount = nbIndividual / 10;
    if (elitecount < 1) elitecount = 1;

    for (int i = elitecount; i < nbIndividual ; i++) {
        int PIndex = rand() % elitecount;
        population[i] = population[PIndex];

        mutate_array(population[i].B1, sizeof(population[i].B1)/sizeof(float), mutation_rate);
        mutate_array(population[i].B2, sizeof(population[i].B2)/sizeof(float), mutation_rate);
        mutate_array(population[i].W1, sizeof(population[i].W1)/sizeof(float), mutation_rate);
        mutate_array(population[i].W2, sizeof(population[i].W2)/sizeof(float), mutation_rate);
    }
}

// ==========================================
// 5. LE MOTEUR PRINCIPAL
// ==========================================

int main(void) {
    srand(time(NULL));

    // --- HYPERPARAMETRES ---
    int MAX_DAYS = 2000;
    int generations = 100;
    const int nbIndividual = 1000;
    float mutation_rate = 0.05f;
    int num_thread = 12;

    // --- LECTURE DES DONNEES ---
    float *historical_prices = malloc(MAX_DAYS * sizeof(float));
    int total_days = load_prices("data/data.csv", historical_prices, MAX_DAYS);

    if (total_days > 0) {
        printf("[SUCCESS] %d jours charges.\n", total_days);
        compute_returns(historical_prices, total_days);
        printf("[INFO] Donnees converties en variations financieres.\n");
    } else {
        printf("[FATAL] Echec donnees.\n");
        free(historical_prices);
        return 1;
    }

    // --- CREATION DE LA POPULATION ---
    struct NeuralNetwork *pointerIndividual = malloc(nbIndividual * sizeof(struct NeuralNetwork));
    if (pointerIndividual == NULL) {
        printf("[FATAL] Echec d'allocation memoire.\n");
        free(historical_prices);
        return 1;
    }
    printf("[INFO] %d Cerveaux crees en memoire.\n", nbIndividual);

    randomize_weights(nbIndividual, pointerIndividual);
    printf("[INFO] Poids aleatoires injectes. DEBUT DE L'ENTRAINEMENT...\n\n");

    // --- BOUCLE D'ENTRAINEMENT ---
    for (int i = 0 ; i < generations ; i++) {

        evaluate_population(nbIndividual, pointerIndividual, historical_prices, total_days , num_thread);
        sort_population(nbIndividual, pointerIndividual);

        printf("Generation %d | Best Fitness: %f\n", i, pointerIndividual[0].fitness);

        if (i < generations - 1) {
            breed_and_mutate(nbIndividual, pointerIndividual, mutation_rate);
        }
    }

    printf("\n--- RESULTAT FINAL ---\n");
    printf("LE MEILLEUR CERVEAU - Score: %f\n", pointerIndividual[0].fitness);
    printf("LE PIRE CERVEAU     - Score: %f\n\n", pointerIndividual[nbIndividual - 1].fitness);

    // --- NETTOYAGE ---
    free(pointerIndividual);
    free(historical_prices);
    printf("[INFO] Fin du programme. Memoire liberee.\n");

    return 0;
}