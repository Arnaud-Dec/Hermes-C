#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "NeuralNetwork.h"


// Fonction qui prend le Cerveau (nn) et les Yeux (input) et renvoie une Décision (float)
float prediction(struct NeuralNetwork *nn, float input[INPUT_SIZE]) {
    
    // 1. Couche Cachée
    float hidden[HIDDEN_SIZE];
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float val = nn->B1[j]; // Note la flèche '->' car nn est un pointeur !
        
        for (int k = 0; k < INPUT_SIZE; k++) {
            val += input[k] * nn->W1[j * INPUT_SIZE + k];
        }
        
        // ReLU
        if (val < 0) val = 0;
        
        hidden[j] = val;
    }

    // 2. Couche de Sortie
    float final_val = nn->B2[0];
    
    for (int k = 0; k < HIDDEN_SIZE; k++) {
        final_val += hidden[k] * nn->W2[k];
    }

    return final_val;
}


int load_csv(const char *filename, float *output_array) {

    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        printf("Erreur: Impossible d'ouvrir le CSV %s\n", filename);
        return 0;
    }
    char line[1024];
    int count = 0;

    fgets(line, sizeof(line), f);
    
    while (fgets(line, sizeof(line), f)){
        char * token = strtok(line,",");
        token = strtok(NULL, ",");
        if (token != NULL){
            output_array[count] = atof(token);
            count++;
        }
    }
    fclose(f);
    return count;
}


#define MAX_DATA 2000 

int main(){
    printf("--- Start Hermes-C ---\n");

    // 1. Chargement du Cerveau 🧠
    struct NeuralNetwork nn;
    FILE *f = fopen("../../models/model.bin", "rb");
    if (f == NULL) { printf("Erreur Model !\n"); return 1; }

    fread(&nn.min_val, sizeof(float) , 1, f); // Lecture Min
    fread(&nn.max_val, sizeof(float) , 1, f); // Lecture Max
    
    fread(nn.W1, sizeof(float), INPUT_SIZE * HIDDEN_SIZE, f);
    fread(nn.B1, sizeof(float), HIDDEN_SIZE, f);
    fread(nn.W2, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fread(nn.B2, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);

    printf("Modele charge.\n");
    printf("Config Scaler -> Min: %.2f, Max: %.2f\n", nn.min_val, nn.max_val);

    // 2. Chargement des Prix Réels (CSV) 📈
    float prices[MAX_DATA];
    int count = load_csv("../../data/data.csv", prices);
    printf("Donnees chargees: %d jours.\n", count);

    if (count < INPUT_SIZE) {
        printf("Pas assez de donnees !\n");
        return 1;
    }

    // 3. Préparation et Normalisation (Le Maillon Manquant) 🔗
    float input[INPUT_SIZE];
    int start_index = count - INPUT_SIZE; // On commence 60 jours avant la fin

    printf("--- Input Data (Last 60 days) ---\n");
    for (int i = 0; i < INPUT_SIZE; i++) {
        float raw_price = prices[start_index + i];
        
        // LA FORMULE MAGIQUE : (Val - Min) / (Max - Min)
        // On évite la division par zéro par sécurité
        float range = nn.max_val - nn.min_val;
        float normalized_price = (raw_price - nn.min_val) / range;

        input[i] = normalized_price;
    }
    
    // Juste pour vérifier, on affiche la dernière valeur normalisée (doit être entre 0 et 1)
    printf("Dernier prix brut : %.2f -> Normalise : %.4f\n", prices[count-1], input[INPUT_SIZE-1]);

    // 4. Prédiction 🔮
    float prediction_normalisee = prediction(&nn, input);

    // 5. Dénormalisation (Pour avoir un prix en Dollars) 💵
    // Formule inverse : Val * (Max - Min) + Min
    float prediction_prix = prediction_normalisee * (nn.max_val - nn.min_val) + nn.min_val;

    printf("\n--------------------------------\n");
    printf("PREDICTION DU PRIX DEMAIN : %.2f $\n", prediction_prix);
    printf("--------------------------------\n");

    return 0;
}



