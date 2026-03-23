#ifndef HERMES_C_THREADARGS_H
#define HERMES_C_THREADARGS_H

// Structure pour passer les arguments à nos threads
struct ThreadArgs {
    int start_index;             // Le premier cerveau à évaluer (ex: 0)
    int end_index;               // Le dernier cerveau à évaluer (ex: 125)
    struct NeuralNetwork **pop;  // Le pointeur vers la population globale
    const float *prices;         // L'historique des prix
    int total_days;              // Le nombre total de jours
};

#endif //HERMES_C_THREADARGS_H