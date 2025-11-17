import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# =======================
# Neural Network
# =======================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, X, weights):
        # Decode genome (weights)
        ih_size = self.input_size * self.hidden_size
        ho_size = self.hidden_size * self.output_size

        w_input_hidden = weights[:ih_size].reshape(self.input_size, self.hidden_size)
        w_hidden_output = weights[ih_size:ih_size+ho_size].reshape(self.hidden_size, self.output_size)

        hidden = np.tanh(np.dot(X, w_input_hidden))
        output = self.softmax(np.dot(hidden, w_hidden_output))
        return output

    def softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

# =======================
# Genetic Algorithm
# =======================
class GeneticAlgorithm:
    def __init__(self, nn, pop_size=50, mutation_rate=0.05):
        self.nn = nn
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        self.num_weights = (nn.input_size*nn.hidden_size) + (nn.hidden_size*nn.output_size)
        self.population = np.random.uniform(-1, 1, (pop_size, self.num_weights))
        self.history_best = []  # best accuracy per generation
        self.history_avg = []   # average accuracy per generation

    def fitness(self, X, y):
        scores = []
        for individual in self.population:
            preds = self.nn.forward(X, individual)
            acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))
            scores.append(acc)
        return np.array(scores)

    def select(self, fitness_scores):
        probs = fitness_scores / fitness_scores.sum()
        idx = np.random.choice(np.arange(self.pop_size), size=2, p=probs)
        return self.population[idx[0]], self.population[idx[1]]

    def crossover(self, parent1, parent2):
        point = np.random.randint(1, self.num_weights-1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2

    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.5)
        return individual

    def evolve(self, X, y, generations=20):
        for gen in range(generations):
            fitness_scores = self.fitness(X, y)
            new_population = []

            # Keep elite
            elite_idx = np.argmax(fitness_scores)
            new_population.append(self.population[elite_idx])

            while len(new_population) < self.pop_size:
                p1, p2 = self.select(fitness_scores)
                c1, c2 = self.crossover(p1, p2)
                c1, c2 = self.mutate(c1), self.mutate(c2)
                new_population.extend([c1, c2])

            self.population = np.array(new_population[:self.pop_size])
            best_score = np.max(fitness_scores)
            avg_score = np.mean(fitness_scores)
            self.history_best.append(best_score)
            self.history_avg.append(avg_score)
            print(f"Generation {gen+1}: Best Accuracy = {best_score:.4f}, Avg Accuracy = {avg_score:.4f}")

        return self.population[np.argmax(self.fitness(X, y))]

    def plot_progress(self):
        plt.plot(self.history_best, marker='o', label="Best Accuracy")
        plt.plot(self.history_avg, marker='x', linestyle='--', label="Average Accuracy")
        plt.title("GA Training Progress")
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

# =======================
# Run GA on MNIST subset
# =======================
if __name__ == "__main__":
    print("Loading MNIST (this may take a minute)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = (X / 255.0).astype(np.float32)
    y = y.astype(int)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.to_numpy().reshape(-1, 1))



    # Train/test split (small subset for demo)
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, train_size=2000, test_size=500, stratify=y, random_state=42)

    nn = NeuralNetwork(input_size=784, hidden_size=64, output_size=10)
    ga = GeneticAlgorithm(nn, pop_size=30, mutation_rate=0.1)
    best_weights = ga.evolve(X_train, y_train, generations=10)

    # Evaluate on training set
    preds_train = nn.forward(X_train, best_weights)
    acc_train = np.mean(np.argmax(preds_train, axis=1) == np.argmax(y_train, axis=1))
    print("Final Training Accuracy:", acc_train)

    # Evaluate on unseen test set
    preds_test = nn.forward(X_test, best_weights)
    acc_test = np.mean(np.argmax(preds_test, axis=1) == np.argmax(y_test, axis=1))
    print("Final Test Accuracy:", acc_test)

    # Plot GA progress (best vs avg accuracy)
    ga.plot_progress()
