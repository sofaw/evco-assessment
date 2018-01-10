import matplotlib.pyplot as plt
import numpy as np

gameMax = 185

# runStats = [run1=[fitsGen1, ... fitsGenM], run2=[], ..., runN[]] nRun*nGen array
#                   [0][0]    ... [0][M]

# Plot mean fitness at each generation over multiple runs for a single algorithm
# pointFunction is for calculating the y value at each generation e.g. mean
# errorFunction is for calculating the error values for each point e.g. std dev
def plot_single_algorithm(runStats, pointFunction, errorFunction):
    numGen = len(runStats[0])
    #plt.axis(xmin=0, xmax=numGen, ymin=0, ymax=gameMax)

    genStats = preprocess_list(runStats)
    y = [None] * numGen
    error = [None] * numGen
    for g in range(numGen):
        y[g] = pointFunction(genStats[g])
        error[g] = errorFunction(genStats[g])

    x = np.arange(0, numGen, 1)
    plt.errorbar(x=x, y=y, yerr=error, capsize=10)
    plt.show()

def plot_single_algorithm_iqr_error(runStats, pointFunction):
    numGen = len(runStats[0])
    #plt.axis(xmin=0, xmax=numGen, ymin=0, ymax=gameMax)

    genStats = preprocess_list(runStats)
    y = [None] * numGen
    error = [None] * numGen
    for g in range(numGen):
        y[g] = pointFunction(genStats[g])
        # Calculate upper and lower quartiles
        error[g] = [np.percentile(genStats[g], 25), np.percentile(genStats[g], 75)]

    x = np.arange(0, numGen, 1)
    plt.errorbar(x=x, y=y, yerr=np.array(error).T, capsize=10)
    plt.show()

def plot_box_and_whisker(runStats):
    genStats = preprocess_list(runStats)
    plt.boxplot(genStats)
    plt.show()

# Plot box and whisker for best sample from each population at each generation
def plot_best_box_and_whisker(runStats):
    numGens = len(runStats[0])
    numRuns = len(runStats)

    best = [None] * numRuns
    for r in range(numRuns):
        bestGens = [None] * numGens
        for g in range(numGens):
            bestGens[g] = max(runStats[r][g])
        best[r] = bestGens
    plt.boxplot(np.array(best))  # So boxplot uses columns
    plt.show()


def plot_best(runStats):
    genStats = preprocess_list(runStats)
    best = select_best(genStats)
    x = np.arange(1, len(genStats) + 1, 1)
    plt.plot(x, best, marker='o')
    plt.show()


def select_best(genStats):
    best = [None] * len(genStats)
    for i in range(len(genStats)):
        best[i] = max(genStats[i])
    return best


def flatten_list(list):
    flat_list = []
    for sublist in list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

# Index fitnesses by generation and flatten
def preprocess_list(list):
    numGen = len(list[0])
    nonFlatGenStats = zip(*list)
    flatGenStats = [None] * numGen
    for g in range(numGen):
        flatGenStats[g] = flatten_list(nonFlatGenStats[g])
    return flatGenStats

mydata = [[[1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2]], [[3, 3], [3, 3], [3, 3], [3, 3]], [[4, 4], [4, 4], [4, 4]]]
newdata = [[[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]]]
moredata = [[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6]], [[2, 2, 2], [8, 8, 8]]] # 2 runs, 2 generations, popSize=3

plot_best(moredata)
