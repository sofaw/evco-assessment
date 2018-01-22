import pickle

import matplotlib.pyplot as plt


def get_line(logbooks, colour, label, axis):
    numGens = len(logbooks[0])

    fit_max = [0] * numGens
    # Get the maximum at each generation over all runs
    for i in range(len(logbooks)):
        fit_max = map(max, zip(logbooks[i].select("max"), fit_max))

    gen = logbooks[0].select("gen")

    return axis.plot(gen, fit_max, colour, label=label)


def plot_max_from_logbooks(logbooks_path_list, colours_list, label_list):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Maximum fitness")

    lines = []

    for i in range(len(logbooks_path_list)):
        logbooks = pickle.load(open(logbooks_path_list[i], "rb"))
        lines += get_line(logbooks, colours_list[i], label_list[i], ax1)

    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="center right")

    plt.show()


def column(matrix, i):
    return [row[i] for row in matrix]


def gen_boxplot(logbooks, colour, plot):
    numRuns = len(logbooks)
    numGens = len(logbooks[0])

    # Get max at each generation for each run
    fit_max_by_run = [0] * numRuns
    for i in range(len(logbooks)):
        fit_max_by_run[i] = logbooks[i].select("max")

    fit_max_by_gen = [0] * numGens
    for i in range(numGens):
        fit_max_by_gen[i] = column(fit_max_by_run, i)

    boxplots = plot.boxplot(fit_max_by_gen, patch_artist=True)

    for patch in boxplots['boxes']:
        patch.set_facecolor(colour)


def plot_box_and_whisker_from_logbooks(logbooks_path_list, colours_list):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Maximum fitness")

    for i in range(len(logbooks_path_list)):
        logbooks = pickle.load(open(logbooks_path_list[i], "rb"))
        gen_boxplot(logbooks, colours_list[i], plt)

    plt.show()
