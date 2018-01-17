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
    ax1.set_ylabel("Fitness")

    lines = []

    for i in range(len(logbooks_path_list)):
        logbooks = pickle.load(open(logbooks_path_list[i], "rb"))
        lines += get_line(logbooks, colours_list[i], label_list[i], ax1)

    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="center right")

    plt.show()


# Testing
#plot_max_from_logbooks(["dumps/logbook.p", "dumps/adf_logbook.p"], ["y", "b"], ["Without ADF", "ADF"])