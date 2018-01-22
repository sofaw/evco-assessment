import pickle

from scipy.stats import mannwhitneyu


def column(matrix, i):
    return [row[i] for row in matrix]


def get_final_gen_max(logbook):
    numGens = len(logbook[0])
    numRuns = len(logbook)

    fit_max_by_run = [0] * numRuns
    for i in range(numRuns):
        fit_max_by_run[i] = logbook[i].select("max")

    return column(fit_max_by_run, numGens - 1)


def eval_mannwhitney(logbook_a_path, logbook_b_path):
    logbook_a = pickle.load(open(logbook_a_path, "rb"))
    logbook_b = pickle.load(open(logbook_b_path, "rb"))

    final_gen_max_a = get_final_gen_max(logbook_a)
    final_gen_max_b = get_final_gen_max(logbook_b)

    stat, pvalue = mannwhitneyu(final_gen_max_a, final_gen_max_b, alternative='greater')
    print stat
    print pvalue


def calc_ranksum(a, b):
    m = len(a)
    n = len(b)

    ordered = sorted(a + b)
    rank_map = {}  # Maps a value to its rank e.g. {3 : 1, 4 : 2, 5: 3.5}
    current_rank = 1
    i = 0
    while i < len(ordered):
        dups = 0
        curr = ordered[i]
        j = i + 1
        while j < len(ordered) and ordered[j] == curr:
            dups += 1
            j += 1

        i = j

        rank_map[curr] = current_rank + (0.5 * dups)
        current_rank += 1 + (1 * dups)

    ranksum_a = 0
    for i in range(len(a)):
        ranksum_a += rank_map[a[i]]
    ranksum_b = 0
    for i in range(len(b)):
        ranksum_b += rank_map[b[i]]

    return ranksum_a, ranksum_b


def effect_size(logbook_a_path, logbook_b_path):
    logbook_a = pickle.load(open(logbook_a_path, "rb"))
    logbook_b = pickle.load(open(logbook_b_path, "rb"))

    final_gen_max_a = get_final_gen_max(logbook_a)
    final_gen_max_b = get_final_gen_max(logbook_b)

    ranksum_a, ranksum_b = calc_ranksum(final_gen_max_a, final_gen_max_b)
    m = float(len(final_gen_max_a))
    n = float(len(final_gen_max_b))

    return float((ranksum_a / m - (m + 1) / 2.0) / n)
