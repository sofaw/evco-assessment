import argparse
import curses
import pickle
import random
from functools import partial

import numpy as np
from deap import base, creator, gp, tools

import logbooks_plotting as lp


def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
XSIZE, YSIZE = 14, 14
NFOOD = 1  # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)


# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN
    global XSIZE, YSIZE

    def __init__(self):
        self.direction = S_RIGHT
        self.body = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def _reset(self):
        self.direction = S_RIGHT
        self.body[:] = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def getAheadLocation(self):
        self.ahead = [self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1),
                      self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    ## You are free to define more sensing options to the snake
    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (YSIZE - 1) or self.body[0][1] == 0 or self.body[0][1] == (
                    XSIZE - 1): self.hit = True
        if self.body[0] in self.body[1:]: self.hit = True
        return (self.hit)

    def sense_wall_ahead(self):
        self.getAheadLocation()
        return (
            self.ahead[0] == 0 or self.ahead[0] == (YSIZE - 1) or self.ahead[1] == 0 or self.ahead[1] == (XSIZE - 1))

    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def sense_tail_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.body

    # Additional functions:
    def sense_food_direction_left(self):
        if len(self.food) == 0:
            return False
        return self.body[0][1] > self.food[0][1]

    def sense_food_direction_right(self):
        if len(self.food) == 0:
            return False
        return self.body[0][1] < self.food[0][1]

    def sense_food_direction_up(self):
        if len(self.food) == 0:
            return False
        return self.body[0][0] > self.food[0][0]

    def sense_food_direction_down(self):
        if len(self.food) == 0:
            return False
        return self.body[0][0] < self.food[0][0]

    def sense_tail_up(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == (self.body[i][0] + 1)) and (self.body[0][1] == self.body[i][1]):
                return True
        return False

    def sense_tail_right(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == self.body[i][0]) and (self.body[0][1] == self.body[i][1] - 1):
                return True
        return False

    def sense_tail_down(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == (self.body[i][0] - 1)) and (self.body[0][1] == self.body[i][1]):
                return True
        return False

    def sense_tail_left(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == self.body[i][0]) and (self.body[0][1] == self.body[i][1] + 1):
                return True
        return False

    def sense_wall_up(self):
        if self.body[0][0] == 1:
            return True
        return False

    def sense_wall_right(self):
        if self.body[0][1] == (XSIZE - 2):
            return True
        return False

    def sense_wall_down(self):
        if self.body[0][0] == (YSIZE - 2):
            return True
        return False

    def sense_wall_left(self):
        if self.body[0][1] == (1):
            return True
        return False

    def sense_danger_up(self):
        return (self.sense_wall_up() or self.sense_tail_up())

    def sense_danger_right(self):
        return (self.sense_wall_right() or self.sense_tail_right())

    def sense_danger_down(self):
        return (self.sense_wall_down() or self.sense_tail_down())

    def sense_danger_left(self):
        return (self.sense_wall_left() or self.sense_tail_left())

    def sense_tail_two_up(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == (self.body[i][0] + 2)) and (self.body[0][1] == self.body[i][1]):
                return True
        return False

    def sense_tail_two_right(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == self.body[i][0]) and (self.body[0][1] == self.body[i][1] - 2):
                return True
        return False

    def sense_tail_two_down(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == (self.body[i][0] - 2)) and (self.body[0][1] == self.body[i][1]):
                return True
        return False

    def sense_tail_two_left(self):
        for i in range(1, len(self.body)):
            if (self.body[0][0] == self.body[i][0]) and (self.body[0][1] == self.body[i][1] + 2):
                return True
        return False

    def sense_wall_two_up(self):
        if self.body[0][0] == 2:
            return True
        return False

    def sense_wall_two_right(self):
        if self.body[0][1] == (XSIZE - 3):
            return True
        return False

    def sense_wall_two_down(self):
        if self.body[0][0] == (YSIZE - 3):
            return True
        return False

    def sense_wall_two_left(self):
        if self.body[0][1] == 2:
            return True
        return False

    def sense_danger_two_up(self):
        return (self.sense_wall_two_up() or self.sense_tail_two_up())

    def sense_danger_two_right(self):
        return (self.sense_wall_two_right() or self.sense_tail_two_right())

    def sense_danger_two_down(self):
        return (self.sense_wall_two_down() or self.sense_tail_two_down())

    def sense_danger_two_left(self):
        return (self.sense_wall_two_left() or self.sense_tail_two_left())

    def sense_food_one_up(self):
        food = self.food[0]
        head = self.body[0]
        return (food[0] == head[0] - 1) and food[1] == head[1]

    def sense_food_one_right(self):
        food = self.food[0]
        head = self.body[0]
        return food[0] == head[0] and food[1] == (head[1] + 1)

    def sense_food_one_down(self):
        food = self.food[0]
        head = self.body[0]
        return (food[0] == head[0] + 1) and food[1] == head[1]

    def sense_food_one_left(self):
        food = self.food[0]
        head = self.body[0]
        return food[0] == head[0] and food[1] == (head[1] - 1)

    def if_food_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_direction_left, out1, out2)

    def if_food_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_direction_right, out1, out2)

    def if_food_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_direction_up, out1, out2)

    def if_food_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_direction_down, out1, out2)

    def if_danger_up(self, out1, out2):
        return partial(if_then_else, self.sense_danger_up, out1, out2)

    def if_danger_right(self, out1, out2):
        return partial(if_then_else, self.sense_danger_right, out1, out2)

    def if_danger_down(self, out1, out2):
        return partial(if_then_else, self.sense_danger_down, out1, out2)

    def if_danger_left(self, out1, out2):
        return partial(if_then_else, self.sense_danger_left, out1, out2)

    def if_danger_two_up(self, out1, out2):
        return partial(if_then_else, self.sense_danger_two_up, out1, out2)

    def if_danger_two_right(self, out1, out2):
        return partial(if_then_else, self.sense_danger_two_right, out1, out2)

    def if_danger_two_down(self, out1, out2):
        return partial(if_then_else, self.sense_danger_two_down, out1, out2)

    def if_danger_two_left(self, out1, out2):
        return partial(if_then_else, self.sense_danger_left, out1, out2)

    def if_food_one_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_one_up, out1, out2)

    def if_food_one_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_one_right, out1, out2)

    def if_food_one_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_one_down, out1, out2)

    def if_food_one_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_one_left, out1, out2)


# This function places a food item in the environment
def placeFood(snake):
    food = []
    while len(food) < NFOOD:
        potentialfood = [random.randint(1, (YSIZE - 2)), random.randint(1, (XSIZE - 2))]
        if not (potentialfood in snake.body) and not (potentialfood in food):
            food.append(potentialfood)
    snake.food = food  # let the snake know where the food is
    return (food)


snake = SnakePlayer()


# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def displayStrategyRun(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset)

    curses.initscr()
    win = curses.newwin(YSIZE, XSIZE, 0, 0)
    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)
    win.timeout(120)

    snake._reset()
    food = placeFood(snake)

    for f in food:
        win.addch(f[0], f[1], '@')

    timer = 0
    collided = False
    while not collided and not timer == ((2 * XSIZE) * YSIZE):

        # Set up the display
        win.border(0)
        win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
        win.getch()

        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        routine()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            for f in food: win.addch(f[0], f[1], ' ')
            food = placeFood(snake)
            for f in food: win.addch(f[0], f[1], '@')
            timer = 0
        else:
            last = snake.body.pop()
            win.addch(last[0], last[1], ' ')
            timer += 1  # timesteps since last eaten
        win.addch(snake.body[0][0], snake.body[0][1], 'o')

        collided = snake.snakeHasCollided()
        hitBounds = (timer == ((2 * XSIZE) * YSIZE))

    curses.endwin()

    # print collided
    # print hitBounds
    raw_input("Press to continue...")

    return snake.score


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset)

    snake._reset()
    food = placeFood(snake)
    timer = 0
    timeAlive = 0
    while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:

        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        routine()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            food = placeFood(snake)
            timer = 0
            timeAlive += 1
        else:
            snake.body.pop()
            timer += 1  # timesteps since last eaten
            timeAlive += 1

    return snake.score, timeAlive


def evalSnake(individual):
    totalScore = 0
    totalTimeAlive = 0
    numToAvg = 2
    for i in range(numToAvg):
        score, timeAlive = runGame(individual)
        totalScore += score
        totalTimeAlive += timeAlive
    return (totalScore / numToAvg), totalTimeAlive


# Parameters
numGens = 40
popSize = 500
CXPB = 0.9
MUTPB = 0.02

# GP primitives and terminals
pset = gp.PrimitiveSet("main", 0)
pset.addPrimitive(snake.if_food_up, 2)
pset.addPrimitive(snake.if_food_right, 2)
pset.addPrimitive(snake.if_food_down, 2)
pset.addPrimitive(snake.if_food_left, 2)
pset.addPrimitive(snake.if_danger_up, 2)
pset.addPrimitive(snake.if_danger_right, 2)
pset.addPrimitive(snake.if_danger_down, 2)
pset.addPrimitive(snake.if_danger_left, 2)
pset.addPrimitive(snake.if_danger_two_up, 2)
pset.addPrimitive(snake.if_danger_two_right, 2)
pset.addPrimitive(snake.if_danger_two_down, 2)
pset.addPrimitive(snake.if_danger_two_left, 2)
pset.addTerminal(snake.changeDirectionUp)
pset.addTerminal(snake.changeDirectionRight)
pset.addTerminal(snake.changeDirectionDown)
pset.addTerminal(snake.changeDirectionLeft)

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genGrow, pset=pset, min_=2, max_=8)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalSnake)
toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.3, fitness_first=True)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


# Performs a single run of the evolutionary algorithm given a randomSeed value
# Returns the population at the final generation and a list of fitnesses at each generation
def single_run(randomSeed):
    random.seed(randomSeed)

    pop = toolbox.population(n=popSize)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate the entire population
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    g = 1
    # Begin the evolution
    while g < numGens:
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = offspring
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        print(logbook.stream)

        g = g + 1

    return pop, stats, logbook, hof


def run_n_times(numRuns):
    pops = [None] * numRuns  # Contains the final population for each run of the algorithm
    stats = [None] * numRuns
    logbooks = [None] * numRuns
    hofs = [None] * numRuns
    for i in range(numRuns):
        print "run " + str(i)
        results = single_run(i)
        pops[i] = results[0]
        stats[i] = results[1]
        logbooks[i] = results[2]
        hofs[i] = results[3]
    return pops, stats, logbooks, hofs


def main():
    global snake
    global pset

    parser = argparse.ArgumentParser(
        description='Run evolutionary algorithm to play snake for specified number of runs.')
    parser.add_argument('--num_runs', type=int, nargs='?', default='1',
                        help='The number of runs of the evolutionary algorithm to perform')
    parser.add_argument('--plot_decision_graphs', type=int, nargs='?', default='0',
                        help='Plot the decision graphs for the top n individuals from the final population of each '
                             'run.')
    parser.add_argument('--display_strategy_runs', type=int, nargs='?', default='0',
                        help='Display the strategy for the top n individuals from the final population of each run.')
    parser.add_argument('--single_run_seed', type=int, nargs='?', default='-1',
                        help='Perform a single run with the given seed')
    parser.add_argument('--save_logbook', type=str, default=False,
                        help='Save the logbook to given path.')
    args = parser.parse_args()

    # Run the algorithm
    if (args.single_run_seed < 0):
        numRuns = args.num_runs
        print "Running ", numRuns, " time(s)..."
        pops, stats, logbooks, hofs = run_n_times(numRuns)

        # Save logbook
        if len(args.save_logbook) > 0:
            pickle.dump(logbooks, open(args.save_logbook, "wb"))

        # Print decision graphs
        numDecisionGraphs = args.plot_decision_graphs
        if numDecisionGraphs > 0:
            # For each run, print decision graphs for top 'numDecisionGraphs' individuals in final population
            for i in range(numRuns):
                top_n = tools.selBest(pops[i], numDecisionGraphs)
                for j in range(numDecisionGraphs):
                    filename = "decision/run_" + str(i) + "_num_" + str(j) + ".pdf"
                    lp.plot_decision_graph(top_n[j], filename)

        # Display strategies
        numStrategyRuns = args.display_strategy_runs
        if numStrategyRuns > 0:
            # For each run, display strategy for top 'numStrategyRuns' individuals in final population
            for i in range(numRuns):
                top_n = tools.selBest(pops[i], numStrategyRuns)
                for j in range(numStrategyRuns):
                    displayStrategyRun(top_n[j])

    else:
        pop, stats, logbook, hof = single_run(args.single_run_seed)

        # Print decision graphs
        numDecisionGraphs = args.plot_decision_graphs
        if numDecisionGraphs > 0:
            top_n = tools.selBest(pop, numDecisionGraphs)
            for j in range(numDecisionGraphs):
                filename = "single_decision/seed_" + str(args.single_run_seed) + "_" + str(j) + ".pdf"
                lp.plot_decision_graph(top_n[j], filename)

        numStrategyRuns = args.display_strategy_runs
        if numStrategyRuns > 0:
            top_n = tools.selBest(pop, numStrategyRuns)
            for j in range(numStrategyRuns):
                displayStrategyRun(top_n[j])


main()
