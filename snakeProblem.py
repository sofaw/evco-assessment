# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses

import scipy.stats

import helperFunctions as hf
import numpy as np
import pygraphviz as pgv
import random
import resultsPlotting as rp
import sys
import operator
from functools import partial
from deap import algorithms, base, creator, gp, tools

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
    def if_food_left(self, out1, out2):
        return partial(hf.if_then_else, self.sense_food_direction_left, out1, out2)
    def if_food_right(self, out1, out2):
        return partial(hf.if_then_else, self.sense_food_direction_right, out1, out2)
    def if_food_up(self, out1, out2):
        return partial(hf.if_then_else, self.sense_food_direction_up, out1, out2)
    def if_food_down(self, out1, out2):
        return partial(hf.if_then_else, self.sense_food_direction_down, out1, out2)

    def if_wall_ahead(self, out1, out2):
        return partial(hf.if_then_else, self.sense_wall_ahead, out1, out2)

    def if_food_ahead(self, out1, out2):
        return partial(hf.if_then_else, self.sense_food_ahead, out1, out2)

    def if_tail_ahead(self, out1, out2):
        return partial(hf.if_then_else, self.sense_food_ahead, out1, out2)


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

## MY CODE ##
# Constants
numRuns = 30
numGens = 40
popSize = 100
CXPB = 0.25
MUTPB = 0.25

# GP primitives and terminals
pset = gp.PrimitiveSet("main", 0)  # No external input to the procedure since decisions are based on sensing functions
pset.addPrimitive(snake.if_wall_ahead, 2)
pset.addPrimitive(snake.if_food_ahead, 2)
pset.addPrimitive(snake.if_tail_ahead, 2)
pset.addPrimitive(snake.if_food_up, 2)
pset.addPrimitive(snake.if_food_right, 2)
pset.addPrimitive(snake.if_food_down, 2)
pset.addPrimitive(snake.if_food_left, 2)
pset.addTerminal(snake.changeDirectionUp)  # Terminals are snake movements
pset.addTerminal(snake.changeDirectionRight)
pset.addTerminal(snake.changeDirectionDown)
pset.addTerminal(snake.changeDirectionLeft)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)  # Each leaf has the same depth between min and max
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

## MY CODE ##

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

    #print collided
    #print hitBounds
    raw_input("Press to continue...")

    return snake.score,


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset)

    #totalScore = 0

    snake._reset()
    food = placeFood(snake)
    timer = 0
    while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:

        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        routine()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            food = placeFood(snake)
            timer = 0
        else:
            snake.body.pop()
            timer += 1  # timesteps since last eaten

        #totalScore += snake.score

    #return totalScore,
    return snake.score,


## MY CODE ##

toolbox.register("evaluate", runGame)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def runEvolution(randomSeed):
    random.seed(randomSeed)

    pop = toolbox.population(n=popSize) # pop is list of individuals

    # Evaluate the entire population
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Variable containing statistics for each generation
    genFits = [None] * numGens

    # Begin the evolution
    while g < numGens:
        #print("-- Generation %i --" % g)

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

        pop[:] = offspring

        # Gather all the fitnesses in one list
        fits = [ind.fitness.values[0] for ind in pop]

        # Save fitnesses at each generation for plotting
        genFits[g] = fits

        # A new generation
        g = g + 1

    return pop, genFits

## MY CODE ##

def main():
    global snake
    global pset


    ## MY CODE ##
    if "multiple_runs" in sys.argv:
        runStats = [None] * numRuns
        finalPops = [None] * numRuns
        for i in range(numRuns):
            results = runEvolution(i)
            finalPops[i] = results[0]
            runStats[i] = results[1]
        rp.plot_best_box_and_whisker(runStats)

    else:
        results = runEvolution(10)
        pop = results[0]
        #for ind in pop:
        #    displayStrategyRun(ind)
        #genStats = results[1]

        #for s in genStats:
        #    print s.median

        # Plot the best tree
        #expr = tools.selBest(pop, 1)[0]
        for ind in pop:
            displayStrategyRun(ind)
        nodes, edges, labels = gp.graph(expr)

        g = pgv.AGraph(nodeSep=1.0)
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")
        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
        g.draw("snake_tree.pdf")


## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #


main()

