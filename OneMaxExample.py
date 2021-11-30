###################################################################
"""
 @author: EmanueleMusumeci (https://github.com/EmanueleMusumeci) #
 
 Example of a OneMax optimization problem, consisting in evolutionarily 
 synthesizing a string of bits all set to 1 using the number of correct 
 bits as a fitness function, solved using a genetic algorithm

"""
##################################################################

from modules.optimization import GeneticBinaryOptimizer
from modules.crossover import RandomSplit
from modules.mutation import RandomBitFlip
from modules.population_initialization import BinaryPopulationInitializer
from modules.visualization import Print
from modules.early_stopping import ImprovementHistoryWithPatience
from modules.fitness_evaluation import OneMax

#SEED = 1234567890
#random.seed(SEED)

# Directly influences the difficulty of the problem
STRING_SIZE = 100

# Influences execution time but also the higher the population size 
# the better are the chances of reaching a global optimum
POPULATION_SIZE = 400

# Execution time (NOTICE: if the problem is too difficult, the early_stopping_criterion 
# will stop the training after seeing no improvements for 10 turns)
N_ITERATIONS = 300

#Genetic algorithm hyperparameters:
# Probability for each bit of each population individual to randomly flip during the mutation phase 
MUTATION_PROBABILITY = 1.0/STRING_SIZE

# After forming pairs of winners of the previous iterations (the "parents" of the next generation)
# this probability determines the chance that the children are generated by crossover (while instead
# being identical to the parents)
CROSSOVER_PROBABILITY = 0.9

# Pool size of candidates for the tournament selection step
TOURNAMENT_SELECTION_CANDIDATES = 3

genetic_learner = GeneticBinaryOptimizer(
    BinaryPopulationInitializer(POPULATION_SIZE, STRING_SIZE),
    OneMax(),
    RandomSplit(CROSSOVER_PROBABILITY),
    RandomBitFlip(MUTATION_PROBABILITY),
    early_stopping_criterion=ImprovementHistoryWithPatience(10),
    result_visualization_method=Print(),
    n_parent_candidates=min(TOURNAMENT_SELECTION_CANDIDATES, POPULATION_SIZE-1),
)

#Starts learning (will automaticall show improvements if there are any)
genetic_learner.learn(N_ITERATIONS)