###################################################################
"""
 @author: EmanueleMusumeci (https://github.com/EmanueleMusumeci) 
 
 Example of an Unsupervised clustering problem of data in a high-dimensional space, 
 where the semantics of the data fields are unknown as well as the ground truth
 classification, therefore the importance of each field towards the final clustering
 is unknown as well. 
 
 Here the Genetic Algorithm is applied to determine which data fields are important towards
 the quality of the final clustering result.

 The Genetic Algorithm is used to optimize a binary string with a length equal to the 
 dimensionality of the dataset samples and the purpose of the optimization process is
 to select which subset of these fields (the "dimensions") gives the best clustering.

 In this case the FITNESS function is hand-crafted and varies from case to case: in our example
 the fitness function to be optimized is the separation of the centroids.

 Also the CLUSTERING method is chosen according to the problem at hand, in this case we can choose
 between KMeans, GMM and Spectral (non-linear) clustering.

 Results are not necessarily optimal but allow a thourough exploration of the solution space for
 this kind of unsupervised learning setting. 

"""
##################################################################

import os

from modules.clustering import GMM_clusterer, KMeans_clusterer, Spectral_clusterer
from modules.crossover import RandomSplit
from modules.early_stopping import ImprovementHistoryWithPatience
from modules.fitness_evaluation import ClusterCentroidDistance, ClusterMinimumPointDistance, ClusterAccuracy
from modules.mutation import RandomBitFlip
from modules.population_initialization import BinaryPopulationInitializer
from modules.visualization import Cluster3DVisualizer, Cluster2DVisualizer, Cluster2D3DVisualizer

from modules.dataset import Dataset
from modules.optimization import GeneticBinaryOptimizer

if __name__=="__main__":
    BASE_DIR = os.path.dirname(__file__)
    print("Base directory: "+BASE_DIR)

    SNAPSHOT_DIR = "snapshots"
    print("Snapshots directory: "+SNAPSHOT_DIR)
    
    DATASET_DIR = "data"
    print("Dataset directory: "+DATASET_DIR)

############################################
###          Dataset generation          ###
############################################

    #DATASET_FILENAME = os.path.join(DATASET_DIR,"GeneratedBlobDataset.csv")
    DATASET_FILENAME = None
    if DATASET_FILENAME is not None:
        #If a dataset file name is specified, load it and return an instance of that dataset
        dataset, ground_truth_cluster_labels, \
        N_SAMPLES, N_DIMENSIONS, N_CLUSTERS = Dataset.load_blob_dataset(DATASET_FILENAME, show = False)
    else:
        #Otherwise generate a dataset using some custom parameters as N_CLUSTERS gaussian distributions of data points
        
        #Total number of samples in the dataset
        N_SAMPLES = 1000 
        #Dimensionality of each dataset sample
        N_DIMENSIONS = 250    
        #Number of clusters in the dataset
        N_CLUSTERS = 3
        #"Hypercube" (of dimensionality N_DIMENSIONS) where all cluster centroids should be contained
        CLUSTERS_CENTERS_RANGE = (-1.5,1.5)
        #Variance of each "blob" of data points
        CLUSTERS_STD_DEV = 6
        
        dataset, ground_truth_cluster_labels = Dataset.generate_blob_dataset(N_SAMPLES, N_DIMENSIONS, N_CLUSTERS, 
            save_to = os.path.join(DATASET_DIR,"GeneratedBlobDataset.csv"), 
            clusters_centers_range=CLUSTERS_CENTERS_RANGE, std_dev=CLUSTERS_STD_DEV)


############################################
###   Genetic algorithm hyperparameters  ###
############################################

    POPULATION_SIZE = 500
    N_ITERATIONS = 500
    CROSSOVER_PROBABILITY = 0.9
    #MUTATION_PROBABILITY = 1.0/N_DIMENSIONS
    MUTATION_PROBABILITY = 0.8
    TOURNAMENT_SELECTION_CANDIDATES = 20

    EARLY_STOPPING_PATIENCE = 10
    
    #FITNESS_EVALUATION_METHOD = "CentroidDistance"
    #FITNESS_EVALUATION_METHOD = "MinimumPointDistance"
    FITNESS_EVALUATION_METHOD = "Accuracy"
    
    CLUSTERING_METHOD = "KMEANS"
    #CLUSTERING_METHOD = "GMM"
    #CLUSTERING_METHOD = "SPECTRAL"

    NORMALIZE = True

#-----------------#
#   Directories   #
#-----------------#    

    MODEL_NAME = CLUSTERING_METHOD + "_" + str(N_CLUSTERS) + "_clusters" + ("_"+FITNESS_EVALUATION_METHOD) + ("_normalized" if NORMALIZE else "")
    print("\nModel name (in snapshots directory): "+MODEL_NAME)

    SAVE_CHECKPOINTS_TO_DIR = os.path.join(SNAPSHOT_DIR, MODEL_NAME)
    print("\nCheckpoints will be saved in: "+MODEL_NAME)

    SAVE_IMAGES_TO_DIR = os.path.join(SNAPSHOT_DIR, MODEL_NAME, "images")
    print("\nResult images will be saved in: "+SAVE_IMAGES_TO_DIR)


#---------------#
#   Clusterer   #
#---------------#
    
    
    if CLUSTERING_METHOD == "KMEANS":
        clusterer = KMeans_clusterer(
            dataset, 
            N_CLUSTERS,
            normalize_data=NORMALIZE,
            use_random_seed=True
        )
    elif CLUSTERING_METHOD == "GMM":
        clusterer = GMM_clusterer(
                dataset, 
                N_CLUSTERS,
                normalize_data=NORMALIZE,
                use_random_seed=True
            )
    elif CLUSTERING_METHOD == "SPECTRAL":
        clusterer = Spectral_clusterer(
                dataset, 
                N_CLUSTERS,
                normalize_data=NORMALIZE
            )


#---------------#
#   Visualizer  #
#---------------#

    #Sets the range of values represented on the axes of the 2D and 3D dataset scatter plots
    SCALE_RANGE = (0.5, 0.5, 0.5) 
        
    #VISUALIZE = "2D"
    #VISUALIZE = "3D"
    VISUALIZE = "2D3D"

    if VISUALIZE == "2D":
        visualizer_method = Cluster2DVisualizer(
            clusterer, 
            show=False, 
            save_snapshot_images_to=SAVE_IMAGES_TO_DIR,
            scale_range=SCALE_RANGE
            )
    elif VISUALIZE == "3D":
        visualizer_method = Cluster3DVisualizer(
            clusterer, 
            show=False, 
            save_snapshot_images_to=SAVE_IMAGES_TO_DIR,
            scale_range=SCALE_RANGE
            )
    elif VISUALIZE == "2D3D":
        visualizer_method = Cluster2D3DVisualizer(
            clusterer, 
            show=False, 
            save_snapshot_images_to=SAVE_IMAGES_TO_DIR,
            scale_range=SCALE_RANGE
            )


#------------------------#
#    Fitness function    #
#------------------------#

    if FITNESS_EVALUATION_METHOD == "CentroidDistance":
        fitness_function = ClusterCentroidDistance(
            clusterer
        )
    elif FITNESS_EVALUATION_METHOD == "MinimumPointDistance":
        fitness_function = ClusterMinimumPointDistance(
            clusterer
        )
    elif FITNESS_EVALUATION_METHOD == "Accuracy":
        fitness_function = ClusterAccuracy(
            clusterer,
            ground_truth_cluster_labels
        )


#---------------#
#    Learner    #
#---------------#

    #Visualize ground truth dataset
    visualizer_method([1]*N_DIMENSIONS, "original", 
                    figure_title = type(fitness_function).__name__ + " optimization, " + \
                        "Ground truth clusters",
                        ground_truth = ground_truth_cluster_labels)

    genetic_learner = GeneticBinaryOptimizer(
        BinaryPopulationInitializer(POPULATION_SIZE, N_DIMENSIONS),
        fitness_function,
        RandomSplit(CROSSOVER_PROBABILITY),
        RandomBitFlip(MUTATION_PROBABILITY),
        early_stopping_criterion=ImprovementHistoryWithPatience(EARLY_STOPPING_PATIENCE),
        result_visualization_method=visualizer_method,
        n_parent_candidates=min(TOURNAMENT_SELECTION_CANDIDATES, POPULATION_SIZE-1),
        checkpoints_dir=SAVE_CHECKPOINTS_TO_DIR
    )

    genetic_learner.learn(N_ITERATIONS)
    
    genetic_learner.result_visualization.generate_gif()
    genetic_learner.result_visualization.generate_final_comparison()