import csv

import numpy as np
from sklearn import datasets

from utils import create_indices, load_data_from_csv

from sklearn.datasets import make_blobs
from visualization import visualize_2D

class Dataset:

    #Creates an instance of the Dataset
    def __init__(self, field_labels, samples = None, cluster_labels=None) -> None:
        self.samples = []
        self.couples_to_samples = {}
        if samples is not None:
            self.samples = samples
        else:
            self.couples_to_samples = {}
        
        self.field_labels = field_labels
        self.cluster_labels = cluster_labels

        #Create the label vocabulary
        label_to_idx, idx_to_label = create_indices(field_labels)

        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label

    #Create a sample instance and add it to this dataset instance
    def add_sample(self, sample):
        self.samples.append(sample)

    @classmethod
    def generate_blob_dataset(cls, n_samples, n_fields, n_clusters, save_to = None, show = False, 
                            clusters_centers_range = (-10,10), std_dev = 10):
        dataset, classifications = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_fields, 
        random_state=0, center_box=clusters_centers_range, cluster_std=std_dev)

        print("Dataset of shape "+str(np.shape(dataset))+" randomly generated ")

        if show:
            visualize_2D(dataset, show=True, cluster_predictions=classifications)

        if save_to is not None:
            #Taken from https://www.tutorialspoint.com/How-to-save-a-Python-Dictionary-to-CSV-file
            
            try:
                with open(save_to, 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for data, classif in zip(dataset, classifications):
                        writer.writerow(data.tolist() + [classif])
            except IOError:
                print("I/O error")

        return Dataset([str(i) for i in range(n_fields)], samples = dataset, cluster_labels=classifications), classifications
     
    @classmethod
    def load_blob_dataset(cls, filename, show = False):
        dataset = []
        classifications = []
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    parsed_row = []
                    for item in row[:-1]:
                        parsed_row.append(float(item))
                    dataset.append(parsed_row)
                    classifications.append(int(row[-1]))
        except IOError:
            print("I/O error")

        datasets = np.array(dataset)
        classifications = np.array(classifications)

        n_samples = np.shape(dataset)[0]
        n_fields = np.shape(dataset)[1]
        n_clusters = len(np.unique(classifications, return_counts=True)[1])
        
        #print(n_samples)
        #print(n_fields)
        #print(n_clusters)

        print("Dataset of shape "+str(np.shape(dataset))+" loaded")

        if show:
            visualize_2D(dataset, show=True, cluster_predictions=classifications)

        return Dataset([str(i) for i in range(n_fields)], samples = dataset, cluster_labels=classifications), classifications, n_samples, n_fields, n_clusters
      
    #Loads a dataset from a file and returns it as an instance of this class
    @classmethod
    def load_dataset(cls, dataset_dir, dataset_filename):
        data, labels = load_data_from_csv(dataset_dir, dataset_filename)
                
        dataset = Dataset(labels)

        #Initialize dataset as a list of objects
        for sample in data:    
            dataset.add_sample(sample)
            
        return dataset

    def __repr__(self):
        return [sample.__repr__() for sample in self.samples]

    def __str__(self):
        return "\n\nDataset with "+str(len(self.samples))+" entries\n"+str(self.__repr__())

    def __get__(self, index):
        return self.sample[index]

    #Selects values from the selected fields, specified by passing a vector of 0/1 for each dataset field,
    #for all samples, returning the result as a numpy array (2D tensor)
    def samples_to_numpy_array(self, field_selection = None):
        if field_selection is None:
            scale_indices = np.ones(self.shape)
        else:
            scale_indices = np.nonzero(field_selection)[0]

        arr = []
        for sample in self.samples:
            selected_sample_fields = []
            for idx in scale_indices:
                selected_sample_fields.append(sample[idx])
            arr.append(selected_sample_fields)

        return np.stack(arr)
