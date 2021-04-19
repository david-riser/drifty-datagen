"""

File: services/generator/tasks.py
Author: David Riser
Date Created: 4/19/2021
Date Modified: 4/19/2021
Purpose: This file defines prefect tasks
required to build the application.

"""
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import task, Task

import utils


@task
def save_batch(output_dir, features, labels):
    """

    Save the current batch of labeled data
    into a pickle file.

    """

    utils.check_and_create(output_dir)
    dataset_name = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    outfile_name = f"{output_dir}/{dataset_name}.pkl"

    with open(outfile_name, "wb") as outfile:
        pickle.dump(
            {
                "features":features,
                "labels":labels
            }, outfile)


class Generator(Task):
    """
    Generate a Poisson random number of data
    points in N classes with M features. N and M
    are defined by the shape of the centers array.

    centers.shape = (N, M) = (n_classes, n_features)

    The center drifts over time by simulating a Gaussian
    random walk in each dimension with a standard deviation
    defined by the jumps variable.

    """
    def __init__(self, centers, stds, jumps):
        self.centers = centers
        self.stds = stds
        self.jumps = jumps
        self.n_classes, self.n_features = self.centers.shape


    def run(self, n_samples):
        """

        Sample a batch of datapoints with expected size
        n_samples but actual size following a Poisson distr
        about that number.

        """
        n_samples = np.random.poisson(n_samples)
        x = np.zeros((n_samples, self.n_features))
        y = np.random.choice(np.arange(self.n_classes), size=n_samples)

        for label in range(self.n_classes):
            indices = np.where(y == label)[0]
            for axis in range(self.n_features):
                x[indices, axis] = np.random.normal(
                    loc=self.centers[label,axis],
                    scale=self.stds[label,axis],
                    size=len(indices)
                )

        self._update_centers()

        return x, y


    def _update_centers(self):
        """ Gaussian random walk on the centers. """
        for label in range(self.n_classes):
            for axis in range(self.n_features):
                self.centers[label, axis] = np.random.normal(
                    loc=self.centers[label, axis],
                    scale=self.jumps[label,axis]
                )


class MonitorCSVWriter(Task):
    """ Class to write a monitoring CSV file each time the
    task graph is executed. """
    def __init__(self, output_filename):
        self.output_filename = output_filename


    def run(self, features, labels):
        """ Calculate some aggregate statistics and then
        squirt them out into the logfile. """

        class_a_indices = np.where(labels == 0)[0]
        class_a_mean = np.mean(features[class_a_indices], axis=1)
        class_a_std = np.std(features[class_a_indices], axis=1)
        n_samples_a = len(class_a_indices)

        class_b_indices = np.where(labels == 1)[0]
        class_b_mean = np.mean(features[class_b_indices], axis=1)
        class_b_std = np.std(features[class_b_indices], axis=1)
        n_samples_b = len(class_b_indices)

        data = {
            "n_samples_a":n_samples_a, "n_samples_b":n_samples_b,
            "class_a_mean1":class_a_mean[0], "class_a_mean2":class_a_mean[1],
            "class_a_std1":class_a_std[0], "class_a_std2":class_a_std[1],
            "class_b_mean1":class_b_mean[0], "class_b_mean2":class_b_mean[1],
            "class_b_std1":class_b_std[0], "class_b_std2":class_b_std[1]
        }

        dataframe = pd.DataFrame([data,])

        out_path = Path(self.output_filename)
        if not out_path.exists():
            dataframe.to_csv(self.output_filename, index=False)
        else:
            old_dataframe = pd.read_csv(self.output_filename)
            new_dataframe = pd.concat([dataframe, old_dataframe])
            new_dataframe.to_csv(self.output_filename, index=False)
