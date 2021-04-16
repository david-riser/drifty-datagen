import numpy as np
import pickle 

from datetime import datetime, timedelta

from prefect import task, Flow
from prefect.schedules import IntervalSchedule
from sklearn.datasets import make_classification


@task(nout=2)
def generate_batch(n_samples, n_features, class_sep):
    """ 

    Docstring. 
    
    """
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_classes=2, class_sep=class_sep
    )


@task(nout=2)
def make_blobs(n_samples, centers=None, stds=None, sample_poisson=True):
    """ 
    
    Make a binary classification problem with a variable
    number of datapoints per call.
    
    """

    if sample_poisson:
        n_samples = np.random.poisson(n_samples)

    if centers is None:
        centers = np.zeros(2)

    if stds is None:
        stds = np.ones(2)

    cols = centers.shape[0]
    if cols is not 2:
        raise TypeError("Centers must be an array of size (2,)")

    cols = stds.shape[0]
    if cols is not 2:
        raise TypeError("Stds must be an array of size (2,)")

    labels = np.random.choice([0, 1], size=n_samples, replace=True)
    features = np.zeros((n_samples,))

    for label in [0, 1]:
        indices = np.where(labels == label)[0]
        features[indices] = np.random.normal(loc=centers[label], scale=stds[label], size=len(indices))

    return features, labels


@task
def save_batch(features, labels):
    """ 
    
    Docstring.
    

    """
    dataset_name = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    outfilename = f"data/raw/{dataset_name}.pkl"
    with open(outfilename, "wb") as outfile:
        pickle.dump(
            {
                "features":features,
                "labels":labels
            }, outfile)
        
    
class Generator:

    def __init__(self, centers, stds, jumps):
        self.centers = centers
        self.stds = stds
        self.jumps = jumps
        self.n_classes, self.n_features = self.centers.shape

    def generate(self, n_samples):

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
        for label in range(self.n_classes):
            for axis in range(self.n_features):
                self.centers[label, axis] = np.random.normal(
                    loc=self.centers[label, axis], 
                    scale=self.jumps[label,axis]
                )

                
if __name__ == "__main__":

    
    schedule = IntervalSchedule(
        start_date=datetime.utcnow() + timedelta(seconds=1),
        interval=timedelta(seconds=10),
    )

    centers = np.array([1., 1.4])
    stds = np.array([0.1, 0.1])
    
    with Flow("Data Generation Flow", schedule=schedule) as flow:
        #features, labels = generate_batch(
        #    n_samples=100,
        #    n_features=2,
        #    class_sep=1.0
        #)
        features, labels = make_blobs(n_samples=200, centers=centers, stds=stds)
        save_batch(features, labels)
        flow.run() 
        


