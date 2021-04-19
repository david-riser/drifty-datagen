import argparse
import numpy as np
import pickle 
import yaml

from datetime import datetime, timedelta
from pathlib import Path
from prefect import task, Flow, Task
from prefect.schedules import IntervalSchedule


def check_and_create(target):
    """ Use pathlib to check and create the output
    folder if it does not exist."""
    output_path = Path(target)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        

@task
def save_batch(output_dir, features, labels):
    """ 

    Save the current batch of labeled data
    into a pickle file.

    """

    check_and_create(output_dir)
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



def load_generation_params(config):
    """ From the provided configuration file
    load the parameters that define class
    location, shape, and evolution. """
    
    centers = np.array([
        [config["generator"]["center_x1"], config["generator"]["center_y1"]],
        [config["generator"]["center_x2"], config["generator"]["center_y2"]]
    ])
    stds = np.array([
        [config["generator"]["std_x1"], config["generator"]["std_y1"]],
        [config["generator"]["std_x2"], config["generator"]["std_y2"]]
    ])
    jumps = np.array([
        [config["generator"]["jump_x1"], config["generator"]["jump_y1"]],
        [config["generator"]["jump_x2"], config["generator"]["jump_y2"]]
    ])
    
    return centers, stds, jumps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, type=str,
        help = "Yaml configuration file."
    )
    args = parser.parse_args()

    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    centers, stds, jumps = load_generation_params(config)
    
    schedule = IntervalSchedule(
        start_date=datetime.utcnow() + timedelta(seconds=1),
        interval=timedelta(seconds=config["program"]["time_interval"]),
    )
    generator = Generator(centers, stds, jumps)

    with Flow("DataFlow", schedule=schedule) as flow:
        features, labels = generator.run(
            n_samples=config["generator"]["n_samples"]
        )
        save_batch(config["program"]["output_dir"], features, labels)
        flow.run() 
        


