import pickle 

from abc import ABC
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


@task
def save_batch(features, labels):
    """ 
    
    Docstring.
    

    """
    now = datetime.now()
    outfilename = f"data/raw/{now.year}.{now.month}.{now.day}.{now.minute}.{now.second}.outfile.pkl"
    with open(outfilename, "wb") as outfile:
        pickle.dump(
            {
                "features":features,
                "labels":labels
            }, outfile)
        


# class ScheduledParameter(ABC):
    
        

if __name__ == "__main__":

    
    schedule = IntervalSchedule(
        start_date=datetime.utcnow() + timedelta(seconds=1),
        interval=timedelta(seconds=10),
    )

    with Flow("Data Generation Flow", schedule=schedule) as flow:
        features, labels = generate_batch(
            n_samples=100,
            n_features=2,
            class_sep=1.0
        )
        save_batch(features, labels)
        flow.run() 
        


