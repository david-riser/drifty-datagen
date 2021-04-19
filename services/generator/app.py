"""

File: services/generator/app.py
Author: David Riser
Date Created: 4/19/2021
Date Modified: 4/19/2021
Purpose: This application generates realistic drifting
data for binary classification.

"""
import argparse
from datetime import datetime, timedelta

import yaml
from prefect import Flow
from prefect.schedules import IntervalSchedule

import utils
import tasks

class Application:
    """ Singleton used to configure and run the
    data generation app. """
    def __init__(self, args):
        self.args = args

    def configure(self):
        """ Load the configuration file
        and populate the required arrays.
        """
        with open(self.args.config, "r") as ymlfile:
            self.config = yaml.safe_load(ymlfile)

        # Configure the Prefect task scheduler
        self.schedule = IntervalSchedule(
            start_date=datetime.utcnow() + timedelta(seconds=1),
            interval=timedelta(
                seconds=self.config["program"]["time_interval"]
            ),
        )

        self.centers, self.stds, self.jumps = utils.load_generation_params(self.config)
        self.generator = tasks.Generator(self.centers, self.stds, self.jumps)


    def run(self):
        """ Start the workflow. """
        with Flow("DataGenerator", schedule=self.schedule) as flow:
            features, labels = self.generator.run(
                n_samples=self.config["generator"]["n_samples"]
            )
            tasks.save_batch(self.config["program"]["output_dir"], features, labels)
            flow.run()


if __name__ == "__main__":

    # Collect configuration file from the user
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, type=str,
        help = "Yaml configuration file."
    )
    args = parser.parse_args()

    # Launch the application
    app = Application(args)
    app.configure()
    app.run()
