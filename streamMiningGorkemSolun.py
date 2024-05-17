# Görkem Kadir Solun 22003214, Stream Mining, Project 5, 15.05.2024

import numpy as np
import pandas as pd
import requests

# Fix the float error
np.float = float
from skmultiflow.data import AGRAWALGenerator, SEAGenerator
from skmultiflow.meta import AdaptiveRandomForest, DynamicWeightedMajorityClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.evaluation import EvaluatePrequential

# Random seed for reproducibility
random_seed = 1

# PART A Dataset Preparation
# Use the AGRAWALGenerator and SEAGenerator classes from scikit-multiflow to create 100,000 data instances for each dataset.
# Apply concept drift to the SEAGenerator dataset at specified points.
# Save the generated datasets to files.

# Generate Synthetic Datasets
# Generate the AGRAWAL and SEA datasets with 100,000 instances each and save them to files. Concept drift will be applied to the SEA dataset at specified points.


# Function to generate AGRAWAL dataset
def generate_agrawal_dataset(num_instances=100000, filename="AGRAWAL_Dataset.csv"):
    agrawal_gen = AGRAWALGenerator(random_state=random_seed)
    data = []
    for _ in range(num_instances):
        X, y = agrawal_gen.next_sample()
        data.append(np.hstack((X[0], y[0])))
    df = pd.DataFrame(data, columns=[i for i in range(X.shape[1])] + ["target"])
    df.to_csv(filename, index=False)
    print(f"{filename} created successfully with {num_instances} instances.")


generate_agrawal_dataset()


# Function to generate SEA dataset with concept drift
def generate_sea_dataset_with_drift(num_instances=100000, filename="SEA_Dataset.csv"):
    sea_gen = SEAGenerator(random_state=random_seed)
    data = []
    drift_points = [25000, 50000, 75000]

    for i in range(num_instances):
        # Change generator parameters at drift points
        if i in drift_points:
            sea_gen.next_sample()
            sea_gen.random_state += 1  # Example of parameter change to simulate drift

        X, y = sea_gen.next_sample()
        data.append(np.hstack((X[0], y[0])))

    df = pd.DataFrame(data, columns=[i for i in range(X.shape[1])] + ["target"])
    df.to_csv(filename, index=False)
    print(
        f"{filename} created successfully with {num_instances} instances and drifts at {drift_points}."
    )


generate_sea_dataset_with_drift()


# Download the real datasets
# Download the Spam and Electricity datasets from the provided GitHub link.

# URLs to the real datasets
spam_url = "https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/spam.csv"
electricity_url = "https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/elec.csv"

# Load the datasets
spam_data = pd.read_csv(spam_url)
electricity_data = pd.read_csv(electricity_url)

# Transform the headers to integers 1, 2, 3, ..., n-1, target
spam_data.columns = [i for i in range(1, spam_data.shape[1])] + ["target"]
electricity_data.columns = [i for i in range(1, electricity_data.shape[1])] + ["target"]

# Save datasets to files
spam_data.to_csv("Spam_Dataset.csv", index=False)
electricity_data.to_csv("Electricity_Dataset.csv", index=False)

# PART B Implementations for Handling Concept Drift

agrawal_data = pd.read_csv("AGRAWAL_Dataset.csv")
sea_data = pd.read_csv("SEA_Dataset.csv")

# Prepare data streams
