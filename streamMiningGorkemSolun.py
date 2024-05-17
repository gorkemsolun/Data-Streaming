# Görkem Kadir Solun 22003214, Stream Mining, Project 5, 15.05.2024

# NOTE: Use python 3.8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Fix the float error
np.float = float
from skmultiflow.data import AGRAWALGenerator, SEAGenerator, DataStream, FileStream
from skmultiflow.meta import AdaptiveRandomForest, DynamicWeightedMajorityClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection import ADWIN
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.data import FileStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.drift_detection import DDM


# Random seed for reproducibility
random_seed = 1
np.random.seed(random_seed)

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
    df = pd.DataFrame(data, columns=[i for i in range(X.shape[1])] + ["label"])
    df.to_csv(filename, index=False)
    print(f"{filename} created successfully with {num_instances} instances.")


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

    df = pd.DataFrame(data, columns=[i for i in range(X.shape[1])] + ["label"])
    df.to_csv(filename, index=False)
    print(
        f"{filename} created successfully with {num_instances} instances and drifts at {drift_points}."
    )


# Generate datasets with taking approval from the user
if input("Do you want to generate the synthetic datasets? (y/n): ") == "y":
    generate_sea_dataset_with_drift()
    generate_agrawal_dataset()


# Download the real datasets
# Download the Spam and Electricity datasets from the provided GitHub link.

# URLs to the real datasets
spam_url = "https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/spam.csv"
electricity_url = "https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/elec.csv"

# Load the datasets
spam_data = pd.read_csv(spam_url)
electricity_data = pd.read_csv(electricity_url)

# Transform the headers to integers 0, 1, 2, 3, ..., n-2, label
spam_data.columns = [i for i in range(0, spam_data.shape[1] - 1)] + ["label"]
electricity_data.columns = [i for i in range(0, electricity_data.shape[1] - 1)] + [
    "label"
]

# Save datasets to files
spam_data.to_csv("Spam_Dataset.csv", index=False)
electricity_data.to_csv("Electricity_Dataset.csv", index=False)

# PART B Implementations for Handling Concept Drift

# Part B.1 b.1) Implement an instance of the following classification models available on scikit-multiflow.
# Adaptive Random Forest (ARF)
# Streaming Agnostic Model with k-Nearest Neighbors (SAM-kNN)
# Dynamic Weighted Majority (DWM)


# Manually plot the results if needed
# Extract the metrics you need from the results
def plot_results(results, dataset_name):
    accuracy_results = results.get_measurements("accuracy")
    plt.figure()
    for model_name, accuracy in accuracy_results.items():
        plt.plot(accuracy, label=model_name)
    plt.xlabel("Samples")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} Accuracy Over Time")
    plt.legend()
    plt.show()


# Create FileStream for the datasets
agrawal_stream = AGRAWALGenerator(
    random_state=random_seed
)  # FileStream("AGRAWAL_Dataset.csv", target_idx=-1)  # Same as the generated dataset but using the generator
spam_stream = FileStream("Spam_Dataset.csv", target_idx=-1)
electricity_stream = FileStream("Electricity_Dataset.csv", target_idx=-1)
sea_stream = FileStream("SEA_Dataset.csv", target_idx=-1)

# Initialize classifiers
arf = AdaptiveRandomForest()
samknn = SAMKNNClassifier()
dwm = DynamicWeightedMajorityClassifier()

# Initialize evaluator
evaluator = EvaluatePrequential(
    max_samples=100000, show_plot=False, metrics=["accuracy", "kappa"]
)

# Evaluate classifiers on each dataset
print("Evaluating on AGRAWAL Dataset")
results_agarwal = evaluator.evaluate(
    stream=agrawal_stream,
    model=[arf, samknn, dwm],
    model_names=["ARF", "SAM-kNN", "DWM"],
)

print("Evaluating on Spam Dataset")
results_spam = evaluator.evaluate(
    stream=spam_stream, model=[arf, samknn, dwm], model_names=["ARF", "SAM-kNN", "DWM"]
)

print("Evaluating on SEA Dataset")
results_sea = evaluator.evaluate(
    stream=sea_stream, model=[arf, samknn, dwm], model_names=["ARF", "SAM-kNN", "DWM"]
)

print("Evaluating on Electricity Dataset")
results_electricity = evaluator.evaluate(
    stream=electricity_stream,
    model=[arf, samknn, dwm],
    model_names=["ARF", "SAM-kNN", "DWM"],
)

# b.2) Build your own ensemble from scratch. Use HoeffdingTreeClassifier classifier as your base learner.
# Find a solution for handling drift in your ensemble and implement it. The ensemble approach should be
# implemented from scratch. You are only allowed to import the HoeffdingTreeClassifier, your preferred
# drift detector, and other basic libraries (numpy, pandas, etc.)


# Create a custom ensemble with HoeffdingTreeClassifier as the base learner and ADWIN as the drift detector
class CustomEnsemble:
    # Initialize the ensemble with n_estimators HoeffdingTreeClassifiers and drift detectors
    def __init__(self, n_estimators=10, drift_detector=ADWIN):
        self.n_estimators = n_estimators
        self.drift_detector = drift_detector
        self.ensemble = [HoeffdingTreeClassifier() for _ in range(n_estimators)]
        self.detectors = [drift_detector() for _ in range(n_estimators)]
        self.weights = (
            np.ones(n_estimators) / n_estimators
        )  # Initialize weights equally

    # Partial fit the ensemble with the new data
    def partial_fit(self, X, y, classes=None):
        for i in range(self.n_estimators):
            self.ensemble[i].partial_fit(X, y, classes)
            self.detectors[i].add_element(y)
            if self.detectors[i].detected_change():
                print(f"Drift detected in estimator {i}")
                self.ensemble[i] = HoeffdingTreeClassifier()
                self.detectors[i] = self.drift_detector()

    # Predict the class labels for the input samples
    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.ensemble])
        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions
        )


# Evaluate the custom ensemble on the datasets
def evaluate_custom_ensemble():
    # Load the datasets
    agrawal_stream = DataStream(np.loadtxt("AGRAWAL_Dataset.csv", delimiter=","))
    sea_drifted_stream = DataStream(np.loadtxt("SEA_Dataset.csv", delimiter=","))
    spam_stream = DataStream(np.loadtxt("Spam_Dataset.csv", delimiter=","))
    electricity_stream = DataStream(
        np.loadtxt("Electricity_Dataset.csv", delimiter=",")
    )

    # Initialize the custom ensemble
    custom_ensemble = CustomEnsemble()

    evaluator = EvaluatePrequential(
        max_samples=100000, show_plot=False, metrics=["accuracy", "kappa"]
    )

    print("Evaluating on AGRAWAL dataset")
    results_agrawal = evaluator.evaluate(  # Evaluate on AGRAWAL dataset
        stream=agrawal_stream,
        model=custom_ensemble,
        model_names=["CustomEnsemble"],
    )

    print("Evaluating on SEA dataset")
    results_sea = evaluator.evaluate(  # Evaluate on SEA dataset
        stream=sea_drifted_stream,
        model=custom_ensemble,
        model_names=["CustomEnsemble"],
    )

    print("Evaluating on Spam dataset")
    results_spam = evaluator.evaluate(  # Evaluate on Spam dataset
        stream=spam_stream,
        model=custom_ensemble,
        model_names=["CustomEnsemble"],
    )

    print("Evaluating on Electricity dataset")
    results_electricity = evaluator.evaluate(  # Evaluate on Electricity dataset
        stream=electricity_stream,
        model=custom_ensemble,
        model_names=["CustomEnsemble"],
    )

    plot_results(results_agrawal, "AGRAWAL")
    plot_results(results_sea, "SEA")
    plot_results(results_spam, "Spam")
    plot_results(results_electricity, "Electricity")


# Evaluate the custom ensemble
evaluate_custom_ensemble()

# PART C Implementations for Handling Adversarial Attack

# c.1) In this assignment, our focus is on instance-based attacks. For two synthetic datasets generated in part(a), synthesize a malicious attack in two points:

# Load synthetic datasets
agrawal_datafile = pd.read_csv("AGRAWAL_Dataset.csv")
sea_datafile = pd.read_csv("SEA_Dataset.csv")


# Function to flip labels
def flipper(df, start, end, flip_percentage):
    indices = np.random.choice(
        range(start, end), size=int((end - start) * flip_percentage), replace=False
    )
    df.loc[indices, "label"] = 1 - df.loc[indices, "label"]
    return df


# Inject adversarial attacks
agrawal_datafile = flipper(agrawal_datafile, 40000, 40500, 0.10)
agrawal_datafile = flipper(agrawal_datafile, 60000, 60500, 0.20)
sea_datafile = flipper(sea_datafile, 40000, 40500, 0.10)
sea_datafile = flipper(sea_datafile, 60000, 60500, 0.20)

# Save the modified datasets
agrawal_datafile.to_csv("AGRAWAL_Dataset_adversarial.csv", index=False)
sea_datafile.to_csv("SEA_Dataset_adversarial.csv", index=False)

# c.2) Propose and implement a solution for handling this attack. You can modify your ensemble model in b.2 for this purpose


# Create a custom ensemble with HoeffdingTreeClassifier as the base learner and DDM as the drift detector
class AdversarialAwareEnsemble:
    # Initialize the ensemble with HoeffdingTreeClassifiers and drift detectors
    def __init__(self, base_estimator, drift_detector=DDM(), threshold=0.10):
        self.base_estimator = base_estimator
        self.drift_detector = drift_detector
        self.threshold = threshold
        self.ensemble = []
        self.warning_zone = False

    # Fit the ensemble with the new data
    def fit(self, X, y):
        if not self.ensemble:
            self.ensemble.append(self.base_estimator.clone())
        for clf in self.ensemble:
            clf.partial_fit(X, y, classes=[0, 1])

    # Predict the class labels for the input samples
    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.ensemble])
        return np.apply_along_axis(
            lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=predictions
        )

    # Partial fit the ensemble with the new data
    def partial_fit(self, X, y):
        pred = self.predict(X)
        error_rate = np.mean(pred != y)
        self.drift_detector.add_element(error_rate)
        if self.drift_detector.detected_change():
            if self.warning_zone:
                self.ensemble.append(self.base_estimator.clone())
                self.warning_zone = False
            else:
                self.warning_zone = True
        if error_rate > self.threshold:
            return
        for clf in self.ensemble:
            clf.partial_fit(X, y, classes=[0, 1])


# Load adversarial datasets
agrawal_stream = DataStream(
    np.loadtxt("AGRAWAL_Dataset_adversarial.csv", delimiter=",")
)
sea_stream = DataStream(np.loadtxt("SEA_Dataset_adversarial.csv", delimiter=","))

# Initialize classifiers
arf = AdaptiveRandomForestClassifier()
adversarial_aware_ensemble = AdversarialAwareEnsemble(arf)

# Initialize evaluator
evaluator = EvaluatePrequential(
    show_plot=False, pretrain_size=200, max_samples=100000, metrics=["accuracy"]
)

# Evaluate classifiers on AGRAWAL dataset
print("Evaluating on AGRAWAL dataset")
evaluator.evaluate(
    stream=agrawal_stream,
    model=[arf, adversarial_aware_ensemble],
    model_names=["ARF", "Adversarial Aware Ensemble"],
)

# Evaluate classifiers on SEA dataset
print("Evaluating on SEA dataset")
evaluator.evaluate(
    stream=sea_stream,
    model=[arf, adversarial_aware_ensemble],
    model_names=["ARF", "Adversarial Aware Ensemble"],
)
