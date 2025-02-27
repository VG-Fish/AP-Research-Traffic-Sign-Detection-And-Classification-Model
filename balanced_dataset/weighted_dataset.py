from ultralytics.data.dataset import YOLODataset
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
    
    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
              weights.append(1)
              continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))

    def verify_class_balance(self, num_samples=10000):
        """
        Verifies whether the __getitem__ method in the YOLOWeightedDataset class returns a balanced class output.

        Args:
            dataset: An instance of YOLOWeightedDataset.
            num_samples: Number of samples to draw from the dataset.

        Returns:
            class_counts: A dictionary containing the class counts.
        """
        all_labels = []
        num_samples = min(len(self.labels), num_samples)

        if self.train_mode:
            choices = np.random.choice(len(self.labels), size=num_samples, p=self.probabilities)
        else:
            choices = np.random.choice(len(self.labels), size=num_samples, replace=False)

        for i in choices:
            label = self.labels[i]["cls"]
            all_labels.extend(label.reshape(-1).astype(int))

        class_counts = Counter(all_labels)
        return class_counts
    
    def plot_class_balance(self, weighted_cnts, unweighted_cnts):
        """
        Plots the comparison of class distribution between training and validation modes.

        Args:
            weighted_cnts: A dictionary containing the class counts in weighted mode.
            unweighted_cnts: A dictionary containing the class counts in unweighted mode.
            class_names: A list of class names.
        """
        class_names = set(self.data["names"].values())
        classes = range(len(class_names))
        weighted_values = [weighted_cnts.get(c, 0) for c in classes]
        unweighted_values = [unweighted_cnts.get(c, 0) for c in classes]

        width = 0.35  # Bar width

        fig, ax = plt.subplots()
        ax.bar(classes, unweighted_values, width, label='Normal mode')
        ax.bar([c + width for c in classes], weighted_values, width, label='Weighted Mode')

        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution in Normal vs Weighted Modes')
        ax.set_xticks([c + width / 2 for c in classes])
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()

        plt.show()
    
    def evaluate_balance(self, num_samples=1000):
        """Evaluates current weighting strategy"""
        class_counts = self.verify_class_balance(num_samples)
        # Calculate balance metrics
        counts = np.array([class_counts.get(i, 0) for i in range(len(self.data["names"]))])
        non_zero = counts[counts > 0]
        metrics = {
            'max_min_ratio': np.max(non_zero) / np.min(non_zero) if len(non_zero) > 1 else float('inf'),
            'std_normalized': np.std(non_zero) / np.mean(non_zero) if len(non_zero) > 0 else float('inf'),
            'zero_classes': np.sum(counts == 0)
        }
        return class_counts, metrics