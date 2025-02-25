from ultralytics.data.dataset import YOLODataset
import numpy as np
from collections import Counter
from pprint import pprint
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
        self.agg_func = self.to_safe_agg(np.mean)

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
    
    def to_safe_agg(self, func):
        return lambda x: np.mean(np.abs(func(x)))

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0] * len(self.data["names"])
        for label in self.labels:
            traffic_sign_class_ids = label['cls'].reshape(-1).astype(int)
            for id in traffic_sign_class_ids:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        # To prevent divide by zero errors
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        self.agg_func = self.to_safe_agg(self.agg_func)
        weights = []
        for label in self.labels:
            traffic_sign_class_ids = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background images
            if traffic_sign_class_ids.size == 0:
              weights.append(1)
              continue

            # You can change this weight aggregation function to aggregate weights differently
            # weight = np.mean(self.class_weights[cls])
            # weight = np.max(self.class_weights[cls])
            weight = self.agg_func(self.class_weights[traffic_sign_class_ids])
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

        index = np.random.choice(len(self.labels), p=self.probabilities)
        return self.transforms(self.get_image_and_label(index))

    def verify_class_balance(self, num_samples=1000):
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