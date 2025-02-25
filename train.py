from ultralytics import YOLO
import ultralytics.data.build as build
from weighted_dataset import YOLOWeightedDataset
import torch

build.YOLODataset = YOLOWeightedDataset

"""
To disable sleep:
sudo pmset -b sleep 0
sudo pmset -b disablesleep 1

To enable sleep:
sudo pmset -b sleep 5
sudo pmset -b disablesleep 0

To purge RAM memory:
sudo purge

To disable CPU throttling:
sudo pmset -a lidwake 0
sudo pmset -a disablesleep 1

To keep the macbook awake:
caffeinate -i -d -m -u -t VALUE

To get GPU usage:
sudo powermetrics --samplers gpu_power

To get memory usage:
vm_stat

To increase open file limit:
ulimit -n 100000
"""

model = YOLO("yolo11n.pt")

def clear_cache(_):
    torch.mps.empty_cache()

model.add_callback("on_train_batch_start", clear_cache)
model.add_callback("on_val_batch_start", clear_cache)

try:
    results = model.train(
        data="mapillary.yaml",
        project="train",
        name="small_objects",
        epochs=1,
        val=False,
        device="mps",
        patience=3,
        batch=48,
        save_period=1,
        imgsz=640,
        exist_ok=True,
        optimizer="AdamW",
        amp=True, # mixed precision training
        plots=True,
        max_det=73, # The max number of annotations for an image is 73
        show_boxes=True,
        cos_lr=True, # Learning rate oscillates for better convergence
        save_json=True,
        augment=True,
        seed=16,
    )
except KeyboardInterrupt:
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    def verify_class_balance(dataset, num_samples=1000):
        """
        Verifies whether the __getitem__ method in the YOLOWeightedDataset class returns a balanced class output.

        Args:
            dataset: An instance of YOLOWeightedDataset.
            num_samples: Number of samples to draw from the dataset.

        Returns:
            class_counts: A dictionary containing the class counts.
        """
        all_labels = []
        num_samples = min(len(dataset.labels), num_samples)

        if dataset.train_mode:
            choices = np.random.choice(len(dataset.labels), size=num_samples, p=dataset.probabilities)
        else:
            choices = np.random.choice(len(dataset.labels), size=num_samples, replace=False)

        for i in choices:
            label = dataset.labels[i]["cls"]
            all_labels.extend(label.reshape(-1).astype(int))

        class_counts = Counter(all_labels)
        return class_counts

    def plot_class_balance(weighted_cnts, unweighted_cnts, class_names):
        """
        Plots the comparison of class distribution between training and validation modes.

        Args:
            weighted_cnts: A dictionary containing the class counts in weighted mode.
            unweighted_cnts: A dictionary containing the class counts in unweighted mode.
            class_names: A list of class names.
        """
        classes = range(len(class_names))
        weighted_values = [weighted_cnts.get(c, 0) for c in classes]
        unweighted_values = [unweighted_cnts.get(c, 0) for c in classes]

        width = 0.35  # Bar width

        _, ax = plt.subplots()
        ax.bar(classes, unweighted_values, width, label='Normal mode')
        ax.bar([c + width for c in classes], weighted_values, width, label='Weighted Mode')

        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution in Normal vs Weighted Modes')
        ax.set_xticks([c + width / 2 for c in classes])
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()

        plt.show()
    
    from pprint import pprint
    # You can test different aggregation functions np.max, np.sum, np.median, np.mean
    def plot():
        model.trainer.train_loader.dataset.weights = model.trainer.train_loader.dataset.calculate_weights()
        model.trainer.train_loader.dataset.probabilities = model.trainer.train_loader.dataset.calculate_probabilities()

        # Get class counts in weighted mode
        model.trainer.train_loader.dataset.train_mode = True
        weighted_counts = verify_class_balance(model.trainer.train_loader.dataset, num_samples=100_000)
        # pprint(weighted_counts)
        pprint(model.trainer.train_loader.dataset.evaluate_balance())

        # Get class counts in default mode
        model.trainer.train_loader.dataset.train_mode = False
        default_counts = verify_class_balance(model.trainer.train_loader.dataset, num_samples=100_000)

        # Plot the comparison
        plot_class_balance(weighted_counts, default_counts, set(model.trainer.train_loader.dataset.data["names"].values()))
    
    funcs = [
        lambda x: x ** 10,
        lambda x: x ** 20,
        lambda x: x ** 5,
        lambda x: x ** 3,
        lambda x: x ** 2,
        lambda x: x ** 1.5,
        lambda x: x ** 0.5,
    ]
    for idx, func in enumerate(funcs):
        model.trainer.train_loader.dataset.agg_func = func
        plot()