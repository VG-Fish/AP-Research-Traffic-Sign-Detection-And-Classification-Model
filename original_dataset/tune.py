from ultralytics import YOLO
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune import schedulers

bayesopt = BayesOptSearch()

model = YOLO("yolo11n.pt")

search_space = {
    "conf": tune.uniform(0.0, 1.0),
    "lr0": tune.loguniform(0.0001, 0.1),
    "lrf": tune.uniform(0.01, 1.0),
    "cls": tune.uniform(0.2, 4.0),
    "box": tune.uniform(0.02, 0.2),
    "degrees": tune.randint(0.0, 45.0),
    "scale": tune.uniform(0.0, 0.9), 
    "mosaic": tune.uniform(0.0, 1.0), 	
    "mixup": tune.uniform(0.0, 1.0), 	
    "copy_paste": tune.uniform(0.0, 1.0),
    "batch": tune.randint(8.0, 64.0),
}

def yolo(config):
    model = YOLO("yolo11n.pt") 
    results = model.train(
        data="mapillary.yaml",
        epochs=10,
        lr0=config["lr0"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    return {"mAP": results.results_dict["metrics/mAP50-95(B)"]}

tuner = tune.Tuner(
    yolo,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=10,
        scheduler=schedulers.ASHAScheduler(grace_period=1),
        mode="max",
        metric="mAP"
    )
)

# Run the hyperparameter tuning
results = tuner.fit()

# Get the best trial
best_trial = results.get_best_result(metric="mAP", mode="max")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final mAP: {best_trial.metrics['mAP']}")


"""results = model.tune(
    # absolute path is required for some reason.
    data="/Users/vishy/Desktop/AP Research/Code/mapillary.yaml", 
    use_ray=True,
    space=search_space,
    epochs=10,
    iterations=100,
    optimizer="AdamW",
    save=True,
    plots=True,
    project="train_tuning",
    name="tune_test",
    val=False,
    device="mps",
    exist_ok=True,
    save_period=1,
    fraction=0.1,
    batch=16,
    imgsz=640,
    amp=True,
    show_boxes=True,
    seed=16,
    cos_lr=True,
    search_alg=bayesopt,
)"""