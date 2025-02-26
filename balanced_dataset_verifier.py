from os import listdir

DATASET = "balanced_augmented_mapillary_dataset"
total_image_amount = 0

for directory in ["train", "val"]:
    images = sorted(listdir(f"{DATASET}/{directory}/images"))
    labels = sorted(listdir(f"{DATASET}/{directory}/labels"))

    assert len(images) == len(labels)
    for image, label in zip(images, labels):
        image = image.split(".")[0]
        label = label.split(".")[0]
        assert image == label, f"{directory} {image} {label}"
    
    total_image_amount += len(labels)

total_image_amount += len(listdir(f"{DATASET}/test/images"))
print(total_image_amount)
