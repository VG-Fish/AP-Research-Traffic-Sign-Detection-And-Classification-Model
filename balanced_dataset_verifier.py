from os import listdir

def verify_dataset():
    DATASET = "balanced_augmented_mapillary_dataset"
    total_image_amount = 0

    for directory in ["train", "val"]:
        images = sorted(listdir(f"{DATASET}/{directory}/images"))
        labels = sorted(listdir(f"{DATASET}/{directory}/labels"))

        try:
            assert len(images) == len(labels)
        except AssertionError:
            print(f"{directory = } {len(images) = } {len(labels) = }")
            a, b = set(images), set(labels)
            print(a - b)
            print(b - a)
        
        for image, label in zip(images, labels):
            image_name = image.split(".")[0]
            label_name = label.split(".")[0]
            assert image_name == label_name, f"{directory = } {image = } {label = }"
        
        total_image_amount += len(labels)

    total_image_amount += len(listdir(f"{DATASET}/test/images"))
    print(total_image_amount)
