import fastdup
from multiprocessing import freeze_support
import pandas as pd
from os.path import exists
from os import remove

def extract_image_duplicates(row):
    filenames = row["filenames"]
    image = filenames[0]
    duplicates = filenames[1:] if len(filenames) > 1 else []

    return pd.Series({"image": image, "duplicates": duplicates, "mean_distance": row["mean_distance"]})

def remove_stat_with_threshold(fd, stat, threshold):
    stat_images = fd.img_stats()[[stat, "filename"]]
    stat_images = stat_images[stat_images[stat] <= threshold]

    for file in stat_images["filename"]:
        label_path = f"{file.split('.')[0]}.txt"
        if exists(file):
            remove(file)
        if exists(label_path):
            remove(label_path)
    print(f"\nRemoved {len(stat_images)} images.")

def main():
    fd = fastdup.create(
        input_dir="balanced_mapillary_dataset",
        work_dir="balanced_dataset_analysis",
    )

    fd.run()

    remove_stat_with_threshold(fd, "blur", 50)

    connected_components_df , _ = fd.connected_components()

    duplicates_df = (
        connected_components_df
        .groupby("component_id")
        .agg(
            filenames=("filename", list),
            count=("filename", "size"),
            mean_distance=("mean_distance", "mean")
        )
        .sort_values("mean_distance", ascending=False)
    )

    df = duplicates_df.apply(extract_image_duplicates, axis=1)
    
    df[df["mean_distance"] >= 0.95]["duplicates"].to_json("smart_dataset/duplicates.json")

    fd.summary()

    # fd.explore()

if __name__ == '__main__':
    freeze_support()
    main()