import fastdup
from multiprocessing import freeze_support
import pandas as pd

def extract_image_duplicates(row):
    filenames = row["filenames"]
    image = filenames[0]
    duplicates = filenames[1:] if len(filenames) > 1 else []

    return pd.Series({"image": image, "duplicates": duplicates, "mean_distance": row["mean_distance"]})

def main():
    fd = fastdup.create(
        input_dir="rare_balanced_augmented_mapillary_dataset",
        work_dir="rare_balanced_dataset_analysis",
    )

    fd.run()

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
    
    print(df[df["mean_distance"] == 1]["duplicates"].to_json("smart_dataset/duplicates.json"))

    fd.summary()

    # fd.explore()


if __name__ == '__main__':
    freeze_support()
    main()