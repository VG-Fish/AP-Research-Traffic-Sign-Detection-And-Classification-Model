import fastdup
from multiprocessing import freeze_support

def main():
    fd = fastdup.create(
        input_dir="rare_balanced_augmented_mapillary_dataset",
        work_dir="rare_balanced_dataset_analysis",
    )

    fd.run(overwrite=True)

    fd.explore()

if __name__ == '__main__':
    freeze_support()
    main()