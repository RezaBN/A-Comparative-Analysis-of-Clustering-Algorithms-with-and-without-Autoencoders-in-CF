[Back: Main](../README.md)

# Initialization README

The Initialization script is part of the Movie Recommendation System project and is used to set up command-line arguments for configuring the system. This script defines various parameters that influence the behavior of the recommendation system.

## Usage

To use the Initialization script, you need to run it with Python and provide command-line arguments to configure the behavior of other parts of the project. Here are the available command-line arguments:

- `--dataset`: Choose a dataset (default: "MovieLens_100K" but can be set to "Your_Dataset").
- `--hidden_units`: Set the number of hidden units in the autoencoder (default: 64).
- `--epochs`: Set the number of training epochs for the autoencoder (default: 10).
- `--batch_size`: Set the batch size for training the autoencoder (default: 32).
- `--num_clusters`: Set the number of clusters for clustering (default: 3).
- `--pcc_threshold`: Set the threshold for PCC-based recommendation (default: 0.3).
- `--jaccard_threshold`: Set the threshold for Jaccard-based recommendation (default: 0.16).
- `--clustering_method`: Choose a clustering method (default: "kmeans" but can be one of: "kmeans," "kmedoids," "dbscan," "som," or "fcm").


## Custom Dataset

If you choose to use your own dataset, make sure the dataset is structured appropriately for this project and that you have the necessary permissions to use it. You can specify your dataset as the `--dataset` argument when running the system.

Example usage:
```shell
python initialization.py --dataset Your_Dataset --hidden_units 128 --epochs 20 --batch_size 64 --num_clusters 5 --pcc_threshold 0.2 --jaccard_threshold 0.1 --clustering_method kmeans

Modify these command-line arguments to tailor the recommendation system's configuration to your specific requirements.

## License

This script is part of the Movie Recommendation System project and is available under the MIT License.

Feel free to contribute and enhance the project!

[Next: Data Preparing](data_preparing_package.md)

[Back to Main](../README.md)


# Citation

If you use this Movie Recommendation System project in your research or as a part of your work, we kindly request that you cite it using the following BibTeX entry:

```bibtex
@misc{BarzegarNozari2023,
  title = {Movie Recommendation System},
  author = {Barzegar Nozari, Reza},
  year = {2023},
  howpublished = {\url{https://github.com/YourGitHubUsername/YourRepositoryName}},
  note = {Accessed: Insert Access Date}
}