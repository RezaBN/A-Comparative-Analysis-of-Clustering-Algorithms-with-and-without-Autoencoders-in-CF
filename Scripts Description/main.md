[Back: Visualizatio](visualizatopn.md)

# Main Script

The Main script is the core of the Movie Recommendation System project. It orchestrates various components such as data preparation, clustering, recommendation, and visualization. The script reads and preprocesses the dataset, trains an autoencoder to obtain user embeddings, performs clustering, evaluates clusters, and runs a recommendation system based on user preferences.

## Functions

### 1. Main

The `main` function serves as the entry point of the script. It performs the following key tasks:

- Reads and preprocesses the dataset based on user-defined parameters.
- Trains an autoencoder to generate user embeddings.
- Performs clustering using the selected method.
- Evaluates clustering results with Davies-Bouldin and Dunn indices.
- Runs the recommendation system with user-specified thresholds.
- Saves the results in an Excel file and prints evaluation metrics.
- Visualizes and saves prediction results using the Visualization script.

## Usage

To run the Main script, execute it as the main program. The script imports settings and functions from the Initialization, Data Preparing, Clustering, Recommendation, and Visualization packages. You can specify the desired dataset, clustering method, and other parameters by modifying the command-line arguments.

```python
python main.py --dataset Your_Dataset --hidden_units 64 --epochs 10 --batch_size 32 --num_clusters 3 --pcc_threshold 0.3 --jaccard_threshold 0.16 --clustering_method dbscan

Modify these command-line arguments to tailor the recommendation system's configuration to your specific requirements.

## License

This script is part of the Movie Recommendation System project and is available under the MIT License.

Feel free to contribute and enhance the project!

# END

[Back to Main Page](../README.md)


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
