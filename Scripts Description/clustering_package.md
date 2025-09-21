[Back: Data Preparing](data_preparing_package.md)

# Clustering Package

The Clustering Package is responsible for training an autoencoder for user embeddings and performing clustering using different methods. It includes functions for training an autoencoder, performing various clustering methods (K-means, K-medoids, DBSCAN, SOM, FCM), and evaluating clusters using the Davies-Bouldin and Dunn indices. This section explains the functionality and usage of the clustering part of the Movie Recommendation System project.

## Training the Autoencoder

The clustering package provides functionality to train an autoencoder on the input data and obtain user embeddings. The trained autoencoder model can be used to create meaningful representations of users in a lower-dimensional space.

### Autoencoder Architecture

The autoencoder is defined with an architecture consisting of an input layer, a hidden layer, and an output layer. The number of hidden units can be customized using the `--hidden_units` command-line argument.

## Clustering Methods

The clustering package supports various clustering methods to group users into clusters based on their embeddings. You can choose from the following clustering methods:

- **K-means Clustering**: Use the K-means algorithm to cluster users into a specified number of clusters.

- **K-medoids Clustering**: Perform K-medoids clustering on user embeddings. You can specify the number of clusters and initial medoid indexes.

- **DBSCAN Clustering**: Apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to cluster users. You can customize epsilon (`eps`) and the minimum number of samples (`min_samples`) for DBSCAN.

- **SOM (Self-Organizing Map) Clustering**: Train a Self-Organizing Map on the user embeddings and extract clusters.

- **Fuzzy C-means Clustering (FCM)**: Perform Fuzzy C-Means clustering on user embeddings. You can specify the number of clusters and adjust the fuzziness parameter.

## Evaluating Clusters

After clustering, the package provides evaluation metrics for the clusters, including the Davies-Bouldin Index and Dunn Index. These metrics help assess the quality and separation of clusters.

## Example Usage

Here's an example of how to use the clustering package to train an autoencoder and perform clustering:

```python
import clustering_package

# Train an autoencoder
trained_encoder = clustering_package.train_autoencoder(train_matrix, num_hidden_units=64, epochs=10, batch_size=32)

# Obtain user embeddings
user_embeddings = trained_encoder.predict(train_matrix)

# Perform K-means clustering
cluster_labels, centroids = clustering_package.kmeans_clustering(user_embeddings, num_clusters=3, n_init=10)

## License
This script is part of the Movie Recommendation System project and is available under the MIT License.

Feel free to contribute and enhance the project!


[Next: Recommendation](recommendation_package.md)

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
