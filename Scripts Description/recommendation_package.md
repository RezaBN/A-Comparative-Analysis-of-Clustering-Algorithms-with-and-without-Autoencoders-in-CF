[Back: Clustering](clustering_package.md)

# Recommendation Package

The Recommendation Package is responsible for making movie recommendations to users based on clustering results. This section explains the functionality and usage of the recommendation part of the Movie Recommendation System project.

## Making Movie Recommendations

The recommendation package uses the clustered user data and the user embeddings to provide movie recommendations for individual users. It evaluates the recommendations using two approaches: Pearson Correlation Coefficient (PCC) and Jaccard similarity.

### Assigning Users to Clusters

Before making recommendations, users are assigned to specific clusters based on their embeddings. The user is assigned to the cluster whose centroid is closest to their embeddings.

### Finding Similar Users

For each user, the package identifies similar users within the same cluster based on two similarity metrics:

- **Pearson Correlation Coefficient (PCC)**: A correlation-based similarity metric used to find users with similar movie preferences.

- **Jaccard Similarity**: A set-based similarity metric for finding users with similar interactions with movies.

### Predicting Movie Ratings

After finding similar users, the package predicts movie ratings for the target user. Predictions are made by considering the movie ratings of similar users in the cluster.

### Evaluating Recommendations

The recommendations are evaluated using standard evaluation metrics, including accuracy, precision, recall, and F-score. These metrics help assess the quality of recommendations made by the system.

## Example Usage

Here's an example of how to use the recommendation package to make movie recommendations for a user:

```python
import recommendation_package

# Assign the user to a cluster
cluster_id = recommendation_package.assign_user_to_cluster(trained_encoder, test_user, centroids)

# Find similar users in the same cluster
pcc_neighbors, pcc, jaccard_neighbors, jaccard = recommendation_package.find_neighbors(cluster_id, test_user, train_matrix, cluster_labels, pcc_threshold, jaccard_threshold)

# Predict movie ratings
predictions_pcc = recommendation_package.predict_ratings(test_user, pcc_neighbors, pcc)
predictions_jaccard = recommendation_package.predict_ratings(test_user, jaccard_neighbors, jaccard)

# Evaluate recommendations
results = recommendation_package.evaluate_recommendations(predictions_pcc, test_user)


## License
This script is part of the Movie Recommendation System project and is available under the MIT License.

Feel free to contribute and enhance the project!


[Next: Visualization](visualization.md)

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
