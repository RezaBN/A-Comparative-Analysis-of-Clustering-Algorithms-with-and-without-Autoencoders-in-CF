
import numpy as np
import pandas as pd
import sklearn
import tensorflow
import pyclustering
import skfuzzy as fuzz
import minisom


# Autoencoder Training Function
def train_autoencoder(train_matrix, num_hidden_units=64, epochs=10, batch_size=32):
    """
    Train an autoencoder on the input data and return the encoder part for embeddings.

    Args:
    train_matrix (numpy.ndarray): The input training matrix.
    num_hidden_units (int): Number of neurons in the hidden layer (default is 64).
    epochs (int): Number of training epochs (default is 10).
    batch_size (int): Batch size for training (default is 32).

    Returns:
    tensorflow.keras.models.Model: The trained encoder model.
    """
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model

    # Define the dimensions of the user-item matrix
    num_users, num_items = train_matrix.shape

    # Define the architecture of the Autoencoder
    input_layer = Input(shape=(num_items,))
    encoded = Dense(num_hidden_units, activation='relu')(input_layer)
    decoded = Dense(num_items, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the Autoencoder
    autoencoder.fit(train_matrix, train_matrix, epochs=epochs, batch_size=batch_size)

    # Create an encoder model
    encoder = Model(input_layer, encoded)

    return encoder



# K-means clustering function
def kmeans_clustering(user_embeddings, num_clusters=3, n_init=10):
    """
    Perform K-means clustering on user embeddings and return cluster labels and centroids.

    Args:
    user_embeddings_df (pandas.DataFrame): DataFrame containing user embeddings.
    num_clusters (int): Number of clusters (default is 3).
    n_init (int): Number of times the K-means algorithm will be run with different centroid seeds (default is 10).

    Returns:
    numpy.ndarray: Cluster labels for each user.
    numpy.ndarray: Cluster centroids.
    """
    from sklearn.cluster import KMeans
    
    # Create a K-means clustering model
    kmeans = KMeans(n_clusters=num_clusters, n_init=n_init)

    # Fit the model to the user embeddings
    kmeans.fit(user_embeddings)

    # Get the cluster centroids
    centroids = kmeans.cluster_centers_

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    return cluster_labels, centroids



# K-medoids Clustering Function
def kmedoids_clustering(user_embeddings, num_clusters, initial_medoid_indexes=None):
    """
    Perform K-medoids clustering on user embeddings and return cluster labels and centroids.

    Args:
    user_embeddings_df (pandas.DataFrame): DataFrame containing user embeddings.
    num_clusters (int): Number of clusters (default is 3).
    initial_medoid_indexes (list): List of initial medoid indexes (default is None).

    Returns:
    numpy.ndarray: Cluster labels for each user.
    numpy.ndarray: Cluster centroids.
    """

    from pyclustering.cluster.kmedoids import kmedoids

    # Create a K-medoids instance with the number of clusters
    kmedoids_instance = kmedoids(user_embeddings, initial_index_medoids=initial_medoid_indexes, amount_clusters=num_clusters)

    # Run K-medoids clustering
    kmedoids_instance.process()

    # Get the medoids (centroids) indexes
    centroids_indexes = kmedoids_instance.get_medoids()

    # Extract the actual centroids from your data
    centroids = user_embeddings[centroids_indexes]

    clusters = kmedoids_instance.get_clusters()

    # Create a list to store cluster labels for each user
    cluster_labels = [-1] * user_embeddings.shape[0]

    # Assign cluster labels to users in the DataFrame
    for cluster_index, cluster in enumerate(clusters):
        for user_index in cluster:
            cluster_labels[user_index] = cluster_index

    return cluster_labels, centroids



# DBSCAN Clustering Function
def dbscan_clustering(user_embeddings, eps=9, min_samples=10):
    """
    Perform DBSCAN clustering on user embeddings and return cluster labels and centroids.

    Args:
    user_embeddings (numpy.ndarray): Array containing user embeddings.
    eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other (default is 9).
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point (default is 10).

    Returns:
    numpy.ndarray: Cluster labels for each user.
    numpy.ndarray: Cluster centroids.
    """
    from sklearn.cluster import DBSCAN
    
    # Create a DBSCAN clustering model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model to the user embeddings and obtain cluster labels
    dbscan_labels = dbscan.fit_predict(user_embeddings)

    # Filter the data to exclude noise points (optional)
#    core_points_mask = dbscan_labels != -1
#    clustered_data = user_embeddings[core_points_mask]

    # Calculate cluster centroids as the mean of each cluster
    unique_labels = np.unique(dbscan_labels)
    cluster_centroids = []

    for label in unique_labels:
#        if label == -1:  # Skip noise points
#            continue
        cluster_points = user_embeddings[dbscan_labels == label]
        cluster_centroid = np.mean(cluster_points, axis=0)
        cluster_centroids.append(cluster_centroid)

    # Return cluster labels and centroids
    return dbscan_labels, np.array(cluster_centroids)



# SOM Clustering Function
def train_som_and_get_clusters(user_embeddings, som_shape=(1, 3), num_iterations=500):
    """
    Train a Self-Organizing Map (SOM) on the input user_embeddings and return cluster labels.

    Parameters:
    - user_embeddings: Input data as a NumPy array where each row is a data point.
    - som_shape: Shape of the SOM grid (default is a 1x3 grid).
    - num_iterations: Number of SOM training iterations (default is 500).

    Returns:
    - cluster_labels: List of cluster labels, one for each data point.
    """

    from minisom import MiniSom
    
    # Initialization and training
    som = MiniSom(som_shape[0], som_shape[1], user_embeddings.shape[1], sigma=0.5, learning_rate=0.5,
                  neighborhood_function='gaussian', random_seed=10)
    
    som.train_batch(user_embeddings, num_iterations, verbose=True)

    # Find the cluster for each data point
    winner_coordinates = np.array([som.winner(x) for x in user_embeddings]).T

    # Convert coordinates to cluster indices
    cluster_labels = np.ravel_multi_index(winner_coordinates, som_shape)

    # Get the cluster centroids
    unique_clusters = np.unique(cluster_labels)
    centroids = [np.mean(user_embeddings[np.array(cluster_labels) == cluster], axis=0) for cluster in unique_clusters]

    return cluster_labels, centroids



# Fuzzy C-means Clustering Function
def perform_fcm_clustering(user_embeddings, num_clusters=3, fuzziness=2.0, error=0.005, max_iterations=1000):
    """
    Perform Fuzzy C-Means (FCM) clustering on the input user_embeddings.

    Parameters:
    - user_embeddings: Input data as a NumPy array where each row is a data point.
    - num_clusters: Number of clusters (default is 3).
    - fuzziness: Fuzziness parameter (typically set to 2.0) (default is 2.0).
    - error: Error tolerance to stop the FCM algorithm (default is 0.005).
    - max_iterations: Maximum number of iterations (default is 1000).

    Returns:
    - cluster_labels: List of cluster labels, one for each data point.
    - centroids: List of cluster centroids.
    """

    # Perform FCM clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(user_embeddings.T, num_clusters, fuzziness, error=error, maxiter=max_iterations, init=None)

    # Find the cluster with the highest membership for each data point
    cluster_labels = np.argmax(u, axis=0)

    # Get cluster centroids
    centroids = cntr

    return cluster_labels, centroids



# The main function for evaluating performed clusters by Davies Bouldin Index and Dunn Index
def evaluate_clusters(clustered_data, labels, centroids):
    
    from sklearn.metrics import pairwise_distances
    from scipy.spatial import distance

    # Calculate the Davies-Bouldin (DB) index
    def davies_bouldin_index(data, labels, centroids):
        num_clusters = len(centroids)
        distances = pairwise_distances(data, centroids, metric='euclidean')

        db_indices = []
        labels = np.array(labels)
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                continue  # Skip empty clusters
                
            cluster_points = data[labels == i]
            centroid = centroids[i]
            cluster_distance = distances[labels == i, i]  # Distances for cluster i
            avg_distance = np.mean(cluster_distance)

            max_similarity = 0
            for j in range(num_clusters):
                if i != j:
                    other_cluster_indices = np.where(labels == j)[0]
                    if len(other_cluster_indices) == 0:
                        continue  # Skip empty clusters
                    other_cluster_distance = distances[labels == j, j]  # Distances for cluster j
                    separation = distance.euclidean(centroid, centroids[j])
                    similarity = (avg_distance + np.mean(other_cluster_distance)) / separation
                    if similarity > max_similarity:
                        max_similarity = similarity

            db_indices.append(max_similarity)

        return np.mean(db_indices)

    # Calculate the Dunn index
    def dunn_index(data, labels, centroids):
        num_clusters = len(np.unique(labels))
        max_intra_cluster_distances = np.zeros(num_clusters)
        labels = np.array(labels)
        
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                continue  # Skip empty clusters
                
            cluster_points = data[labels == i]
            max_intra_cluster_distance = np.max(pairwise_distances(cluster_points, metric='euclidean'))
            max_intra_cluster_distances[i] = max_intra_cluster_distance

        # Calculate pairwise distances between centroids
        distances = pairwise_distances(centroids, metric='euclidean')

        # Replace 0 distances with a large value (e.g., infinity) to disregard them
        distances[distances == 0] = np.inf
        
        # Find the minimum inter-cluster distance
        min_inter_cluster_distance = np.min(distances)

        dunn_index_value = min_inter_cluster_distance / np.max(max_intra_cluster_distances)
        return dunn_index_value


    # performing DB index and Dunn index
    db_index_value = davies_bouldin_index(clustered_data, labels, centroids)
    dunn_index_value = dunn_index(clustered_data, labels, centroids)
    
    return db_index_value, dunn_index_value


