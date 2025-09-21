import argparse


# Define command-line arguments
parser = argparse.ArgumentParser(description="Movie Recommendation System")
parser.add_argument("--dataset", choices=["MovieLens_100K", "Your_Dataset"], default="MovieLens_100K", help="Choose a dataset")
parser.add_argument("--hidden_units", type=int, default=64, help="Number of hidden units in the autoencoder")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for the autoencoder")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training the autoencoder")
parser.add_argument("--num_clusters", type=int, default=3, help="Number of clusters for clustering")
parser.add_argument("--pcc_threshold", type=float, default=0.3, help="Threshold for PCC-based recommendation")
parser.add_argument("--jaccard_threshold", type=float, default=0.16, help="Threshold for Jaccard-based recommendation")
parser.add_argument("--clustering_method", choices=["kmeans", "kmedoids", "dbscan", "som", "fcm"], default="dbscan", help="Choose a clustering method")
