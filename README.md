# Autoencoder & Clustering Combination based Recommender

Welcome to the Movie Recommendation System project! This system uses autoencoders for feature learning, clustering for grouping users, and collaborative filtering for movie recommendations.

This repository implements the algorithms described in the paper "**Improving Collaborative Filtering Performance: A Comparative Analysis of Clustering Algorithms with and without Autoencoders**" which is published in Multimedia Tools and Applications.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Initialization](/Scripts%20Description/initialization.md)
- [Data Preparation](/Scripts%20Description/data_preparing_package.md)
- [Clustering](/Scripts%20Description/clustering_package.md)
- [Recommendation](/Scripts%20Description/recommendation_package.md)
- [Visualization](/Scripts%20Description/visualization.md)
- [Main Script](/Scripts%20Description/main.md)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Further Information](#further-information)

## Project Structure

This project is organized into separate modules to improve modularity and readability:

- **Initialization**: Contains the command-line argument setup and project initialization. [More Details](/Scripts%20Description/initialization.md)
- **Data Preparation**: Handles data loading, processing, and user-item matrix creation. [More Details](/Scripts%20Description/data_preparing_package.md)
- **Clustering**: Implements various clustering methods to group users into clusters. [More Details](/Scripts%20Description/clustering_package.md)
- **Recommendation**: Includes recommendation algorithms and evaluation. [More Details](/Scripts%20Description/recommendation_package.md)
- **Visualization**: Provides visualization of the results. [More Details](/Scripts%20Description/visualization.md)
- **Main**: Orchestrates the entire process and presents results. [More Details](/Scripts%20Description/main.md)


## Initialization (README)

The Initialization script defines command-line arguments and configurations for the project. These settings allow you to customize the dataset, autoencoder parameters, clustering method, and more.

- [Detailed Initialization README](/Scripts%20Description/initialization.md)

## Data Preparing Package (README)

The Data Preparing Package contains functions for reading and preparing the dataset, creating the user-item matrix, and splitting it into training and testing sets.

- [Detailed Data Preparing README](/Scripts%20Description/data_preparing_package.md)

## Clustering Package (README)

The Clustering Package implements various clustering methods, including K-means, K-medoids, DBSCAN, SOM, and FCM. It also provides functions for evaluating clustering results.

- [Detailed Clustering Package README](/Scripts%20Description/clustering_package.md)

## Recommendation Package (README)

The Recommendation Package focuses on recommending movies based on user preferences. It offers functions for assigning users to clusters, finding neighbors, predicting ratings, and evaluating recommendations.

- [Detailed Recommendation Package README](/Scripts%20Description/recommendation_package.md)

## Visualization (README)

The Visualization script generates visualizations for clustering and prediction results, saving them in a 'results' folder.

- [Detailed Visualization README](/Scripts%20Description/visualization.md)

## Main Script (README)

The main script orchestrates various components, including data preparation, clustering, recommendation, and visualization. It reads the dataset, trains an autoencoder, performs clustering, and evaluates clusters using Davies-Bouldin and Dunn indices. It also runs the recommendation system and generates visualizations.

- [Detailed Main Script README](/Scripts%20Description/main.md)


## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine.
2. Create [Virtualenv](https://virtualenv.pypa.io/en/latest/index.html).
3. Ensure you have the required dependencies installed. You can install the dependencies by running:

   ```bash
   pip install -r requirements.txt

4. Run the `main.py`

5. Review the results printed and visualized.

Follow the instructions in the relevant READMEs to execute different parts of the project:

- Initialization: Set up project configuration and command-line arguments.
- Data Preparation: Load and preprocess the dataset.
- Clustering: Perform clustering using different methods.
- Recommendation: Run movie recommendations.
- Visualization: Generate visualizations for results.

## Contributing
We welcome contributions from the community. If you'd like to contribute to this project, please follow our contribution guidelines.

## License

This project is available under the MIT License. Feel free to contribute and enhance the system!

- [License](LICENSE)


[Next: Initialization](/Scripts%20Description/initialization.md)

# Citation

If you use this Movie Recommendation System project in your research or as a part of your work, we kindly request that you cite it using the following BibTeX entry:

bibtex
@misc{BarzegarNozari2025,
  title = {Autoencoder & Clustering Combination based Recommender},
  author = {Barzegar Nozari, Reza},
  year = {2025},
  howpublished = {\url{[https://github.com/RezaBN/Autoencoder & Clustering Combination based Recommender](https://github.com/RezaBN/A-Comparative-Analysis-of-Clustering-Algorithms-with-and-without-Autoencoders-in-CF)}},
  note = {Accessed: Insert Access Date}
}

# Further Information

If you'd like to learn more about the technologies and concepts used in the Movie Recommendation System project, the following resources can be helpful:

1. **MovieLens Dataset:** The official MovieLens dataset can be accessed on their [website](https://grouplens.org/datasets/movielens/).

2. **Scikit-learn Documentation:** For more information on the Scikit-learn library and machine learning concepts, visit the [Scikit-learn documentation](https://scikit-learn.org/stable/documentation.html).

3. **TensorFlow Documentation:** Explore TensorFlow, an open-source machine learning framework, at the [official TensorFlow documentation](https://www.tensorflow.org/).

4. **Matplotlib Documentation:** For detailed information on creating visualizations with Matplotlib, check out the [Matplotlib documentation](https://matplotlib.org/stable/contents.html).

5. **PyClustering Documentation:** To dive deeper into clustering algorithms, visit the [PyClustering documentation](https://pyclustering.github.io/).

6. **Fuzzy Clustering with Scikit-Fuzzy:** Explore fuzzy clustering with Scikit-Fuzzy by referring to their [documentation](https://pythonhosted.org/scikit-fuzzy/).

7. **MiniSom Documentation:** For Self-Organizing Maps (SOM), you can find the MiniSom library documentation [here](https://github.com/JustGlowing/minisom).

8. **Davies-Bouldin Index:** Learn more about the Davies-Bouldin Index used for cluster evaluation [here](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index).

9. **Dunn Index:** For information on the Dunn Index, refer to the [Dunn index Wikipedia page](https://en.wikipedia.org/wiki/Dunn_index).

10. **Jaccard Index:** Understand the Jaccard Index used in similarity calculations [here](https://en.wikipedia.org/wiki/Jaccard_index).

11. **Pearson Correlation Coefficient (PCC):** The Pearson Correlation Coefficient, often used in recommendation systems, is explained [here](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

12. **Autoencoders:** Learn about autoencoders, a type of neural network used in this project, in this [Autoencoder guide](https://blog.keras.io/building-autoencoders-in-keras.html).


Feel free to explore these resources to gain a deeper understanding of the technologies, libraries, and concepts used in this project.

[Next: Initialization](/Scripts%20Description/initialization.md)










