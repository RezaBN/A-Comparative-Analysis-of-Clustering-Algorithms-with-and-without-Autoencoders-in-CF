[Back: Initialization](initialization.md)

# Data Preparing Package

The Data Preparing Package is responsible for reading the dataset, creating the user-item matrix, and splitting it into training and testing sets. This section explains the functionality and usage of the data preparation part of the Movie Recommendation System project.

## Functions:
### Reading the Dataset

The data preparation package allows you to read the dataset used for movie recommendations. By default, the system supports the "MovieLens_100K" dataset, but you can specify your own dataset using the `--dataset` command-line argument.

#### Supported Dataset

- **MovieLens_100K**: The default dataset provided with the system. It includes movie ratings from users.

#### Custom Dataset

If you wish to use your own dataset, please ensure it follows the appropriate structure and format. The data should contain information about user ratings, movie IDs, and user IDs. The data should be in a CSV format, and you can specify your dataset as the `--dataset` argument when running the system.

### Creating the User-Item Matrix

The system creates a user-item matrix, where rows represent users, columns represent movies, and cells contain ratings. The matrix is created from the dataset, and missing values are filled with zeros (NaN values become zeros).

### Splitting the User-Item Matrix

The user-item matrix is divided into training and testing sets for model training and evaluation. This splitting is done using scikit-learn's `train_test_split` function. The default split ratio is 80% for training and 20% for testing. You can modify this split ratio in the `test_train` function.

## Example Usage

Here's an example of how to use the data preparation package to prepare and split the dataset:

```python
import data_preparing_package

# Read the dataset (you can specify your dataset with the --dataset argument)
dataset = data_preparing_package.read_data("MovieLens_100K")

# Create the user-item matrix
user_item_matrix = data_preparing_package.create_useritem_matrix(dataset)

# Split the user-item matrix into training and testing sets
train_matrix, test_matrix = data_preparing_package.test_train(user_item_matrix)


## License
This script is part of the Movie Recommendation System project and is available under the MIT License.

Feel free to contribute and enhance the project!

[Next: Clustering](clustering_package.md)

[Go back to the main](../README.md)


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
