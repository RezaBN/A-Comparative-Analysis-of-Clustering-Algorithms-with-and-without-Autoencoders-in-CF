[Back: Recommendation](recommendation_package.md)

# Visualization

The Visualization part of the Movie Recommendation System project focuses on creating visual representations of the results, metrics, and recommendations. In this section, we'll explore the purpose of visualization and provide examples of how to generate and save visualizations for analysis.

## Purpose of Visualization

The primary purpose of visualization is to present the results and performance metrics of the recommendation system in an easily understandable and visually appealing format. Visualization helps in:

- Communicating the quality of recommendations to stakeholders.
- Comparing different clustering and recommendation methods.
- Identifying trends and patterns in recommendation accuracy and precision.

## Visualization Examples

### Prediction Accuracy Visualization

One of the key visualizations created is the prediction accuracy visualization. It provides a comparison between the accuracy of two recommendation methods: PCC (Pearson Correlation Coefficient) and Jaccard similarity.

![Prediction Accuracy](results/prediction_accuracy_method.png)

### Prediction Precision Visualization

The prediction precision visualization represents the precision of recommendations made by both PCC and Jaccard similarity-based methods.

![Prediction Precision](results/prediction_precision_method.png)

### F-Score Visualization

The F-Score visualization offers a comparison of the F-scores for PCC and Jaccard similarity-based recommendations. The F-Score combines precision and recall into a single metric.

![F-Score](results/prediction_fscore_method.png)

## Generating Visualizations

You can generate visualizations using Python libraries like Matplotlib. The `visualize_results` function in the project code takes care of creating and saving visualizations. The saved images are stored in the "results" folder for easy access and sharing.

Here's an example of how to use the `visualize_results` function:

```python
import visualization

# Example results to visualize (replace with actual results)
results = {
    'PCC_Accuracy': 0.85,
    'Jaccard_Accuracy': 0.78,
    'PCC_Precision': 0.82,
    'Jaccard_Precision': 0.75,
    'PCC_FScore': 0.83,
    'Jaccard_FScore': 0.76
}

# Call the visualize_results function
visualization.visualize_results(results, args)


## License
This script is part of the Movie Recommendation System project and is available under the MIT License.

Feel free to contribute and enhance the project!


[Next: Main Script](main.md)

[Back to the Main Page](../README.md)


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
