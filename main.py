import numpy as np
import pandas as pd
import data_preparing_package
import clustering_package
import recommendation_package
import xlsxwriter
import os


def main():
    # get initial arguments from parser imported from initialization
    ''' 
    Note: You can change the initial arguments in the initialization. Specifing Autoencoder parameters, dataset, 
          number of desierd clusters, clusteering method kind, and so on.
    '''
    from initialization import parser
    args = parser.parse_args()
    
    # Read data from the specified dataset
    df = data_preparing_package.read_data(args.dataset)

    # Create the user-item matrix
    user_item_matrix = data_preparing_package.create_useritem_matrix(df)

    # Split the user-item matrix into training and testing sets
    train_matrix, test_matrix = data_preparing_package.test_train(user_item_matrix)

    # Train an autoencoder to obtain user embeddings
    trained_encoder = clustering_package.train_autoencoder(train_matrix, num_hidden_units=args.hidden_units, epochs=args.epochs, batch_size=args.batch_size)
    user_embeddings = trained_encoder.predict(train_matrix)

    # Perform clustering with the selected method and user-specified number of clusters
    if args.clustering_method == "kmeans":
        cluster_labels, centroids = clustering_package.kmeans_clustering(user_embeddings, num_clusters=args.num_clusters, n_init=10)
    elif args.clustering_method == "kmedoids":
        cluster_labels, centroids = clustering_package.kmedoids_clustering(user_embeddings, num_clusters=args.num_clusters, initial_medoid_indexes=[i + 0 for i in range(args.num_clusters)])
    elif args.clustering_method == "dbscan":
        cluster_labels, centroids = clustering_package.dbscan_clustering(user_embeddings, eps=9, min_samples=10)
    elif args.clustering_method == "som":
        cluster_labels, centroids = clustering_package.train_som_and_get_clusters(user_embeddings)
    elif args.clustering_method == "fcm":
        cluster_labels, centroids = clustering_package.perform_fcm_clustering(user_embeddings, num_clusters=args.num_clusters)

    # Evaluate the clusters using Davies-Bouldin and Dunn indices
    db_index, dunn_index = clustering_package.evaluate_clusters(user_embeddings, cluster_labels, centroids)

    # Run the recommendation system with user-specified thresholds
    results = recommendation_package.recommendation_system(trained_encoder, train_matrix, test_matrix, centroids, cluster_labels, args.pcc_threshold, args.jaccard_threshold)
    
    
    # Saving the results in an excel file:
    # Create DataFrames for clustering and prediction results
    clustering_results_df = pd.DataFrame({
        'Method': [args.clustering_method],
        'Davies-Bouldin Index': [db_index],
        'Dunn Index': [dunn_index]
    })

    prediction_results_df = pd.DataFrame(results, index=[0])
    prediction_results_df.insert(0, 'Method', args.clustering_method)


    # Ensure the result folder exists
    result_folder = 'results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
        
    # Define the output Excel file path inside the result folder
    output_file = os.path.join(result_folder, f'recommendation_results_{args.clustering_method}.xlsx')

    # Create an Excel writer object
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Write clustering results to a sheet
        clustering_results_df.to_excel(writer, sheet_name='Clustering Results', index=False)

        # Write prediction results to a sheet
        prediction_results_df.to_excel(writer, sheet_name='Prediction Results', index=False)

        
    
    # Printing and Visualizing Results:
    print('Clustering evaluation results:')
    print('Davies-Bouldin Index:', db_index)
    print('Dunn Index:', dunn_index)
    
    print(' ')
    
    print('Prediction evaluation results:')
    print("PCC-based Prediction Accuracy:", results['PCC_Accuracy'])
    print("Jaccard-based Prediction Accuracy:", results['Jaccard_Accuracy'])
    print("PCC-based Prediction Precision:", results['PCC_Precision'])
    print("Jaccard-based Prediction Precision:", results['Jaccard_Precision'])
    print("PCC-based Prediction Recall:", results['PCC_Recall'])
    print("Jaccard-based Prediction Recall:", results['Jaccard_Recall'])
    print("PCC-based Prediction F_Score:", results['PCC_FScore'])
    print("Jaccard-based Prediction F_Score:", results['Jaccard_FScore'])
    
    
    # visualizing results and saving (e.g., a bar chart)
    from visualization import visualize_results
    visualize_results(results, args)
     
    
if __name__ == '__main__':
    main()

