
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cdist

def assign_user_to_cluster(trained_encoder, test_user, centroids):
    encoded_test_user = trained_encoder.predict(test_user.reshape(1, -1))
    distances = np.linalg.norm(centroids - encoded_test_user)
    return np.argmin(distances)

def find_neighbors(cluster_id, test_user, train_matrix, cluster_labels, pcc_threshold, jaccard_threshold):

    cluster_indices = np.where(cluster_labels == cluster_id)
    train_matrix = np.array(train_matrix)
    train_cluster = train_matrix[cluster_indices]
    
    pcc = [np.corrcoef(test_user, train_row)[0, 1] for train_row in train_cluster]
    jaccard = [jaccard_score(test_user, train_row, average='macro') for train_row in train_cluster]
    
    ppc_similar_indices = np.where(np.array(pcc) >= pcc_threshold)
    jaccard_similar_indices = np.where(np.array(jaccard) >= jaccard_threshold)
    
    pcc_neighbors = train_matrix[ppc_similar_indices]
    pcc = np.array(pcc)[ppc_similar_indices]
    
    jaccard_neighbors = train_matrix[jaccard_similar_indices]
    jaccard = np.array(jaccard)[jaccard_similar_indices]
    
    return pcc_neighbors, pcc, jaccard_neighbors, jaccard

def predict_ratings(test_user, neighbors, similarities):
    
    ''' Note: Scince we use the prediction to evaluate the method here, only predict for existed rated items and 
        then evaluate prediction with the actual ratings. 
        This also help to reduce the memory requairement and run time. '''
    
    rated_indeces = np.where(test_user != 0)[0]
    num_items = len(rated_indeces)
    numerator_of_prediction = np.zeros(num_items)
    numerator_of_avg_prediction = np.zeros(num_items)
    rated_count = np.zeros(num_items)
    test_user_avg_rating = np.mean(test_user[rated_indeces])
    
    for neighbor, neighbor_ratings in enumerate(neighbors):

        neighbor_weight = similarities[neighbor]
        neighbor_avg_rating = np.mean(neighbor_ratings[neighbor_ratings != 0])
        
        for i, k in enumerate(rated_indeces):
            
            if neighbor_ratings[k] > 0:

                numerator_of_prediction[i] += neighbor_weight * (neighbor_ratings[k] - neighbor_avg_rating)
                numerator_of_avg_prediction[i] += neighbor_ratings[k]
                rated_count[i] += 1
        
    predicted_ratings = test_user_avg_rating + np.divide(numerator_of_prediction, similarities.sum(), out=np.zeros_like(numerator_of_prediction), where=similarities.sum() != 0)
    predicted_ratings[np.isnan(predicted_ratings)] = test_user_avg_rating

    return predicted_ratings

def evaluate_recommendations(predictions, test_user, threshold=4):
    
    rated_indeces = np.where(test_user != 0)[0]
    actual_ratings = test_user[rated_indeces]
    
    positive_mask = (actual_ratings >= threshold) & (predictions >= threshold)
    negative_mask = (actual_ratings < threshold) & (predictions < threshold)
    
    TP = np.sum(positive_mask)
    FP = np.sum(predictions >= threshold) - TP
    TN = np.sum(negative_mask)
    FN = np.sum(predictions < threshold) - TN
    
    accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
    precision = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0
    recall = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
    
    return accuracy, precision, recall

def recommendation_system(trained_encoder, train_matrix, test_matrix, centroids, cluster_labels, pcc_threshold, jaccard_threshold):
    
    num_users, num_items = test_matrix.shape
    test_matrix = np.array(test_matrix)
    pcc_accuracies = []
    jaccard_accuracies = []
    pcc_precisions = []
    jaccard_precisions = []
    pcc_recalls = []
    jaccard_recalls = []
    
    for i in range(num_users):
        test_user = test_matrix[i]
        
        cluster_id = assign_user_to_cluster(trained_encoder, test_user, centroids)
        pcc_neighbors, pcc, jaccard_neighbors, jaccard = find_neighbors(cluster_id, test_user, train_matrix, cluster_labels, pcc_threshold, jaccard_threshold)
        
        predictions_pcc = predict_ratings(test_user, pcc_neighbors, pcc)
        predictions_jaccard = predict_ratings(test_user, jaccard_neighbors, jaccard)

        #pcc_predicted_ratings.append(predictions_pcc)
        #jaccard_predicted_ratings.append(predictions_jaccard)
        
        pcc_accuracy, pcc_precision, pcc_recall = evaluate_recommendations(predictions_pcc, test_user)
        jaccard_accuracy, jaccard_precision, jaccard_recall = evaluate_recommendations(predictions_jaccard, test_user)
        
        pcc_accuracies.append(pcc_accuracy)
        jaccard_accuracies.append(jaccard_accuracy)
        pcc_precisions.append(pcc_precision)
        jaccard_precisions.append(jaccard_precision)
        pcc_recalls.append(pcc_recall)
        jaccard_recalls.append(jaccard_recall)
    
    return {
        'PCC_Accuracy': np.mean(pcc_accuracies),
        'Jaccard_Accuracy': np.mean(jaccard_accuracies),
        'PCC_Precision': np.mean(pcc_precisions),
        'Jaccard_Precision': np.mean(jaccard_precisions),
        'PCC_Recall': np.mean(pcc_recalls),
        'Jaccard_Recall': np.mean(jaccard_recalls),
        'PCC_FScore': (2 * np.mean(pcc_recalls) * np.mean(pcc_precisions)) / (np.mean(pcc_recalls) + np.mean(pcc_precisions)),
        'Jaccard_FScore': (2 * np.mean(jaccard_recalls) * np.mean(jaccard_precisions)) / (np.mean(jaccard_recalls) + np.mean(jaccard_precisions))
    }

