import matplotlib.pyplot as plt
import os

def visualize_results(results, args):
    # Visualization code here
    result_folder = 'results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Create and save Accuracy visualization
    plt.figure()
    plt.bar(["PCC Accuracy", "Jaccard Accuracy"], [results['PCC_Accuracy'], results['Jaccard_Accuracy']])
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.title("Prediction Accuracy")
    plt.savefig(os.path.join(result_folder, f'prediction_accuracy_{args.clustering_method}.png'))

    # Create and save Precision visualization
    plt.figure()
    plt.pie([results['PCC_Precision'], results['Jaccard_Precision']], labels=["PCC Precision", "Jaccard Precision"], autopct='%1.1f%%')
    plt.title("Prediction Precision")
    plt.savefig(os.path.join(result_folder, f'prediction_precision_{args.clustering_method}.png'))

    # Create and save F-Score visualization
    plt.figure()
    plt.bar(["PCC F-Score", "Jaccard F-Score"], [results['PCC_FScore'], results['Jaccard_FScore']])
    plt.xlabel("Method")
    plt.ylabel("F-Score")
    plt.title("Prediction F-Score")
    plt.savefig(os.path.join(result_folder, f'prediction_fscore_{args.clustering_method}.png'))

    plt.show()
    plt.close()