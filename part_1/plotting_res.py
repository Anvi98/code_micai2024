import matplotlib.pyplot as plt

# Data organized as a list of dictionaries
data = [
    {
        'model': 'Logistic Regression',
        'f1_score': [0.5859, 0.55, 0.5378, 0.5408],
        'spearmanr': [0.1288, 0.1151, 0.1135, 0.1142],
        'best_param': 'Clf_C:1',
        'dataset_size': [1927, 3854, 7709, 11564]
    },
    {
        'model': 'Naive Bayes',
        'f1_score': [0.5252, 0.5177, 0.5413, 0.523],
        'spearmanr': [0.0236, 0.0321, 0.0412, 0.0403],
        'best_param': '_',
        'dataset_size': [1927, 3854, 7709, 11564]
    },
    {
        'model': 'SVM',
        'f1_score': [0.57, 0.5286, 0.5501, 0.5464],
        'spearmanr': [0.1428, 0.112, 0.1072, 0.1179],
        'best_param': 'Clf_c:1, kernel:linear',
        'dataset_size': [1927, 3854, 7709, 11564]
    },
    {
        'model': 'Random Forest',
        'f1_score': [0.9175, 0.9024, 0.8855, 0.8546],
        'spearmanr': [0.8318, 0.8048, 0.731, 0.7094],
        'best_param': 'n_estimators:200',
        'dataset_size': [1927, 3854, 7709, 11564]
    }
]
# Plot and save F1-score plot
plt.figure(figsize=(10, 6))
for entry in data:
    model_name = entry['model']
    plt.plot(entry['dataset_size'], entry['f1_score'], marker='o', label=model_name)
plt.xlabel('Dataset Size')
plt.ylabel('F1-score')
plt.title('Performance Evolution: F1-score')
plt.legend()
plt.tight_layout()
plt.savefig('f1_score_evolution.png')
plt.close()

# Plot and save Spearman correlation plot
plt.figure(figsize=(10, 6))
for entry in data:
    model_name = entry['model']
    plt.plot(entry['dataset_size'], entry['spearmanr'], marker='o', label=model_name)
plt.xlabel('Dataset Size')
plt.ylabel('Spearman correlation')
plt.title('Performance Evolution: Spearman correlation')
plt.legend()
plt.tight_layout()
plt.savefig('spearmanr_evolution.png')
plt.close()

