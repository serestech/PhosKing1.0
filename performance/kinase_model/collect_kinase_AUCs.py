
from statistics import mean, median
from pprint import pprint

fold_timestamps = [
    'CNN_FFNN_07_06_03_29_05',
    'CNN_FFNN_07_06_03_33_38',
    'CNN_FFNN_07_06_03_37_49',
    'CNN_FFNN_07_06_03_42_50',
    'CNN_FFNN_07_06_03_39_34',
    'CNN_FFNN_07_06_03_43_43',
    'CNN_FFNN_07_06_03_48_14',
    'CNN_FFNN_07_06_03_53_06',
    'CNN_FFNN_07_06_03_48_49',
    'CNN_FFNN_07_06_03_53_04',
]

models_folder = '/work3/s220260/PhosKing1.0/training/models'

fold_results = []
for fold_timestamp in fold_timestamps:
    with open(f'{models_folder}/{fold_timestamp}_kin.kinase_results', 'r') as results_file:
        results_file_contents = results_file.read()
    fold_result = eval(results_file_contents)
    fold_results.append(fold_result)

kinases = list(fold_results[0].keys())
kinase_average_AUCs = {}
for kinase in kinases:
    kinase_AUCs = [fold_result[kinase] for fold_result in fold_results]
    kinase_average_AUCs[kinase] = mean(kinase_AUCs)

pprint(kinase_average_AUCs)

average_global_AUC = mean(kinase_average_AUCs.values())

print(f'AUC of across kinases: mean {mean(kinase_average_AUCs.values()):.4f} median {median(kinase_average_AUCs.values()):.4f}')

median_AUCs = [median(fold_result.values()) for fold_result in fold_results]
best_median = max(range(10), key=lambda i: median_AUCs[i])

print(f'Fold with best median AUC: {best_median + 1} ({fold_timestamps[best_median]})')

# make table for plotting
table_file = open('kinase_evaluation.csv', 'w')
table_file.write('fold,kinase,auc\n')
for i, fold_result in enumerate(fold_results):
    for kinase, auc in fold_result.items():
        table_file.write(f'{i+1},{kinase},{auc}\n')


