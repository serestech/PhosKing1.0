import sys, os
print(f'Python executable: {sys.executable}')
import numpy as np
import argparse
from os.path import abspath, dirname
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dataset_new import ESM_Embeddings
from kinase_mapping import kinase_mapping as mapping
from kinase_mapping import kinase_mapping_reverse as reverse_mapping
from time import sleep


parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description='Train a phosphorilation prediction model with PyTorch using ESM Embeddings')

FILE_PATH = os.path.abspath(__file__)
HERE = os.path.dirname(FILE_PATH)

parser.add_argument('-t', '--trees', action='store', type=int,
                    dest='trees', default='100',
                    help='Number of trees in the random forest')
parser.add_argument('-f', '--fasta', action='store', dest='fasta_path',
                    default=os.path.abspath(f'{HERE}/../data/kinase_data/homology_reduced/mmseqs_2023-05-07_15-20-36_rep_seq_scored.fasta'),
                    help='Fasta file')
parser.add_argument('-ft', '--features', action='store', dest='features_path',
                    default=os.path.abspath(f'{HERE}/../data/kinase_data/kinase_metadata.tsv'),
                    help='Features table (matching fasta file)')
parser.add_argument('-emb', '--embeddings_dir', action='store', dest='embeddings_dir',
                    default=os.path.abspath(f'{HERE}/../data/embeddings/embeddings_1280_kinase'),
                    help='Folder with embeddings pickles')
parser.add_argument('-aaw', '--aa_window', action='store', type=int,
                    dest='aa_window', default='1',
                    help='Amino acid window for the tensors (concatenated tensor of the 5 amino acids)')
parser.add_argument('-sd', '--small_data', action='store_true',
                    dest='small_data',
                    help='Dataset only loads 1 embedding pickle.')
parser.add_argument('-fp', '--frac_phos', action='store', type=float,
                    dest='frac_phos', default='0.5',
                    help='Fraction of phosphorilated amino acids in dataset')
parser.add_argument('-c', '--cpus', action='store', type=int,
                    dest='n_jobs', default='20',
                    help='Number of threads to use for fitting the random forest')
parser.add_argument('-q', '--quiet', action='store_true',
                    dest='quiet',
                    help='Controls verbosity of sklearn fitting')

args = parser.parse_args()

dataset = ESM_Embeddings(fasta=args.fasta_path,
                         features=args.features_path,
                         embeddings_dir=args.embeddings_dir,
                         phos_fract=float(args.frac_phos),
                         aa_window=int(args.aa_window),
                         small_data=args.small_data,
                         flatten_window=True,
                         mode='kinase')

X, y = [], []
for embeddings, labels in dataset:
    X.append(embeddings.numpy())
    y.append(labels.numpy())

X = np.vstack(X)
y = np.vstack(y)

print(f'{X.shape=}')
print(f'{y.shape=}')

# Assuming you have your input data X and multi-label target variable y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=args.trees, 
                                       n_jobs=args.n_jobs, 
                                       verbose=1 if not args.quiet else 0)

# Wrap the Random Forest classifier with MultiOutputClassifier
multi_label_classifier = MultiOutputClassifier(rf_classifier)

# Train the model
print('Training random forest...')
multi_label_classifier.fit(X_train, y_train)

sleep(3)

classes = multi_label_classifier.classes_

for class_arr in classes:
    assert tuple(class_arr) == (0, 1), f'{classes=}'


# Make predictions on the test set
y_pred_prob = multi_label_classifier.predict_proba(X_test)
y_pred_prob = np.vstack(preds[:, 1] for preds in y_pred_prob).T
print(y_pred_prob.shape)

# Calculate the AUC score for each class
auc_scores = []
for i in range(y_test.shape[1]):
    print(f'{i=}')
    print(f'{type(y_test)=} {type(y_pred_prob)=}')
    auc = roc_auc_score(y_test[:, i], y_pred_prob[:, i])
    auc_scores.append(auc)

print("AUC Scores for each class:")
for i, auc in enumerate(auc_scores):
    print(f"{reverse_mapping[i]}\t{auc:.4f}")
