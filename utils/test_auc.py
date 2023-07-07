import sys
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# py ../../utils/test_auc.py ../../data/train_data/features.tsv pre_model1_test.tsv Test_MusiteDeep.tsv Test_NetPhos3.1.tsv Test_NetPhospan1.0.tsv

if len(sys.argv) < 3:
    print(f'Usage: {sys.argv[0]} <features.tsv> <results1.tsv> [<results2.tsv>] ...')
    sys.exit(1)
else:
    features_file_name = sys.argv[1]
    results_file_names = sys.argv[2:]


fig, axs = plt.subplots(2, figsize=(6,12))

# output_file_name = 'areas_1000.pdf'
# plot_labs = ['PhosKing', 'MusiteDeep', 'NetPhospan1.0', 'NetPhos3.1']
output_file_name = 'areas_.pdf'
plot_labs = ['50% phos', '20% phos']

for i, results_file_name in enumerate(results_file_names):
    results = {}
    with open(results_file_name, 'r') as results_file:
        for line in results_file:
            ID, pos, _, value = line.strip().split('\t')
            if plot_labs[i] == 'NetPhospan1.0':
                pos = str(int(pos) + 10)
            if not results.get(ID):
                results[ID] = {}
            results[ID][pos] = [float(value), 0]

    s = 0
    with open(features_file_name, 'r') as features_file:
        for line in features_file:
            if not line.startswith('#'):
                ID, pos, aa, _, _, _ = line.strip().split('\t')
                if ID in results.keys():
                    try: results[ID][pos][1] = 1
                    except: s += 1

    xy = np.array([vals for ID in results.keys() for vals in results[ID].values()])

    print(i, xy.shape, s)

    res = xy[:, 0]
    labs = xy[:, 1]

    roc_fpr, roc_tpr, roc_thresholds = roc_curve(labs, res)
    roc_auc = roc_auc_score(labs, res)

    pr_prec = []
    pr_recall = []
    pr_auc = 0
    for thr in np.linspace(0, 1, 2501):
        p = res >= thr
        n = np.logical_not(p)
        t = labs
        f = np.logical_not(t)

        tp = np.count_nonzero(np.logical_and(t, p))
        tn = np.count_nonzero(np.logical_and(f, n))
        fp = np.count_nonzero(np.logical_and(f, p))
        fn = np.count_nonzero(np.logical_and(t, n))

        try:
            pr_prec.append(tp/(tp+fp))
        except:
            pr_prec.append(1)
        pr_recall.append(tp/(tp+fn))

        if thr > 0:
            pr_auc += (pr_prec[-1] + pr_prec[-2]) * abs(pr_recall[-1] - pr_recall[-2]) / 2

    axs[0].plot(roc_fpr, roc_tpr, label=plot_labs[i]+f' (AUROC={round(roc_auc, 3)})')
    axs[1].plot(pr_recall, pr_prec, label=plot_labs[i]+f' (AUPRC={round(pr_auc, 3)})')


axs[0].set_xlabel('FPR (1 - specificity)')
axs[0].set_ylabel('TPR (sensitivity or recall)')
axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')
axs[0].legend()
axs[1].legend()

plt.savefig(output_file_name)


