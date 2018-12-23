import pickle
import random
import sys
import os

import numpy as np

from main_logic.semLTR import perform_experiments, load_and_merge_trec_qrels
from main_logic.analysis import analyze_results
path_to_this = os.getcwd()
folder_path = os.path.dirname(path_to_this)

sys.path.extend([folder_path])

# fix the random seed
random.seed(42)
np.random.seed(42)
# load up TREC qrels
qrels = load_and_merge_trec_qrels()

# run the experiments
results = perform_experiments(qrels)

analysis_dict = analyze_results(results)

for key in analysis_dict:
    print(key)
    print(analysis_dict[key])