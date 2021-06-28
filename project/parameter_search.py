import sys
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

from project.analysis import MicrobialAbundanceAnalysis


SPECIES_ABUNDANCE_FP = '/Volumes/Data/project_cbs/abundance.txt'


LEVELS = {
    'k': 'domain',
    'p': 'phylum',
    'c': 'class',
    'o': 'order',
    'f': 'family',
    'g': 'genus',
    's': 'species',
    't': 'strain'
}

DISEASES = [
    'cirrhosis', 'colorectal_neoplasm', 'ibd',
    'impaired_glucose_tolerance', 'obesity', 't2d'
]

SEARCH_SPACE = {
    'objective': ['binary:logistic'],
    'verbosity': [0],
    'use_label_encoder': [False],
    'eta': np.linspace(0, 1, 101),
    'max_depth': list(range(2, 16)),
    'min_child_weight': list(range(1, 21)),
    'subsample': np.linspace(0.5, 1, 101),
    'gamma': np.linspace(0, 0.5, 101),
    'learning_rate': np.linspace(0.01, 0.5, 101),
    'n_estimators': list(range(60, 301)),
}


def find_parameters(analysis, disease, level, output_dir):
    (X_train, y_train), (X_test, y_test), params = analysis.prepare_dataset(
        skip_metadata=True, diseases=[disease], level=level
    )
    X = pd.concat([X_train, X_test])
    y = np.array(list(y_train) + list(y_test))

    optimizer = RandomizedSearchCV(
        xgb.XGBClassifier(
            use_label_encoder=False,
            objective='binary:logistic',
            verbosity=0,
        ),
        SEARCH_SPACE,
        scoring='f1',
        verbose=100,
        n_jobs=4,
        n_iter=50,
    )

    optimizer.fit(X, y)

    with open(os.path.join(output_dir, f'hparams_{disease}_{level}.tsv'), 'w') as handle:
        pd.DataFrame(optimizer.cv_results_).to_csv(handle, sep='\t')


if __name__ == '__main__':
    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)
    analysis = MicrobialAbundanceAnalysis.from_file(SPECIES_ABUNDANCE_FP)
    for level, level_name in LEVELS.items():
        for disease in DISEASES:
            print(f'Starting hyperparameter optimization of {disease} on level {level_name}...')
            find_parameters(analysis, disease, level, output_dir)
            print(f'Finished hyperparameter optimization of {disease} on level {level_name}.')

