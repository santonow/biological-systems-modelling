from itertools import chain

disease_to_target = {
    'n': 'healthy',
    'obesity': 'obesity',
    'leaness': 'healthy',
    'cirrhosis': 'cirrhosis',
    'cancer': 'colorectal_neoplasm',
    'large_adenoma': 'colorectal_neoplasm',
    'small_adenoma': 'colorectal_neoplasm',
    'ibd_ulcerative_colitis': 'ibd',
    'ibd_crohn_disease': 'ibd',
    't2d': 't2d',
    'impaired_glucose_tolerance': 'impaired_glucose_tolerance',
}

disease_to_dataset = {
    'healthy': ['hmp', 'hmpii'],
    't2d': ['WT2D', 't2dmeta_long', 't2dmeta_short'],
    'cirrhosis': ['Quin_gut_liver_cirrhosis'],
    'colorectal_neoplasm': ['Zeller_fecal_colorectal_cancer'],
    'obesity': ['Chatelier_gut_obesity'],
    'ibd': ['Neilsen_genome_assembly', 'metahit'],
    'impaired_glucose_tolerance': ['WT2D']
}

chosen_datasets = set(chain(*[datasets for datasets in disease_to_dataset.values()]))

DEFAULT_PARAMS = {
    'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic',
    'nthread': 4, 'eval_metric': 'auc'
}
