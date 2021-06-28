from collections import defaultdict
from itertools import chain
from typing import Optional, Iterable, List

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from project.constants import disease_to_target, chosen_datasets, DEFAULT_PARAMS, disease_to_dataset


class MicrobialAbundanceAnalysis:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.classifier = None
        self.le = None
        self.metadata_cols = [
            'disease',
            'bodysite',
            'country',
            'sequencing_technology',
            'total_reads',
            'matched_reads',
            'gene_number',
        ]
        self.species_cols: List[str] = [
            colname for colname in df.columns.tolist()
            if colname.startswith('k_')
        ]
        self.models = list()
        self.tax_to_anc = defaultdict(dict)
        for sp in sorted(self.species_cols, key=len, reverse=True):
            tax_list = sp.split('|')
            for i, elem in enumerate(tax_list):
                self.tax_to_anc[sp][elem.split('__')[0]] = '|'.join(tax_list[:(i + 1)])
        self.current_model = None
        self.current_data = None
        self.current_params = None

    @classmethod
    def from_file(cls, fname: str):
        df = pd.read_csv(
            fname, sep="\t", dtype=object,
            header=None, na_values='nd'
        ).T

        header = df.iloc[0]
        df = df[1:]
        df.columns = header

        return cls(df)

    def prepare_dataset(
            self, diseases: Optional[List[str]] = None,
            skip_metadata=False, test_size=0.2, seed=42,
            level='t'
    ):
        df_prep = self.df.copy()
        df_prep = df_prep[self.species_cols].astype('float64')
        df_prep = pd.concat([self.df[self.metadata_cols + ['dataset_name']], df_prep], axis=1)
        if 'age' in self.metadata_cols:
            df_prep['age'] = pd.to_numeric(df_prep.age, errors='coerce')

        skip_datasets = [
            ds for ds in df_prep.dataset_name.unique().tolist()
            if ds not in chosen_datasets
        ]
        for ds in skip_datasets:
            df_prep = df_prep.drop(df_prep.loc[df_prep['dataset_name'] == ds].index, axis=0)

        skip_conditions = [
            d for d in df_prep.disease.unique()
            if d not in disease_to_target
        ]
        for d in skip_conditions:
            df_prep = df_prep.drop(df_prep.loc[df_prep['disease'] == d].index, axis=0)

        df_prep['disease'] = df_prep['disease'].apply(lambda x: disease_to_target[x])

        categorical_columns = [
            column for column in self.metadata_cols
            if column != 'age' and column != 'disease'
        ]
        for column in categorical_columns:
            df_prep[column] = df_prep[column].astype('category').cat.codes

        if diseases is not None:
            df_prep = df_prep.loc[
                df_prep['dataset_name'].isin(
                    set(chain(*[disease_to_dataset[disease] for disease in diseases]))
                )
            ]
            df_prep = df_prep.loc[df_prep['disease'].isin(diseases + ['healthy'])]

        df_prep = df_prep.drop(['dataset_name'], axis=1)

        y = df_prep.disease.tolist()
        disease_name = diseases[0]
        indice_to_disease = ['healthy', diseases[0]]
        y = [1 if d == disease_name else 0 for d in y]

        df_prep = df_prep.drop(['disease'], axis=1)
        # self.le = LabelEncoder()
        # y = self.le.fit_transform(y)

        if skip_metadata:
            df_prep = df_prep.drop(categorical_columns, axis=1)

        new_records = []
        for record in df_prep.to_dict(orient='records'):
            new_record = {k: v for k, v in record.items() if not k.startswith('k__')}
            for key, val in record.items():
                if key.startswith('k__'):
                    target_level = self.tax_to_anc[key].get(level)
                    if target_level is None:
                        target_level = key
                    if target_level not in new_record:
                        new_record[target_level] = 0.0
                    new_record[target_level] += float(val)
            new_records.append(new_record)
        df_prep = pd.DataFrame.from_records(new_records)

        X_train, X_test, y_train, y_test = train_test_split(
            df_prep, y, test_size=test_size, random_state=seed, stratify=y
        )

        params = {
            'diseases': diseases,
            'skip_metadata': skip_metadata,
            'test_size': test_size,
            'seed': seed,
            'level': level,
            'indice_to_disease': indice_to_disease
        }

        return (X_train, y_train), (X_test, y_test), params

    def train_classifier(
        self, train_set, test_set, seed=42,
        diseases: Optional[Iterable[str]] = None,
        skip_metadata=False, xgboost_params=None,
        additional_params=None
    ):
        if xgboost_params is None:
            params = DEFAULT_PARAMS
        else:
            params = xgboost_params

        # params['num_class'] = len(set(train_set.get_label()))
        params['diseases'] = diseases
        params['skip_metadata'] = skip_metadata
        params['seed'] = seed

        self.current_data = train_set, test_set

        if additional_params is not None:
            for k, v in additional_params.items():
                params[k] = v

        self.current_params = params
        self.current_model = xgb.XGBClassifier(**params)
        self.current_model.fit(*train_set)

        self.models.append(
            {
                'model': self.current_model,
                'train_set': train_set,
                'test_set': test_set,
                'params': params
            }
        )

        return self.current_model

    @staticmethod
    def predict(model, test_set):
        preds = model.predict(test_set[0])
        if len(preds.shape) == 1:
            return [1 if pred > 0.5 else 0 for pred in preds]
        else:
            return [list(pred).index(max(pred)) for pred in model.predict(test_set[0])]

    @staticmethod
    def predict_proba(model, test_set):
        return model.predict(test_set[0])




