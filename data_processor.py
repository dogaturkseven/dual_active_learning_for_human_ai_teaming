import sys

import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, data_csv, merged_data=None):
        """
        initialize the DataProcessor with the path to the dataset and optional pre-merged data

        :param data_csv: str, path to the CSV file containing the dataset
        :param merged_data: pandas.DataFrame, optional pre-merged DataFrame
        """
        self.data_csv = data_csv
        self.data_df = merged_data
        self.y_train_labeled_df = None
        self.x_train_labeled_df = None
        self.x_train_unlabeled_df = None
        self.y_train_unlabeled_df = None
        self.x_train_labeled = None
        self.y_train_labeled = None
        self.real_y_for_unlabeled_x_train_df = None
        self.real_x_for_unlabeled_y_train_df = None  # labels for unlabeled instances, same indices with y_train,x_train
        self.x_test = None
        self.y_test = None

        self.x_train_for_classification_model = None
        self.y_train_for_classification_model = None

        self.x_train_for_reconstruction_model = None
        self.y_train_for_reconstruction_model = None

    def load_data(self):
        """
        load the CSV data into a pandas DataFrame
        """
        self.data_df = pd.read_csv(self.data_csv)

    def split_data(self, tra_len, trs_len, tst_len, random_seed=9):
        """
        split the data into labeled and unlabeled datasets and return as arrays

        :param tra_len: int,  number of training samples
        :param trs_len: int,  number of real samples (for unlabeled x data)
        :param tst_len: int, The number of test samples
        :param random_seed: int, Random seed for reproducibility.
        """

        self.data_df = self.data_df.copy()

        np.random.seed(random_seed)

        self.data_df = self.data_df[self.data_df['class'] != 'recommend']

        merged_data_y = pd.get_dummies(self.data_df['class'], dtype=int)

        self.data_df.drop(['class'], axis=1,
                          inplace=True)

        merged_data_X = pd.get_dummies(self.data_df, dtype=int)

        y_train_df = merged_data_y[0:tra_len]
        y_real_df = merged_data_y[tra_len:(tra_len + trs_len)]
        y_test_df = merged_data_y[(tra_len + trs_len):]

        x_train_df = merged_data_X[0:tra_len]
        x_real_df = merged_data_X[tra_len:(tra_len + trs_len)]
        x_test_df = merged_data_X[(tra_len + trs_len):]

        labeled_train_df = y_train_df[y_train_df['unlabeled'] != 1]
        unlabeled_train_df = y_train_df[y_train_df['unlabeled'] == 1]

        labeled_train_indices = y_train_df.loc[y_train_df['unlabeled'] != 1].index

        # list to collect sampled indices
        labeled_sample_indices = []
        x_unlabeled_sample_indices = []
        y_unlabeled_sample_indices = []

        # iterate over each class column
        for class_label in ['not_recom', 'priority', 'spec_prior', 'very_recom']:
            # filter the DataFrame for the current class
            class_df_labeled = labeled_train_df[labeled_train_df[class_label] == 1]
            y_real_class = y_real_df[y_real_df[class_label] == 1]
            labeled_sample_indices.extend(class_df_labeled.sample(5).index.tolist())  # length of training will be 20
            x_unlabeled_sample_indices.extend(
                y_real_class.sample(25).index.tolist())

        self.x_train_labeled_df = x_train_df.loc[labeled_sample_indices]
        self.y_train_labeled_df = y_train_df.loc[labeled_sample_indices]

        self.x_train_unlabeled_df = x_train_df.loc[x_unlabeled_sample_indices]
        self.real_y_for_unlabeled_x_train_df = y_real_df.loc[x_unlabeled_sample_indices]

        # remaining indices after selecting labeled samples

        remaining_labeled_indices = list(set(labeled_train_indices) - set(labeled_sample_indices))
        remaining_class_df = labeled_train_df.loc[remaining_labeled_indices]

        for class_label in ['not_recom', 'priority', 'spec_prior', 'very_recom']:

            class_df_remaining = remaining_class_df[remaining_class_df[class_label] == 1]
            if class_label == 'not_recom':
                sampled_indices = class_df_remaining.sample(25).index.tolist()
            elif class_label == 'priority':
                sampled_indices = class_df_remaining.sample(25).index.tolist()
            elif class_label == 'spec_prior':
                sampled_indices = class_df_remaining.sample(25).index.tolist()
            else:
                sampled_indices = class_df_remaining.sample(18).index.tolist()

            # add these sampled indices to the list
            y_unlabeled_sample_indices.extend(sampled_indices)

            self.y_train_unlabeled_df = y_train_df.loc[y_unlabeled_sample_indices]
            self.real_x_for_unlabeled_y_train_df = x_train_df.loc[y_unlabeled_sample_indices]

        self.x_train_labeled = self.x_train_labeled_df.to_numpy(dtype=int)
        self.y_train_labeled = self.y_train_labeled_df.to_numpy(dtype=int)

        self.x_test = x_test_df.to_numpy(dtype=int)
        self.y_test = y_test_df.to_numpy(dtype=int)

        # in the beginning, the training data is same
        self.x_train_for_classification_model = self.x_train_labeled_df
        self.x_train_for_reconstruction_model = self.x_train_labeled_df
        self.y_train_for_classification_model = self.y_train_labeled_df
        self.y_train_for_reconstruction_model = self.y_train_labeled_df

    def get_unlabeled_dataframe(self):
        """
        get the unlabeled dataset as DataFrames.

        :return: tuple, (x_train_unlabeled_df, y_train_unlabeled_df)
        """
        return self.x_train_unlabeled_df, self.y_train_unlabeled_df

    def update_dataset(self, index, true_label=None, true_instance=None, is_classification=None):
        """
        add human-labeled data to the labeled dataset

        :param index: int, index of the data point being updated :param true_label: true label provided by the human
        oracle (for classification) :param true_instance: true instance provided by the human oracle (for
        reconstruction) :param is_classification: bool, if True, updates the classification model; otherwise,
        updates the reconstruction model
        """

        instance_to_df = None
        label_to_df = None

        if is_classification:  # we get label from human, for unlabeled x
            instance = self.x_train_unlabeled_df.loc[index]
            instance_to_df = pd.DataFrame([instance])
            label_to_df = pd.DataFrame([true_label])
            self.x_train_unlabeled_df = self.x_train_unlabeled_df.drop(index)
            self.x_train_for_classification_model = pd.concat([self.x_train_for_classification_model, instance_to_df])
            self.y_train_for_classification_model = pd.concat([self.y_train_for_classification_model, label_to_df])

        if not is_classification:  # we get instance from human, for unlabeled y
            label = self.y_train_unlabeled_df.loc[index]
            label_to_df = pd.DataFrame([label])
            instance_to_df = pd.DataFrame([true_instance])
            self.y_train_unlabeled_df = self.y_train_unlabeled_df.drop(index)
            self.x_train_for_reconstruction_model = pd.concat([self.x_train_for_reconstruction_model, instance_to_df])
            self.y_train_for_reconstruction_model = pd.concat([self.y_train_for_reconstruction_model, label_to_df])

    def get_real_dataframes(self):  # returns real values for unlabeled data
        """
        returns the real dataframes for unlabeled data

        :return: tuple, (real_x_for_unlabeled_y_train_df, real_y_for_unlabeled_x_train_df)
        """
        return self.real_x_for_unlabeled_y_train_df, self.real_y_for_unlabeled_x_train_df

    def get_test_datasets(self):
        """
        get the test datasets as numpy arrays.

        :return: tuple, (x_test, y_test)
        """
        return self.x_test, self.y_test

    def get_train_datasets_for_classification_model(self):
        """
        get the training datasets for the classification model as numpy arrays.

        :return: tuple, (x_train_classification, y_train_classification)
        """
        x_train_classification_df = self.x_train_for_classification_model
        y_train_classification_df = self.y_train_for_classification_model

        return x_train_classification_df, y_train_classification_df

    def get_train_datasets_for_reconstruction_model(self):
        """
        get the training datasets for the reconstruction model as numpy arrays.

        :return: tuple, (x_train_reconstruction, y_train_reconstruction)
        """
        x_train_reconstruction_df = self.x_train_for_reconstruction_model
        y_train_reconstruction_df = self.y_train_for_reconstruction_model

        return x_train_reconstruction_df, y_train_reconstruction_df
