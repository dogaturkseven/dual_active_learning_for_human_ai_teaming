import math
import numpy as np
import random
from models import ClassificationModel, ReconstructionModel


class ActiveLearning:
    def __init__(self, budget, c_class, c_recon, data_processor, human_oracle):
        """
        active learning loop
        :param budget: limited budget for human oracle queries
        :param c_class: cost of classification query
        :param c_recon: cost of reconstruction  query
        :param data_processor: data object
        :param human_oracle: human oracle for queries
        """
        self.budget = budget
        self.c_class = c_class
        self.c_recon = c_recon
        self.data_processor = data_processor
        self.human_oracle = human_oracle
        self.classification_model = ClassificationModel()
        self.reconstruction_model = ReconstructionModel()
        self.sigma_values = None

    def train_models(self):
        """
        train the models
        """
        X, y = self.data_processor.get_train_datasets_for_classification_model()
        self.classification_model.train(X.to_numpy(dtype=int), y.to_numpy(dtype=int))
        X, y = self.data_processor.get_train_datasets_for_reconstruction_model()
        self.reconstruction_model.train(X.to_numpy(dtype=int), y.to_numpy(dtype=int))

    def calculate_sigma_values(self):
        """
        calculate sigma values representing L2 norm distance of Xfeatures for each yClass
        """
        sigma_values_list = []

        labeled_X_df, labeled_y_df = self.data_processor.get_train_datasets_for_classification_model()

        for c in labeled_y_df.columns:
            # sth like : ['not_recom', 'priority', 'recommend', 'spec_prior', 'unlabeled', 'very_recom']

            # filters x to include only the rows where the corresponding column in y has a value of 1
            filtered_data = labeled_X_df.loc[labeled_X_df.index.isin(labeled_y_df.loc[labeled_y_df[c] == 1].index)]
            # calculates std of each feature for filtered data
            std_devs = filtered_data.describe().loc['std', :].to_numpy()
            sigma = np.linalg.norm(std_devs)

            if math.isnan(sigma):
                std_devs_all = labeled_X_df.describe().loc['std', :].to_numpy()
                sigma = np.linalg.norm(std_devs_all)

            sigma_values_list.append(sigma)

        self.sigma_values = sigma_values_list

    def calculate_risk_for_x(self, real_x, reco_x, predict_correct, sigma):
        """
        calculate risk of the datapoint

        :param real_x: real value of instance x
        :param reco_x: reconstructed value of instance x
        :param predict_correct: bool value to indicate whether the predictions
        for real_x and reco_x is same or not

        :return: risk value for the instance
        """
        raw_risk = np.linalg.norm(real_x - reco_x) / (2 * sigma * sigma)
        risk = np.exp(-raw_risk) if predict_correct else np.exp(raw_risk)
        return risk

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # for numerical stability
        return e_x / e_x.sum(axis=0)  # np.exp(x) / np.sum(np.exp(x), axis=0)

    # def cross_entropy_loss(self, real_y, reconstructed_y):
    #    return -np.sum(real_y * np.log(predicted_probabilities))
    def calculate_risk_for_y(self, real_y, reconstructed_y):
        """
        calculates the risk (cross-entropy loss) for given real and reconstructed y values

        :param real_y: ground truth labels
        :param reconstructed_y: predicted values before applying softmax

        :return: float, the computed risk
        """
        probabilities = self.softmax(reconstructed_y)
        risk = -np.sum(real_y * np.log(probabilities))
        return risk

    def calculate_score(self, normalized_risk, cost):
        """
        computes the score to choose data
        """
        return normalized_risk / cost

    def calculate_risk_values(self, ):
        """
            calculates risk values for unlabeled data using both x and y datasets

            this method evaluates risk for each instance in the unlabeled x and y datasets
            - for the x dataset, it computes the risk based on the prediction and reconstruction of instances
            - for the y dataset, it calculates risk using the real and predicted labels after reconstruction

            :return: tuple of two dictionaries
                - x_risk_dic: dict, where keys are indices and values are risk values for x instances
                - y_risk_dic: dict, where keys are indices and values are risk values for y instances
            """
        self.calculate_sigma_values()
        x_unlabeled_df, y_unlabeled_df = self.data_processor.get_unlabeled_dataframe()

        x_risk_dic = {}
        y_risk_dic = {}

        for index, row in x_unlabeled_df.iterrows():
            real_x = row.to_numpy(dtype=int)
            prediction, predicted_class = self.classification_model.predict_label(real_x)
            reco_x = self.reconstruction_model.predict_instance(prediction)
            reco_prediction, reco_predicted_class = self.classification_model.predict_label(reco_x)
            sigma = self.sigma_values[predicted_class]
            risk = self.calculate_risk_for_x(real_x, reco_x, predicted_class == reco_predicted_class, sigma)

            x_risk_dic[index] = risk

        for index, row in y_unlabeled_df.iterrows():
            real_y = row.to_numpy(dtype=int)
            real_y_class = np.argmax(real_y)
            predicted_x = self.reconstruction_model.predict_instance(real_y)
            predicted_y, predicted_y_class = self.classification_model.predict_label(predicted_x)
            reco_x = self.reconstruction_model.predict_instance(predicted_y)
            risk = self.calculate_risk_for_y(real_y, predicted_y)

            y_risk_dic[index] = risk

        return x_risk_dic, y_risk_dic

    """
    def select_data_point(self):
    
        # select the data point which has the highest risk
        
        selected_index = None
        is_classification = True

        x_risk_dic, y_risk_dic = self.calculate_risk_values()
        x_risk_dic = {key: self.calculate_score(value, self.c_class) for key, value in x_risk_dic.items()}
        y_risk_dic = {key: self.calculate_score(value, self.c_recon) for key, value in y_risk_dic.items()}

        # TODO: delete the plotting code
        value_counts = Counter(y_risk_dic.values())
        print(value_counts)
        unique_values = list(value_counts.keys())
        print(unique_values)
        counts = list(value_counts.values())
        print(counts)

        max_risk_class = max(x_risk_dic.values())
        min_risk_class = min(x_risk_dic.values())
        max_risk_reco = max(y_risk_dic.values())
        min_risk_reco = min(y_risk_dic.values())

        selected_class_index, normalized_max_class = max(x_risk_dic.items(), key=lambda item: item[1])
        selected_reco_index, normalized_max_reco = max(y_risk_dic.items(), key=lambda item: item[1])

        if normalized_max_reco > normalized_max_class:
            is_classification = False
            selected_index = selected_reco_index
            print("selected the query y for reconstruction")

        elif normalized_max_class >= normalized_max_reco:  # eğer eşitlerse cost düşük olanı seçtik
            selected_index = selected_class_index
            print("selected the query x for classification")

        # TODO: burayı sil, şimdi sadece denemek için var - sadece classification seçiyor
        selected_index = selected_reco_index
        is_classification = False
        return selected_index, is_classification"""

    def select_model_randomly(self):
        """
        randomly selects between 'classification' and 'reconstruction' models

        :return: str, The selected model type, either 'classification' or 'reconstruction'
        """
        return random.choice(
            ['classification', 'reconstruction'])

    def select_data_point_randomly(self):
        """
        randomly chooses between 'classification' and 'reconstruction' models and then selects
        a data point from the corresponding unlabeled dataset

        :return: tuple
            - index: int, The index of the randomly selected data point
            - is_classification: bool, Indicates whether the selected model type is 'classification'
        """
        is_classification = False
        model_type = self.select_model_randomly()
        if model_type == 'classification':
            is_classification = True
            unlabeled_df, y_train_unlabeled_df = self.data_processor.get_unlabeled_dataframe()
        else:
            x_train_unlabeled_df, unlabeled_df = self.data_processor.get_unlabeled_dataframe()

        index = unlabeled_df.sample().index[0]
        return index, is_classification

    def query_classification_and_update(self, index):
        """
        queries the human oracle for classification and updates the labeled dataset

        :param index: int, index of the data point to query for classification
        """

        true_label = self.human_oracle.query_classification(index)
        self.data_processor.update_dataset(index, true_label, is_classification=True)
        X, y = self.data_processor.get_train_datasets_for_classification_model()
        self.classification_model.train(X.to_numpy(dtype=int), y.to_numpy(dtype=int))
        self.budget -= self.c_class
        print(f"query for classification completed, remaining budget: {self.budget}")

    def query_reconstruction_and_update(self, index):
        """
        queries the human oracle for reconstruction and updates the labeled dataset

        :param index: int, index of the data point to query for reconstruction.
        """

        true_instance = self.human_oracle.query_reconstruction(index)
        self.data_processor.update_dataset(index, true_instance=true_instance, is_classification=False)
        X, y = self.data_processor.get_train_datasets_for_reconstruction_model()
        self.reconstruction_model.train(X.to_numpy(dtype=int), y.to_numpy(dtype=int))
        self.budget -= self.c_recon
        print(f"query for reconstruction completed, remaining budget: {self.budget}")

    def select_data_points_for_given_model(self, chosen_model):
        """
           selects a data point based on the given model type by evaluating risk values

           :param chosen_model: str, type of model to select data points for,
           either 'classification' or 'reconstruction'

           :return: tuple
               - chosen_index: int, index of the selected data point
               - is_classification: bool, indicates whether the selected model type is 'classification' or not
        """
        x_risk_dic, y_risk_dic = self.calculate_risk_values()
        chosen_index = None
        is_classification = True

        if chosen_model == 'classification':
            chosen_index, max_value_for_x = max(x_risk_dic.items(), key=lambda item: item[1])

        elif chosen_model == 'reconstruction':
            is_classification = False
            chosen_index, max_value_for_y = max(y_risk_dic.items(), key=lambda item: item[1])

        return chosen_index, is_classification

    def select_data_points_randomly_for_given_model(self, chosen_model):
        """
        randomly selects a data point for the given model type

        :param chosen_model: str, type of model to select data points for,
        either 'classification' or 'reconstruction'

        :return: tuple
               - index: int, index of the randomly selected data point
               - is_classification: bool, indicates whether the selected model type is 'classification' or not
        """
        unlabeled_df = None
        is_classification = True
        if chosen_model == 'classification':
            unlabeled_df, y_train_unlabeled_df = self.data_processor.get_unlabeled_dataframe()

        elif chosen_model == 'reconstruction':
            is_classification = False
            x_train_unlabeled_df, unlabeled_df = self.data_processor.get_unlabeled_dataframe()

        rng = np.random.RandomState()
        index = rng.choice(unlabeled_df.index)
        return index, is_classification



