import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score


class ClassificationModel:
    def __init__(self):
        self.model = KernelRidge(alpha=0.05)

    def train(self, X, y):
        """
        train the model
        """
        self.model.fit(X, y)

    # TODO: CHECK THIS FUNCTION
    def predict_label(self, x_instance):
        """"
        predict the label of a given instance.

        :param x_instance: numpy.ndarray, single data instance to predict
        :return: tuple, (prediction, predicted_class)
                 - prediction: raw output from the model (continuous values)
                 - predicted_class: class with the highest score (as an integer)
        """
        y_predicted = self.model.predict(np.reshape(x_instance, (1, len(x_instance))))

        prediction = y_predicted[0]
        predicted_class = np.argmax(y_predicted[0])
        return prediction, predicted_class

    def evaluate_model(self, X_test, y_test):
        """
        evaluate the model's performance on the test set
        evaluation is based on accuracy, which is the proportion of correctly predicted labels

        :param X_test: numpy.ndarray, feature matrix for test data
        :param y_test: numpy.ndarray, true labels for the test data
        :return: float, accuracy score of the model on the test data
        """
        y_pred = []
        for x_instance in X_test:
            prediction, predicted_class = self.predict_label(x_instance)
            y_pred.append(predicted_class)

        y_test_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_classes, y_pred)
        return accuracy


class ReconstructionModel:
    def __init__(self):
        self.alpha_k = None

    def train(self, X, y):
        """
        train the model
        """
        XT = np.transpose(X)
        XTX = np.matmul(XT, X)
        gamma = 0.05
        Id = np.identity(len(XTX))
        self.alpha_k = np.matmul(np.matmul(np.linalg.inv(XTX + gamma * Id), XT), y)

    # TODO: CHECK THIS FUNCTION
    def predict_instance(self, prediction):
        """
        reconstruct an instance based on the provided label

        :param prediction: numpy.ndarray, label used to reconstruct the instance
        :return: numpy.ndarray, reconstructed instance
        """
        x_predicted = np.matmul(self.alpha_k, prediction)
        return x_predicted

    def evaluate_model(self, X_test, y_test):
        """
        evaluate the reconstruction model's performance on the test set
        evaluation is based on mean squared error (MSE) between the real and reconstructed instances

        :param X_test: numpy.ndarray, feature matrix for test data
        :param y_test: numpy.ndarray, Label matrix for test data
        :return: float, mean squared error over the test set
        """

        total_mse = 0.0

        for i in range(X_test.shape[0]):
            real_x = X_test[i]
            real_y = y_test[i]

            predicted_x = self.predict_instance(real_y)
            total_mse += np.mean((real_x - predicted_x) ** 2)

        mse = total_mse / X_test.shape[0]
        return mse
