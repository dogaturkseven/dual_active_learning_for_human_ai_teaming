import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dual_active_learning import ActiveLearning
from data_processor import DataProcessor
from human_oracle import HumanOracle



def generate_list(repetitions):
    """
    generates a list by repeating  ["classification", "reconstruction"]

    :param repetitions: number of times to repeat the pattern
    :return:  list, containing the repeated pattern [c, r, c, r, ...]
    """
    pattern = ["classification", "reconstruction"]
    result = pattern * repetitions
    return result


def generate_random_orders():
    """
    generates a list of random orders for models

    :return: list of lists, containing 10 sublists, each with 10 random orders of the models
    """
    models = ['classification', 'reconstruction']
    rng = np.random.RandomState(9)  # Create a RandomState instance
    random_orders = []
    for _ in range(10):
        order = rng.choice(models, size=10).tolist()  # Generate a list of 10 random choices
        random_orders.append(order)

    return random_orders


if __name__ == "__main__":
    tra_csv = pd.read_csv('data/nursery-ssl10-10-1tra copy.csv')
    trs_csv = pd.read_csv('data/nursery-ssl10-10-1trs copy.csv')
    tst_csv = pd.read_csv('data/nursery-ssl10-10-1tst copy.csv')

    tra_len, trs_len, tst_len = len(tra_csv.index), len(trs_csv.index), len(tst_csv.index)

    merge_csv = pd.concat([tra_csv, trs_csv, tst_csv])

    data_csv = 'data/nursery-ssl10-10-1tra.csv'

    accuracy_progress = []
    mse_progress = []

    # below code for generating 10 random orders to see how risk functions work

    random_orders = generate_random_orders()

    num_random_repeats = 100

    acc_list_for_orders = []
    err_list_for_orders = []

    acc_list_for_random_select = []
    err_list_for_random_select = []

    method_list = ["risk_function", "random"]

    for method in method_list:
        if method == "risk_function":
            for order in random_orders:
                accuracies_for_ordered_model_selection = []
                errors_for_ordered_model_selection = []

                print("------starting with the ordered model selection for data selection with function---------")
                data_processor = DataProcessor(data_csv, merge_csv)
                data_processor.split_data(tra_len, trs_len, tst_len)
                real_x_for_y_df, real_y_for_x_df = data_processor.get_real_dataframes()
                human_oracle = HumanOracle(real_x_for_y_df, real_y_for_x_df)
                budget = 60
                c_class = 1
                c_recon = 3
                active_learner = ActiveLearning(budget, c_class, c_recon, data_processor, human_oracle)
                active_learner.train_models()
                X_test, y_test = active_learner.data_processor.get_test_datasets()
                # include first error and accuracy
                accuracy = active_learner.classification_model.evaluate_model(X_test, y_test)
                accuracies_for_ordered_model_selection.append(accuracy)
                mse_error = active_learner.reconstruction_model.evaluate_model(X_test, y_test)
                errors_for_ordered_model_selection.append(mse_error)

                for chosen_model in order:
                    chosen_index, is_classification = active_learner.select_data_points_for_given_model(chosen_model)
                    if is_classification:
                        active_learner.query_classification_and_update(chosen_index)
                        accuracy = active_learner.classification_model.evaluate_model(X_test, y_test)
                        accuracies_for_ordered_model_selection.append(accuracy)
                        # errors will stay same for the same line length in graph
                        errors_for_ordered_model_selection.append(errors_for_ordered_model_selection[-1])
                    else:
                        active_learner.query_reconstruction_and_update(chosen_index)
                        mse_error = active_learner.reconstruction_model.evaluate_model(X_test, y_test)
                        errors_for_ordered_model_selection.append(mse_error)
                        # accuracies will stay same for the same line length in graph
                        accuracies_for_ordered_model_selection.append(accuracies_for_ordered_model_selection[-1])

                acc_list_for_orders.append(accuracies_for_ordered_model_selection)
                err_list_for_orders.append(errors_for_ordered_model_selection)

        elif method == "random":
            for _ in range(num_random_repeats):  # repeat the random selection process multiple times
                acc_list_for_random_select_all_repeats = []
                err_list_for_random_select_all_repeats = []
                for order in random_orders:
                    accuracies_for_ordered_model_selection_random = []
                    errors_for_ordered_model_selection_random = []

                    print("------starting with the ordered model selection for random data selection---------")
                    data_processor = DataProcessor(data_csv, merge_csv)
                    data_processor.split_data(tra_len, trs_len, tst_len)
                    real_x_for_y_df, real_y_for_x_df = data_processor.get_real_dataframes()
                    human_oracle = HumanOracle(real_x_for_y_df, real_y_for_x_df)
                    budget = 60
                    c_class = 1
                    c_recon = 3
                    active_learner = ActiveLearning(budget, c_class, c_recon, data_processor, human_oracle)
                    active_learner.train_models()
                    X_test, y_test = active_learner.data_processor.get_test_datasets()
                    # include first error and accuracy
                    accuracy = active_learner.classification_model.evaluate_model(X_test, y_test)
                    accuracies_for_ordered_model_selection_random.append(accuracy)
                    mse_error = active_learner.reconstruction_model.evaluate_model(X_test, y_test)
                    errors_for_ordered_model_selection_random.append(mse_error)

                    for chosen_model in order:
                        chosen_index, is_classification = active_learner.select_data_points_randomly_for_given_model(
                            chosen_model)
                        if is_classification:
                            active_learner.query_classification_and_update(chosen_index)
                            accuracy = active_learner.classification_model.evaluate_model(X_test, y_test)
                            accuracies_for_ordered_model_selection_random.append(accuracy)
                            # errors will stay same for the same line length in graph
                            errors_for_ordered_model_selection_random.append(
                                errors_for_ordered_model_selection_random[-1])
                        else:
                            active_learner.query_reconstruction_and_update(chosen_index)
                            mse_error = active_learner.reconstruction_model.evaluate_model(X_test, y_test)
                            errors_for_ordered_model_selection_random.append(mse_error)
                            # accuracies will stay same for the same line length in graph
                            accuracies_for_ordered_model_selection_random.append(
                                accuracies_for_ordered_model_selection_random[-1])

                    acc_list_for_random_select.append(accuracies_for_ordered_model_selection_random)
                    err_list_for_random_select.append(errors_for_ordered_model_selection_random)

                acc_list_for_random_select_all_repeats.append(acc_list_for_random_select)
                err_list_for_random_select_all_repeats.append(err_list_for_random_select)

    # convert lists to numpy arrays for easier manipulation
    acc_orders_array = np.array(acc_list_for_orders)
    err_orders_array = np.array(err_list_for_orders)
    # acc_random_array = np.array(acc_list_for_random_select)
    # err_random_array = np.array(err_list_for_random_select)

    acc_random_array = np.mean(np.array(acc_list_for_random_select_all_repeats), axis=0)
    err_random_array = np.mean(np.array(err_list_for_random_select_all_repeats), axis=0)

    average_acc_orders = np.mean(acc_orders_array, axis=0)
    average_err_orders = np.mean(err_orders_array, axis=0)
    average_acc_random = np.mean(acc_random_array, axis=0)
    average_err_random = np.mean(err_random_array, axis=0)

    sem_acc_orders = np.std(acc_orders_array, axis=0) / np.sqrt(acc_orders_array.shape[0])
    sem_err_orders = np.std(err_orders_array, axis=0) / np.sqrt(err_orders_array.shape[0])
    sem_acc_random = np.std(acc_random_array, axis=0) / np.sqrt(acc_random_array.shape[0])
    sem_err_random = np.std(err_random_array, axis=0) / np.sqrt(err_random_array.shape[0])

    timestamps = np.arange(len(average_acc_orders))

    plt.figure(figsize=(12, 6))

    # plot for accuracies
    plt.subplot(1, 2, 1)
    plt.errorbar(timestamps, average_acc_orders, yerr=sem_acc_orders, marker='o', linestyle='-', color='b',
                 label='Risk Function')
    plt.errorbar(timestamps, average_acc_random, yerr=sem_acc_random, marker='x', linestyle='--', color='purple',
                 label='Random Selection')
    plt.title('Average Accuracy Over Time')
    plt.xlabel('Query Count')
    plt.ylabel('Average Accuracy')
    plt.legend()
    plt.grid(True)

    # plot for errors
    plt.subplot(1, 2, 2)
    plt.errorbar(timestamps, average_err_orders, yerr=sem_err_orders, marker='o', linestyle='-', color='b',
                 label='Risk Function')
    plt.errorbar(timestamps, average_err_random, yerr=sem_err_random, marker='x', linestyle='--', color='purple',
                 label='Random Selection')
    plt.title('Average Error Over Time')
    plt.xlabel('Query Count')
    plt.ylabel('Average Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("plot.png")