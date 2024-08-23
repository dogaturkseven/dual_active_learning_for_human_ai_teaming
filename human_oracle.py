class HumanOracle:
    def __init__(self, real_x_for_y, real_y_for_x):
        """
        initializes the human oracle with true instances and labels.

        :param real_x_for_y: pandas.DataFrame, mapping labels (y) to true instances (x).
        :param real_y_for_x: pandas.DataFrame, mapping instances (x) to true labels (y).
        """
        self.real_x_for_y = real_x_for_y
        self.real_y_for_x = real_y_for_x

    def query_classification(self, index):
        """
        provides the true label for a given instance (x)

        :param index: int, index of the instance (x) to query the true label for
        :return: true label corresponding to the given instance
        """
        true_label = self.real_y_for_x.loc[index]
        return true_label

    def query_reconstruction(self, index):
        """
        provides the true instance (x) for a given label (y)

        :param index: int, index of the label (y) to query the true instance for.
        :return: true instance corresponding to the given label
        """
        true_instance = self.real_x_for_y.loc[index]
        return true_instance
