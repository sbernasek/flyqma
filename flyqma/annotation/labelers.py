import numpy as np


class AttributeLabeler:
    """
    Assigns label to cell measurement data based on an existing attribute.

    Attributes:

        label (str) - name of label field to be added

        attribute (str) - existing cell attribute used to determine labels

        labeler (vectorized func) - callable that maps attribute values to labels

    """

    def __init__(self, label, attribute, labels):
        """
        Instantiate labeler.

        Args:

            label (str) - name of label attribute to be added

            attribute (str) - existing cell attribute used to determine labels

            labels (dict) - {attribute value: label value} pairs

        """

        # store label and attribute field names
        self.label = label
        self.attribute = attribute

        # vectorize labeling function
        self.labeler = np.vectorize(labels.get)

    def __call__(self, data):
        """
        Assign labels by adding <label> field to cell measurement data.

        Args:

           data (pd.DataFrame) - cells measurement data with <attribute> field

        """
        return self.assign_labels(data)

    def assign_labels(self, data):
        """
        Assign labels by adding <label> field to cell measurement data.

        Args:

            data (pd.DataFrame) - cells measurement data with <attribute> field

        """
        data[self.label] = self.labeler(data[self.attribute])


class CelltypeLabeler(AttributeLabeler):
    """
    Assigns <celltype> to cell measurement data based on <genotype> attribute.

    Attributes:

        label (str) - name of label field to be added

        attribute (str) - existing cell attribute used to determine labels

        labeler (vectorized func) - callable that maps attribute values to labels

    """

    def __init__(self, label='celltype', attribute='genotype', labels=None):
        """
        Instantiate celltype labeler.

        Args:

            label (str) - name of label attribute to be added

            attribute (str) - existing cell attribute used to determine labels

            labels (dict) - {genotype value: label} pairs

        """

        # use default genotype labels
        if labels is None:
            labels = {0:'m', 1:'h', 2:'w', -1:'none'}

        super().__init__(label, attribute, labels)
