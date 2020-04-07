import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def multioutput_datagen(generator):
    while True:
        Xi, yi = generator.next()

        target_dict = {
            "decoder_output": Xi,
            "encoding_classifier_output": yi,
            "image_classifier_output": yi,
        }

        yield Xi, target_dict


def three_classifier_output_datagen(generator):
    while True:
        Xi, yi = generator.next()

        target_dict = {
            "decoder_output": Xi,
            "encoding_classifier_output": yi,
            "image_classifier_output": yi,
        }

        yield Xi, target_dict


def two_classifier_output_datagen(generator):
    while True:
        Xi, yi = generator.next()

        target_dict = {
            "decoder_output": Xi,
            "encoding_classifier_output": yi,
        }

        yield Xi, target_dict


class CelebA(object):
    def __init__(
        self,
        main_folder="./data/celeba/",
        image_folder="img_align_celeba_cropped/",
        selected_features=None,
        drop_features=[],
    ):
        self.main_folder = main_folder
        self.images_folder = os.path.join(main_folder, image_folder)
        self.attributes_path = os.path.join(main_folder, "list_attr_celeba.txt")
        self.partition_path = os.path.join(main_folder, "list_eval_partition.txt")
        self.selected_features = selected_features
        self.features_name = []
        self.__prepare(drop_features)

    def __prepare(self, drop_features):
        """do some preprocessing before using the data: e.g. feature selection"""
        # attributes:
        if self.selected_features is None:
            self.attributes = pd.read_csv(
                self.attributes_path, delim_whitespace=True, header=1
            )
            self.num_features = 40
        else:
            self.num_features = len(self.selected_features)
            self.selected_features = self.selected_features.copy()
            self.attributes = pd.read_csv(
                self.attributes_path, delim_whitespace=True, header=1
            )[self.selected_features]

        # remove unwanted features:
        for feature in drop_features:
            if feature in self.attributes:
                self.attributes = self.attributes.drop(feature, axis=1)
                self.num_features -= 1

        self.attributes["image_id"] = self.attributes.index.values
        self.attributes.replace(to_replace=-1, value=0, inplace=True)
        self.attributes["image_id"] = list(self.attributes.index)
        self.features_name = list(self.attributes.columns)[:-1]

        # load ideal partitioning:
        self.partition = pd.read_csv(
            self.partition_path, delim_whitespace=True, header=None, index_col=0
        )
        self.partition.index.rename("image_id", inplace=True)
        self.partition.rename(columns={1: "partition",}, inplace=True)

    def split(self, name="training", drop_zero=False):
        """Returns the ['training', 'validation', 'test'] split of the dataset"""
        # select partition split:
        if name == "training":
            to_drop = self.partition.where(lambda x: x != 0).dropna()
        elif name == "validation":
            to_drop = self.partition.where(lambda x: x != 1).dropna()
        elif name == "test":  # test
            to_drop = self.partition.where(lambda x: x != 2).dropna()
        else:
            raise ValueError(
                "CelebA.split() => `name` must be one of [training, validation, test]"
            )

        partition = self.partition.drop(index=to_drop.index)

        # join attributes with selected partition:
        joint = partition.join(self.attributes, how="inner").drop("partition", axis=1)

        if drop_zero is True:
            return joint.loc[(joint[self.features_name] == '1').any(axis=1)]

        return joint

    def __len__(self):
        """ Get number of samples in CelebA """
        return len(self.attributes)
