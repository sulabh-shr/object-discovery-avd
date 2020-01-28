import os
import abc
import torch


class TripletGeneratorBase(abc.ABC):

    def __init__(self, dataset_root, proposals_path, output_root):
        self.dataset_root = dataset_root
        self.proposals_path = proposals_path
        self.output_root = output_root

    @abc.abstractmethod
    def sample_ref_views(self):
        pass

    @abc.abstractmethod
    def generate_neighbors(self):
        pass

    @abc.abstractmethod
    def sample_neighbors(self):
        pass

    @abc.abstractmethod
    def generate_triplets(self):
        pass

    @abc.abstractmethod
    def generate_train_test(self):
        pass

    @abc.abstractmethod
    def load_pt_file(self):
        pass
