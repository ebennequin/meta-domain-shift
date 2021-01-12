import torch
from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods.utils import euclidean_dist


class ProtoNet(AbstractMetaLearner):
    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        z_support, z_query = self.extract_features(support_images, query_images)

        z_proto = self.get_prototypes(z_support, support_labels)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores
