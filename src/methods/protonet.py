import torch
from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods.utils import euclidean_dist


class ProtoNet(AbstractMetaLearner):
    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        z_support, z_query = self.extract_features(support_images, query_images)

        # Compute n_way prototypes,
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of z_support corresponding to support_labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores
