import torch
from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods.sinkhorn import Sinkhorn
from src.methods.utils import one_hot


class OptimalTransport(AbstractMetaLearner):
    def __init__(self, model_func, regularization, max_iter, stopping_criterion):
        super(OptimalTransport, self).__init__(model_func)
        self.sinkhorn = Sinkhorn(
            eps=regularization, max_iter=max_iter, thresh=stopping_criterion
        )

    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        z_support, z_query = self.extract_features(support_images, query_images)

        _, matching, _ = self.sinkhorn(z_query, z_support)

        scores = torch.matmul(matching, one_hot(support_labels))

        return scores
