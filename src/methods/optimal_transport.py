import torch
import torch.nn as nn
from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods.sinkhorn import Sinkhorn
from src.methods.utils import one_hot


class OptimalTransport(AbstractMetaLearner):
    def __init__(
        self, model_func, regularization, max_iter, stopping_criterion, lambda_cost=0
    ):
        super(OptimalTransport, self).__init__(model_func)
        self.loss_fn = nn.NLLLoss()
        self.lambda_cost = lambda_cost
        self.sinkhorn = Sinkhorn(
            eps=regularization, max_iter=max_iter, thresh=stopping_criterion
        )

    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        z_support, z_query = self.extract_features(support_images, query_images)

        _, transport_plan, _ = self.sinkhorn(z_query, z_support)

        probs = torch.matmul(
            transport_plan / transport_plan.sum(axis=1, keepdims=True),
            one_hot(support_labels),
        )

        return torch.log(probs)

    def set_forward_loss(
        self, support_images, support_labels, query_images, query_labels
    ):
        """
        Overwrites method set_forward_loss in AbstractMetaLearner.
        """
        z_support, z_query = self.extract_features(support_images, query_images)

        cost, transport_plan, _ = self.sinkhorn(z_query, z_support)

        probs = torch.matmul(
            transport_plan / transport_plan.sum(axis=1, keepdims=True),
            one_hot(support_labels),
        )

        scores = torch.log(probs)
        classification_loss = self.loss_fn(scores, query_labels)
        loss = classification_loss + self.lambda_cost * cost

        return scores, loss
