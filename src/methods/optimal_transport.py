import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from configs import experiment_config
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
        self.episode_count = 0
        self.loss_writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)

    def execute(self, support_images, support_labels, query_images):
        """
        Applies Optimal Transport from query images to support images,
            and uses the outcome to predict query classification scores.
        Args:
            support_images (torch.Tensor): shape (number_of_support_set_images, **image_shape)
            support_labels (torch.Tensor): artificial support set labels in range (0, n_way)
            query_images (torch.Tensor): shape (number_of_query_set_images, **image_shape)
        Returns:
            tuple(torch.Tensor, torch.Tensor) : respectively classification scores, and transport cost
        """
        z_support, z_query = self.extract_features(support_images, query_images)

        cost, transport_plan, _ = self.sinkhorn(z_query, z_support)

        probs = torch.matmul(
            transport_plan / transport_plan.sum(axis=1, keepdims=True),
            one_hot(support_labels),
        )

        return torch.log(probs), cost

    def get_loss(self, query_labels, scores, transport_cost):
        """
        Computes loss from classification error and transport cost
        Args:
            query_labels (torch.Tensor): artificial query set labels in range (0, n_way)
            scores (torch.Tensor): classification scores to be compared to ground truth query labels
            transport_cost (torch.Tensor): transport cost of the optimal transport
        Returns:
            torch.Tensor: loss
        """
        classification_loss = self.loss_fn(scores, query_labels)

        if self.training:
            self.loss_writer.add_scalar(
                "Train/classification_loss", classification_loss, self.episode_count
            )
            self.loss_writer.add_scalar(
                "Train/transport_cost", transport_cost, self.episode_count
            )
            self.episode_count = self.episode_count + 1

        loss = classification_loss + self.lambda_cost * transport_cost

        return loss

    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        logprobs, _ = self.execute(support_images, support_labels, query_images)

        return logprobs

    def set_forward_loss(
        self, support_images, support_labels, query_images, query_labels
    ):
        """
        Overwrites method set_forward_loss in AbstractMetaLearner.
        """
        logprobs, cost = self.execute(support_images, support_labels, query_images)

        loss = self.get_loss(query_labels, logprobs, cost)

        return logprobs, loss
