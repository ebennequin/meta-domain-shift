import torch
import torch.nn as nn

from src.methods.optimal_transport import OptimalTransport
from src.methods.utils import euclidean_dist


class PrototypicalOptimalTransport(OptimalTransport):
    def __init__(
        self, model_func, regularization, max_iter, stopping_criterion, lambda_cost=0
    ):
        super(PrototypicalOptimalTransport, self).__init__(
            model_func, regularization, max_iter, stopping_criterion, lambda_cost
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def execute(self, support_images, support_labels, query_images):
        """
        Overwrites method execute from OptimalTransport
        """
        z_support, z_query = self.extract_features(support_images, query_images)

        # Transport support to query instead of transporting query to support
        cost, transport_plan, _ = self.sinkhorn(z_support, z_query)

        z_support_transported = torch.matmul(
            transport_plan / transport_plan.sum(axis=1, keepdims=True), z_query
        )
        z_proto_transported = self.get_prototypes(z_support_transported, support_labels)
        scores = -euclidean_dist(z_query, z_proto_transported)

        return scores, cost
