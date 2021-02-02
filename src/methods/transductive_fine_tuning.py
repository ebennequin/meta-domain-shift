import torch
import torch.nn as nn
from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods.utils import euclidean_dist, entropy
from src.utils import set_device
import copy


from configs.dataset_config import CLASSES


class TransFineTune(AbstractMetaLearner):
    def __init__(
        self,
        model_func,
        transportation=None,
        training_stats=None,
        lr=5.0 * 1e-5,
        epochs=100,
    ):
        super(TransFineTune, self).__init__(
            model_func, transportation=transportation, training_stats=training_stats
        )

        # Hyper-parameters used in the paper.
        self.lr = lr
        self.epochs = epochs

        # Use the output of fc, not the backbone output.
        self.feature.trunk.add_module(
            "fc", set_device(nn.Linear(self.feature.final_feat_dim, CLASSES["train"]))
        )

        # Add a non-linearity to the output
        self.feature.trunk.add_module("relu", nn.ReLU())

        self.linear_model = None

    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        support_images = set_device(support_images)
        query_images = set_device(query_images)
        support_labels = set_device(support_labels)

        # Save parameters
        feature_parameters = copy.deepcopy(self.feature).cpu().state_dict()

        # Init linear model
        self.linear_model = set_device(
            nn.Linear(CLASSES["train"], len(torch.unique(support_labels)))
        )

        self.support_based_initializer(support_images, support_labels)

        # Compute the linear model
        self.fine_tune(support_images, support_labels, query_images)

        # Compute score of query
        scores = self.linear_model(self.feature(query_images))

        # Refresh parameters
        self.feature.load_state_dict(feature_parameters)

        feature_parameters_ = copy.deepcopy(self.feature).cpu().state_dict()

        return scores

    def fine_tune(self, support_images, support_labels, query_images):
        optimizer = torch.optim.Adam(
            params=list(self.linear_model.parameters())
            + list(self.feature.parameters()),
            lr=self.lr,
            weight_decay=0.0,
        ) 

        self.feature.train()
        self.linear_model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()

            z_support, z_query = self.extract_features(support_images, query_images)

            support_output = self.linear_model(z_support)
            query_output = self.linear_model(z_query)

            classif_loss = self.loss_fn(support_output, support_labels)
            entropy_loss = entropy(query_output)

            loss = classif_loss + entropy_loss

            loss.backward()

            optimizer.step()

        self.feature.eval()
        self.linear_model.eval()

    def support_based_initializer(self, support_images, support_labels):
        """
        Support based intialization
        See eq (6) A BASELINE FOR FEW-SHOT IMAGE CLASSIFICATION
        """
        z_support = self.feature(support_images).detach()

        z_proto = self.get_prototypes(z_support, support_labels)

        w = z_proto / z_proto.norm(dim=1, keepdim=True)

        self.linear_model.weight.data = w.clone()
        self.linear_model.bias.data = torch.zeros_like(self.linear_model.bias.data).clone()

    def train_loop(self, epoch, train_loader, optimizer):
        raise NotImplementedError(
            "Transductive Fine-Tuning does not support episodic training."
        )
