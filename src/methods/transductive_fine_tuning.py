import torch
import torch.nn as nn
from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods.utils import euclidean_dist, entropy
from src.utils import set_device



class TransFineTune(AbstractMetaLearner):
    def __init__(
        self,
        model_func,
        transportation=None,
        training_stats=None,
        lr=5.*1e-5,
        epochs=25,
    ):
        super(TransFineTune, self).__init__(
            model_func, transportation=transportation, training_stats=training_stats
        )

        # Hyper-parameters used in the paper.
        self.lr = lr
        self.epochs = epochs

        self.linear_model = None 

    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """

        # Compute the forward pass in the backbone    
        z_support, z_query = self.extract_features(support_images, query_images)

        # Apply non-linearity 
        z_support = nn.ReLU()(z_support)
        z_query = nn.ReLU()(z_query)

        # Compute the linear model
        self.fine_tune(z_support, z_query, support_labels)

        scores = self.linear_model(z_query)

        return scores

    def fine_tune(self, z_support, z_query, support_labels):
        n_way = len(torch.unique(support_labels))
        dim_features = z_support.size(1)

        support_labels = set_device(support_labels)

        self.linear_model = set_device(nn.Linear(dim_features, n_way))

        self.support_based_initializer(z_support, support_labels)

        optimizer = torch.optim.Adam(
            params=self.linear_model.parameters(),
            lr=self.lr,
            weight_decay=0.)

        for _ in range(self.epochs): 
            optimizer.zero_grad()

            if self.training:
                support_output = self.linear_model(z_support) 
                query_output = self.linear_model(z_query)
            else:
                support_output = self.linear_model(z_support.detach()) 
                query_output = self.linear_model(z_query.detach())

            classif_loss = nn.CrossEntropyLoss()(support_output, support_labels)
            entropy_loss = entropy(query_output)

            loss = classif_loss + entropy_loss

            if self.training:
                loss.backward(retain_graph=True)
            else: 
                loss.backward()

            optimizer.step()

    def support_based_initializer(self, z_support, support_labels):
        z_proto = self.get_prototypes(z_support.detach(), support_labels) 

        w = z_proto / z_proto.norm(dim=1, keepdim=True)

        self.linear_model.weight.data = w 
        self.linear_model.bias.data = torch.zeros_like(self.linear_model.bias.data)



        