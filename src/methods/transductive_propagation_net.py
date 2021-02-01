import torch
from src.methods.abstract_meta_learner import AbstractMetaLearner
from src.methods.utils import euclidean_dist
from src.utils import set_device
from src.methods.backbones import MultiLayerPerceptron


class TransPropNet(AbstractMetaLearner):
    def __init__(self, model_func, transportation=None, training_stats=None):
        super(TransPropNet, self).__init__(
            model_func, 
            transportation=transportation,
            training_stats=training_stats
        )

        # Hypernetwork that fits scaling factor in similarity
        self.length_scale = MultiLayerPerceptron(self.feature.final_feat_dim)

        # Hyper-parameters used in the paper.
        self.alpha = 0.99
        self.eps = 1e-8
        self.k = 5 


    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """

        z_support, z_query = self.extract_features(
            support_images, 
            query_images
        )

        similarity = self.get_similarity(
            z_support,
            z_query
        )


        # Normalization of the laplacian
        d = (1./(similarity+self.eps).sum(dim=0)).diag().sqrt()

        laplacian = torch.matmul(
            d,
            torch.matmul(
                similarity,
                d
            )
        )

        scores = self.propagate(laplacian, support_labels)

        return scores[support_images.size(0):]


    def get_similarity(self, z_support, z_query):
        z = torch.cat(
            [
                z_support, 
                z_query
            ],
            dim=0
        )

        # scaling with forward of self.length_scale
        z = z / (self.length_scale(z) + self.eps)

        similarity = torch.exp(-0.5*euclidean_dist(z, z))

        # Keep only top k values in the similarity matrix
        _, idx = similarity.topk(self.k, dim=1)

        similarity_del = similarity.clone()
        similarity_del.scatter_(1, idx, 0)

        return similarity - similarity_del


    def propagate(self, laplacian, support_labels):
        # compute labels as one_hot
        b = support_labels.size(0)
        n_way = len(torch.unique(support_labels))

        ## compute support labels as one hot
        one_hot_labels = set_device(
            torch.zeros(
                support_labels.size(0), 
                n_way
                )
        )

        one_hot_labels[torch.arange(b), support_labels] = 1.0

        ## sample to predict has 0 everywhere
        one_hot_labels = torch.cat(
            [
                one_hot_labels, 
                set_device(
                    torch.zeros(
                        laplacian.size(0) - b, 
                        n_way
                    )
                )
            ]
        )

        # compute label propagation
        propagation = (set_device(torch.eye(laplacian.size(0))) \
            - self.alpha*laplacian + self.eps).inverse()

        scores = torch.matmul(
            propagation,
            one_hot_labels
        )

        return scores



