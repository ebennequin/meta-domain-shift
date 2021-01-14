from abc import abstractmethod

from loguru import logger
import numpy as np
import torch
import torch.nn as nn

from src.methods.utils import confidence_interval
from src.utils import set_device


class AbstractMetaLearner(nn.Module):
    """
    Abstract class for meta-learning models. Extensions of this class must define the set_forward function.
    """

    def __init__(self, model_func, training_stats=None):
        """

        Args:
            model_func (src.backbones object): backbone function
            training_stats (Statistics): training statistics of the model, updated during training

        """
        super(AbstractMetaLearner, self).__init__()
        self.feature = model_func()
        self.training_stats = training_stats
        self.loss_fn = nn.CrossEntropyLoss()

    @abstractmethod
    def set_forward(self, support_images, support_labels, query_images):
        """
        Predict query set labels using information from support set labelled images.
        Must be implemented in all classes extending AbstractMetaLearner.
        Args:
            support_images (torch.Tensor): shape (number_of_support_set_images, **image_shape)
            support_labels (torch.Tensor): artificial support set labels in range (0, n_way)
            query_images (torch.Tensor): shape (number_of_query_set_images, **image_shape)

        Returns:
            torch.Tensor: shape(n_query*n_way, n_way), classification prediction for each query data
        """
        pass

    def set_forward_loss(
        self, support_images, support_labels, query_images, query_labels
    ):
        """
        Predict query set labels using information from support set labelled images, and computes the loss.
        Args:
            support_images (torch.Tensor): shape (number_of_support_set_images, **image_shape)
            support_labels (torch.Tensor): artificial support set labels in range (0, n_way)
            query_images (torch.Tensor): shape (number_of_query_set_images, **image_shape)
            query_labels (torch.Tensor): artificial query set labels in range (0, n_way)

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - shape(n_query*n_way, n_way), classification prediction for each query data
                - shape(,), training loss
        """
        scores = self.set_forward(support_images, support_labels, query_images)
        loss = self.loss_fn(scores, query_labels)

        return scores, loss

    def extract_features(self, support_images, query_images):
        """
        Computes the features vectors of the support and query sets
        Args:
            support_images (torch.Tensor): shape (n_support_images, **image_dim) input data
            query_images (torch.Tensor): shape (n_query_images, **image_dim) input data

        Returns:
            Tuple(torch.Tensor, torch.Tensor): features vectors of the support and query sets, respectively of shapes
            (n_support_images, features_dim) and (n_query_images, features_dim)
        """
        # Set to CUDA if available
        support_images = set_device(support_images)
        query_images = set_device(query_images)

        # Concatenate tensors and compute images features
        number_of_support_images = support_images.shape[0]
        all_images = torch.cat((support_images, query_images))
        z_all = self.feature.forward(all_images)

        # Re-split between support and query images
        z_support = z_all[:number_of_support_images]
        z_query = z_all[number_of_support_images:]

        return z_support, z_query

    def get_prototypes(self, features, labels):
        """
        Compute a prototype for each class.
        Args:
            features (torch.Tensor): shape[n_images, feature_dim], feature vectors
            labels (torch.Tensor): shape[n_images], label corresponding to each feature vector
        Returns:
            torch.Tensor: prototypes. shape[number_of_unique_values_in_labels, feature_dim]
        """

        n_way = len(torch.unique(labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        return torch.cat(
            [features[torch.nonzero(labels == label)].mean(0) for label in range(n_way)]
        )

    @staticmethod
    def evaluate(scores, query_labels):
        """
        Predict labels of query images and returns the number of correct top1 predictions
        Args:
            scores (torch.Tensor): shape (number_of_query_set_images, n_way)
            query_labels (torch.Tensor): artificial query set labels in range (0, n_way)

        Returns:
            float: classification accuracy in [0,1]
        """

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        top1_correct = float((topk_labels[:, 0] == query_labels).sum())
        return float(top1_correct) / len(query_labels)

    def train_loop(self, epoch, train_loader, optimizer):
        """
        Executes one training epoch
        Args:
            epoch (int): current epoch
            train_loader (DataLoader): loader of a given number of episodes
            optimizer (torch.optim.Optimizer): model optimizer

        Returns:
            tuple(float, float): resp. average loss and classification accuracy
        """
        print_freq = 100

        loss_list = []
        acc_list = []
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
            _,
            _,
        ) in enumerate(train_loader):
            optimizer.zero_grad()

            query_labels = set_device(query_labels)
            scores, loss = self.set_forward_loss(
                support_images, support_labels, query_images, query_labels
            )

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            acc_list.append(self.evaluate(scores, query_labels) * 100)

            if episode_index % print_freq == print_freq - 1:
                logger.info(
                    "Epoch {epoch} | Batch {episode_index}/{n_batches} | Loss {loss}".format(
                        epoch=epoch,
                        episode_index=episode_index + 1,
                        n_batches=len(train_loader),
                        loss=np.asarray(loss_list).mean(),
                    )
                )

        return np.asarray(loss_list).mean(), np.asarray(acc_list).mean()

    def eval_loop(self, test_loader):
        """

        Args:
            test_loader (DataLoader): loader of a given number of episodes

        Returns:
            tuple(float, float): resp. average loss and classification accuracy
        """
        acc_all = []
        loss_all = []

        n_tasks = len(test_loader)
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
            source_domain,
            target_domain,
        ) in enumerate(test_loader):

            query_labels = set_device(query_labels)
            scores, loss = self.set_forward_loss(
                support_images, support_labels, query_images, query_labels
            )

            loss_all.append(loss.item())

            acc_all.append(self.evaluate(scores, query_labels) * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        logger.info(
            "%d Test Accuracy = %4.2f%% +- %4.2f%%"
            % (n_tasks, acc_mean, confidence_interval(acc_std, n_tasks))
        )

        return np.asarray(loss_all).mean(), acc_mean
