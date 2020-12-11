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
            n_way (int): number of classes in a classification task
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

    def evaluate(self, support_images, support_labels, query_images, query_labels):
        """
        Predict labels of query images and returns the number of correct top1 predictions
        Args:
            support_images (torch.Tensor): shape (number_of_support_set_images, **image_shape)
            support_labels (torch.Tensor): artificial support set labels in range (0, n_way)
            query_images (torch.Tensor): shape (number_of_query_set_images, **image_shape)
            query_labels (torch.Tensor): artificial query set labels in range (0, n_way)

        Returns:
            tuple(float, int): respectively number of correct top1 predictions, and total number of predictions
        """
        scores = self.set_forward(support_images, support_labels, query_images)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        top1_correct = float((topk_labels[:, 0] == set_device(query_labels)).sum())
        return float(top1_correct), len(query_labels)

    def train_loop(self, epoch, train_loader, optimizer):
        """
        Executes one training epoch
        Args:
            epoch (int): current epoch
            train_loader (DataLoader): loader of a given number of episodes
            optimizer (torch.optim.Optimizer): model optimizer

        """
        print_freq = 10

        avg_loss = 0
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
        ) in enumerate(train_loader):
            optimizer.zero_grad()

            scores = self.set_forward(support_images, support_labels, query_images)

            query_labels = set_device(query_labels)
            loss = self.loss_fn(scores, query_labels)

            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            if episode_index % print_freq == 0:
                logger.info(
                    "Epoch {epoch} | Batch {episode_index}/{n_batches} | Loss {loss}".format(
                        epoch=epoch,
                        episode_index=episode_index,
                        n_batches=len(train_loader),
                        loss=avg_loss / float(episode_index + 1),
                    )
                )

    def eval_loop(self, test_loader):
        """

        Args:
            test_loader (DataLoader): loader of a given number of episodes

        Returns:
            float: average accuracy on evaluation set
        """
        acc_all = []

        n_tasks = len(test_loader)
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
        ) in enumerate(test_loader):

            correct_this, count_this = self.evaluate(
                support_images, support_labels, query_images, query_labels
            )
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        logger.info(
            "%d Test Acc = %4.2f%% +- %4.2f%%"
            % (n_tasks, acc_mean, confidence_interval(acc_std, n_tasks))
        )

        return acc_all, acc_mean, acc_std
