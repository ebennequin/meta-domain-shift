from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from src.utils import set_device

from src.methods.modules.multi_layer_perceptron import MultiLayerPerceptron

#ConventionalBatchNorm = nn.BatchNorm2d


class _MetaBatchNorm(nn.BatchNorm2d):
    def __init__(self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True):

        super(_MetaBatchNorm, self).__init__(
            num_features, 
            eps, 
            momentum,
            affine,
            track_running_stats
        )

        self.loss = 0.

    def prepare_forward(self, mode):
        pass



class ConventionalBatchNorm(_MetaBatchNorm):
    def __init__(self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True):

        super(ConventionalBatchNorm, self).__init__(
            num_features, 
            eps, 
            momentum,
            affine,
            track_running_stats
        )


class TransductiveBatchNorm(_MetaBatchNorm):
    def prepare_forward(self, mode):
        self.running_mean = set_device(torch.zeros_like(self.running_mean))
        self.running_var = set_device(torch.ones_like(self.running_var))
        self.training = True 


class MetaNorm(_MetaBatchNorm):
    def __init__(self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False, 
        hidden_dim: int = 128):

        super(MetaNorm, self).__init__(
            num_features, 
            eps, 
            momentum,
            affine,
            track_running_stats
        )

        self.mean_hyper_net = MultiLayerPerceptron(num_features, num_features, hidden_dim)
        self.var_hyper_net = MultiLayerPerceptron(num_features, num_features, hidden_dim)

        self.infer_mean = None 
        self.infer_var = None 

        self.reset_stats = None
        self.reset_loss = None 

    def prepare_forward(self, mode):
        if mode == 'support':
            self.reset_stats = True 
            self.reset_loss = False
        elif mode == 'query': 
            self.reset_stats = False
            self.reset_loss = True

    def forward(self, x):
        if self.reset_stats:
            self.set_running_mean_var(x)
        if self.reset_loss & self.training: 
            self.compute_loss(x) 
        
        x = (x - self.infer_mean) / (self.infer_var + self.eps).sqrt()
        out = self.weight.reshape(1,-1,1,1) * x + self.bias.reshape(1,-1,1,1)
        return out

    def get_mean_var(self, x):
        x = x.permute([0, 2, 3, 1]).flatten(start_dim=0, end_dim=2)
        mean = self.mean_hyper_net(x).mean(dim=0)
        var = self.var_hyper_net(
            (x - mean)**2
        ).mean(dim=0)
        return mean.reshape(1,-1,1,1), var.reshape(1,-1,1,1)

    def set_running_mean_var(self, x):
        self.infer_mean, self.infer_var = self.get_mean_var(x)

    def compute_loss(self, x):
        mean_query, var_query = self.get_mean_var(x)

        self.loss = 0.5*(
            (var_query /  self.infer_var).log() + \
            ( self.infer_var + (self.infer_mean - mean_query)) / var_query + - 1.
            ).norm()


class NormalizationLayer(_MetaBatchNorm):
    """
    Base class for all normalization layers.
    Derives from nn.BatchNorm2d to maintain compatibility with the pre-trained resnet-18.
    """

    @abstractmethod
    def forward(self, x):
        """
        Normalize activations.
        :param x: input activations
        :return: normalized activations
        """
        pass  # always override this method

    def _normalize(self, x, mean, var):
        """
        Normalize activations.
        :param x: input activations
        :param mean: mean used to normalize
        :param var: var used to normalize
        :return: normalized activations
        """
        return (
            self.weight.view(1, -1, 1, 1) * (x - mean) / torch.sqrt(var + self.eps)
        ) + self.bias.view(1, -1, 1, 1)

    @staticmethod
    def _compute_batch_moments(x):
        """
        Compute conventional batch mean and variance.
        :param x: input activations
        :return: batch mean, batch variance
        """
        return torch.mean(x, dim=(0, 2, 3), keepdim=True), torch.var(
            x, dim=(0, 2, 3), keepdim=True
        )

    @staticmethod
    def _compute_instance_moments(x):
        """
        Compute instance mean and variance.
        :param x: input activations
        :return: instance mean, instance variance
        """
        return torch.mean(x, dim=(2, 3), keepdim=True), torch.var(
            x, dim=(2, 3), keepdim=True
        )

    @staticmethod
    def _compute_layer_moments(x):
        """
        Compute layer mean and variance.
        :param x: input activations
        :return: layer mean, layer variance
        """
        return torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(
            x, dim=(1, 2, 3), keepdim=True
        )

    @staticmethod
    def _compute_pooled_moments(x, alpha, batch_mean, batch_var, augment_moment_fn):
        """
        Combine batch moments with augment moments using blend factor alpha.
        :param x: input activations
        :param alpha: moment blend factor
        :param batch_mean: standard batch mean
        :param batch_var: standard batch variance
        :param augment_moment_fn: function to compute augment moments
        :return: pooled mean, pooled variance
        """
        augment_mean, augment_var = augment_moment_fn(x)
        pooled_mean = alpha * batch_mean + (1.0 - alpha) * augment_mean
        batch_mean_diff = batch_mean - pooled_mean
        augment_mean_diff = augment_mean - pooled_mean
        pooled_var = alpha * (batch_var + (batch_mean_diff * batch_mean_diff)) + (
            1.0 - alpha
        ) * (augment_var + (augment_mean_diff * augment_mean_diff))
        return pooled_mean, pooled_var


class TaskNormBase(NormalizationLayer):
    """TaskNorm base class."""

    def __init__(self, num_features, **kwargs):
        """
        Initialize
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(TaskNormBase, self).__init__(num_features, **kwargs)
        self.register_extra_weights()  # see register_extra_weights (not in original code)
        self.sigmoid = torch.nn.Sigmoid()

    def register_extra_weights(self):
        """
        The parameters here get registered after initialization because the pre-trained resnet model does not have
        these parameters and would fail to load if these were declared at initialization.
        :return: Nothing
        """
        device = self.weight.device

        # Initialize and register the learned parameters 'a' (SCALE) and 'b' (OFFSET)
        # for calculating alpha as a function of context size.
        scale = torch.Tensor([0.0]).to(device)
        offset = torch.Tensor([0.0]).to(device)
        self.register_parameter(
            name="scale", param=torch.nn.Parameter(scale, requires_grad=True)
        )
        self.register_parameter(
            name="offset", param=torch.nn.Parameter(offset, requires_grad=True)
        )

        # Variables to store the context moments to use for normalizing the target.
        self.register_buffer(
            name="batch_mean",
            tensor=torch.zeros(
                (1, self.num_features, 1, 1), requires_grad=True, device=device
            ),
        )
        self.register_buffer(
            name="batch_var",
            tensor=torch.ones(
                (1, self.num_features, 1, 1), requires_grad=True, device=device
            ),
        )

        # Variable to save the context size.
        self.register_buffer(
            name="context_size",
            tensor=torch.zeros((1), requires_grad=False, device=device),
        )

    def _get_augment_moment_fn(self):
        """
        Provides the function to compute augment moemnts.
        :return: function to compute augment moments.
        """
        pass  # always override this function

    def forward(self, x):
        """
        Normalize activations.
        :param x: input activations
        :return: normalized activations
        """
        if (
            self.training
        ):  # compute the pooled moments for the context and save off the moments and context size
            alpha = self.sigmoid(
                self.scale * (x.size())[0] + self.offset
            )  # compute alpha with context size
            batch_mean, batch_var = self._compute_batch_moments(x)
            pooled_mean, pooled_var = self._compute_pooled_moments(
                x, alpha, batch_mean, batch_var, self._get_augment_moment_fn()
            )
            self.context_batch_mean = batch_mean
            self.context_batch_var = batch_var
            self.context_size = torch.full_like(self.context_size, x.size()[0])
        else:  # compute the pooled moments for the target
            alpha = self.sigmoid(
                self.scale * self.context_size + self.offset
            )  # compute alpha with saved context size
            pooled_mean, pooled_var = self._compute_pooled_moments(
                x,
                alpha,
                self.context_batch_mean,
                self.context_batch_var,
                self._get_augment_moment_fn(),
            )

        return self._normalize(x, pooled_mean, pooled_var)  # normalize


class TaskNormI(TaskNormBase):
    """
    TaskNorm-I normalization layer. Just need to override the augment moment function with 'instance'.
    """

    def _get_augment_moment_fn(self):
        """
        Override the base class to get the function to compute instance moments.
        :return: function to compute instance moments
        """
        return self._compute_instance_moments
