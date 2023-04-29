"""This file contains all the models used in the project."""

import torch.nn as nn
from torch import Tensor, maximum, sqrt


class VICReg(nn.Module):
    """This class implements the VICReg model."""

    def __init__(
        self,
        top_encoder: nn.Module,
        top_expander: nn.Module,
        bottom_encoder: nn.Module,
        bottom_expander: nn.Module,
        gamma: float = 1.0,
    ):
        super(VICReg, self).__init__()

        self.top_encoder = top_encoder
        self.top_expander = top_expander
        self.bottom_encoder = bottom_encoder
        self.bottom_expander = bottom_expander
        self.gamma = gamma

        self.eps = 1e-5

    def forward(self, top_data, bottom_data):
        """
        Forward pass of the model.
        :param top_data: Input tensor that will be
        passed through the top encoder and expander.
        :param bottom_data: Input tensor that will be
        passed through the bottom encoder and expander.
        :return: The embeddings of the top and bottom data.
        """
        top_representations = self.top_encoder(top_data)
        bottom_representations = self.bottom_encoder(bottom_data)
        top_embeddings = self.top_expander(top_representations)
        bottom_embeddings = self.bottom_expander(bottom_representations)
        return top_embeddings, bottom_embeddings

    def variance_loss(self, embeddings: Tensor) -> Tensor:
        """
        Computes the variance loss.
        'This term forces the embedding vectors of
        samples within a batch to be different.'
        :param embeddings: The embeddings of the data.
        :return: The variance loss.
        """
        return maximum(
            input=0, other=self.gamma - sqrt(embeddings.var(dim=0) + self.eps)
        ).mean()

    def invariance_loss(
        self, top_embeddings: Tensor, bottom_embeddings: Tensor
    ) -> Tensor:
        """
        Computes the invariance loss.
        'The mean square distance between the embedding vectors'
        :param top_embeddings: The embeddings of the top data.
        :param bottom_embeddings: The embeddings of the bottom data.
        :return: The invariance loss.
        """
        return nn.functional.mse_loss(top_embeddings, bottom_embeddings)

    def covariance_loss(self, embeddings: Tensor) -> Tensor:
        """
        Computes the covariance loss.
        'This term decorrelates the variables of each embedding
        and prevents an informational collapse [...] .'
        :param embeddings: The embeddings of the data.
        :return: The covariance loss.
        """
        pass
        # (embeddings.T @ embeddings)
