import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class StyleLossNet(nn.Module):
    """
    This class takes a pretrained VGG16 network and sections it to allow for style and content losses, as described in:

    J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual Losses for Style Transfer and SR,” Eccv, pp. 1–5, 2016,
        doi: 10.1007/978-3-319-46475-6_43.

    Code is from https://github.com/dxyang/StyleTransfer
    """

    def __init__(self):
        super(StyleLossNet, self).__init__()

        self.loss_mse = torch.nn.MSELoss()

        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

    def calculate_losses(self,
                         a_real: torch.Tensor,         # X
                         a_fake: torch.Tensor,         # G_B(Y, Z)
                         a_fake_z_fake: torch.Tensor,  # G_B(Y, E(X))
                         a_rec: torch.Tensor,          # G_B(G_A(X), E(X))
                         a_rec_z_real: torch.Tensor,   # G_B(G_A(A), E)
                         b_real: torch.Tensor,         # Y
                         b_fake: torch.Tensor,         # G_A(X)
                         b_rec: torch.Tensor           # G_A(G_B(Y, Z))
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates content and style based losses using the pre-trained network.

        The arguments are the pre-calculated results of running various data through various combinations of generators
        in the AsymGAN:
             a_real:         X
             a_fake:         G_B(Y, Z)
             a_fake_z_fake:  G_B(Y, E(X))
             a_rec:          G_B(G_A(X), E(X))
             a_rec_z_real:   G_B(G_A(A), E)
             b_real:         Y
             b_fake:         G_A(X)
             b_fake_z_fake:  G_A(G_B(Y, E(X)))
             b_rec:          G_A(G_B(Y, Z))

        Returns the following losses:
           Style Loss: E[P_φj(G_B(Y, E(X))) - P_φj(X)]_F^2 (Eqn. (16) in the paper)
           Content Loss: avg(E[||φ_j(G_B(Y, E(X))) - φ_j(Y)||^2]) (Eqn. (15) in the paper)
           Total Variation Loss: (ϕ(G_A(A)) + ϕ(G_B(G_A(A), E(A))) + ϕ(G_B(G_A(A), E))
                                  + ϕ(G_B(Y, E)) + ϕ(G_A(G_B(B, E))) + ϕ(G_A(G_B(B, E(A))))) (Eqn. (18) in the paper)

        :return A tuple containing the style loss, the content loss and the total variation loss
        """

        # Calculate style loss...
        # Style Loss: E[P_φj(G_B(Y, E(X))) - P_φj(X)]_F^2 (Eqn. (16) in the paper)
        a_fake_gram = self._calculate_gram_matrix(a_fake_z_fake)
        a_real_gram = self._calculate_gram_matrix(a_real)

        style_loss = self.loss_mse(a_fake_gram[0], a_real_gram[0])
        style_loss += self.loss_mse(a_fake_gram[1], a_real_gram[1])
        style_loss += self.loss_mse(a_fake_gram[2], a_real_gram[2])
        style_loss += self.loss_mse(a_fake_gram[3], a_real_gram[3])

        # Calculate content loss (h_relu_2_2)
        # Content Loss: avg(E[||φ_j(G_B(Y, E(X))) - φ_j(Y)||^2]) (Eqn. (15) in the paper)
        content_loss = self.loss_mse(a_fake_z_fake, b_real)

        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        #
        # Total Variation Loss: (ϕ(G_A(x)) + ϕ(G_B(G_A(x), E(x))) + ϕ(G_B(G_A(x), e))
        #                           + ϕ(G_B(y, e)) + ϕ(G_A(G_B(y, e))) + ϕ(G_A(G_B(y, E(x))))) (Eqn. (18) in the paper)
        tvr_loss_b_fake = self._calculate_tvr_loss(b_fake)
        tvr_loss_a_rec = self._calculate_tvr_loss(a_rec)
        tvr_loss_a_rec_z_real = self._calculate_tvr_loss(a_rec_z_real)
        tvr_loss_a_fake = self._calculate_tvr_loss(a_fake)
        tvr_loss_b_rec = self._calculate_tvr_loss(b_rec)
        tvr_loss_a_fake_z_fake = self._calculate_tvr_loss(a_fake_z_fake)
        tvr_loss = tvr_loss_b_fake + tvr_loss_a_rec + tvr_loss_a_rec_z_real + tvr_loss_a_fake + tvr_loss_b_rec + \
            tvr_loss_a_fake_z_fake

        return style_loss, content_loss, tvr_loss

    @staticmethod
    def _calculate_tvr_loss(x: torch.Tensor) -> torch.Tensor:
        """
        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        """
        diff_i = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        return diff_i + diff_j

    def _calculate_gram_matrix(self, x: torch.Tensor) -> list:
        """
        Calculate the Gram matrix for the tensor provided in 'x'.
        """
        # calculate gram matrices for style feature layer maps we care about
        style_features = self(x)
        gram_matrix = [self._gram(feature_map) for feature_map in style_features]
        return gram_matrix

    @staticmethod
    def _gram(x: torch.Tensor):
        """
        Calculate Gram matrix (G = FF^T)
        """
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w * h)
        f_t = f.transpose(1, 2)
        gram_matrix = f.bmm(f_t) / (ch * h * w)
        return gram_matrix
