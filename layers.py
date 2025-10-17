"""
This module defines LJN.

Parts of this code are adapted from the original DiffusionNet implementation
"""

import torch
import torch.nn as nn
from utils import spth_to_cholespy
from SPLUSolveLayer import SPLUSolveLayer


# region Geometric Helpers
def to_basis(
    values: torch.Tensor, basis: torch.Tensor, massvec: torch.Tensor
) -> torch.Tensor:
    """
    Transform data into an orthonormal basis (w.r.t. massvec).

    Args:
        values (torch.Tensor): Input data of shape (B, V, D).
        basis (torch.Tensor): Basis vectors of shape (B, V, K).
        massvec (torch.Tensor): Mass vector of shape (B, V).

    Returns:
        torch.Tensor: Transformed values of shape (B, K, D).
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Transform data out of an orthonormal basis.

    Args:
        values (torch.Tensor): Input data of shape (K, D).
        basis (torch.Tensor): Basis vectors of shape (V, K).

    Returns:
        torch.Tensor: Reconstructed values of shape (V, D).
    """
    if values.is_complex() or basis.is_complex():
        raise NotImplementedError("Complex analysis not implemented!")
    else:
        return torch.matmul(basis, values)


# endregion


class SpatialGradientFeatures(nn.Module):
    """
    Computes dot products between input vectors using a learned complex-linear layer.
    """

    def __init__(self, C_inout: int):
        super(SpatialGradientFeatures, self).__init__()
        self.C_inout = C_inout
        self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
        self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vectors (torch.Tensor): Input vectors of shape (V, C, 2).

        Returns:
            torch.Tensor: Computed dot products of shape (V, C).
        """
        vectorsA = vectors
        vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
        vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        dots = vectorsA[..., 0] * vectorsBreal + vectorsA[..., 1] * vectorsBimag
        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes.
    """

    def __init__(
        self,
        layer_sizes: list,
        dropout: bool = False,
        activation: nn.Module = nn.ReLU,
        name: str = "miniMLP",
    ):
        super(MiniMLP, self).__init__()
        for i in range(len(layer_sizes) - 1):
            is_last = i + 2 == len(layer_sizes)
            if dropout and i > 0:
                self.add_module(f"{name}_mlp_layer_dropout_{i:03d}", nn.Dropout(p=0.5))

            self.add_module(
                f"{name}_mlp_layer_{i:03d}",
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
            )

            if not is_last:
                self.add_module(f"{name}_mlp_act_{i:03d}", activation())


class DiffusionNetBlock(nn.Module):
    """
    A single block of the DiffusionNet, processing features on vertices.
    """

    def __init__(self, C_width: int, mlp_hidden_dims: list, dropout: bool = True):
        super(DiffusionNetBlock, self).__init__()
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        self.MLP_C = (
            3 * self.C_width
        )  # 2 for concatenated features, 1 for gradient features
        self.gradient_features = SpatialGradientFeatures(self.C_width)
        self.mlp = MiniMLP(
            [self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout
        )

    def forward(
        self,
        x_in: torch.Tensor,
        mass: torch.Tensor,
        L: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        gradX: torch.Tensor,
        gradY: torch.Tensor,
        faces: torch.Tensor = None,
    ) -> torch.Tensor:
        B = x_in.shape[0]
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                f"Tensor has wrong shape = {x_in.shape}. Last dim should be {self.C_width}"
            )

        x_spec = to_basis(x_in, evecs, mass)
        x_diffuse = from_basis(x_spec, evecs)

        x_grads = []
        for b in range(B):
            x_gradX = torch.mm(gradX[b, ...], x_diffuse[b, ...])
            x_gradY = torch.mm(gradY[b, ...], x_diffuse[b, ...])
            x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
        x_grad = torch.stack(x_grads, dim=0)

        x_grad_features = self.gradient_features(x_grad)
        feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        x0_out = self.mlp(feature_combined)
        x0_out = x0_out + x_in
        return x0_out


class DiffusionNet(nn.Module):
    """
    A neural network for learning on meshes, using diffusion and gradient features.
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        C_width: int = 128,
        N_block: int = 4,
        last_activation: callable = None,
        outputs_at: str = "vertices",
        mlp_hidden_dims: list = None,
        dropout: bool = True,
    ):
        """
        Args:
            C_in (int): Input channel dimension.
            C_out (int): Output channel dimension.
            C_width (int): Dimension of internal DiffusionNet blocks.
            N_block (int): Number of DiffusionNet blocks.
            last_activation (callable, optional): Activation to apply to the final output. Defaults to None.
            outputs_at (str, optional): Mesh element to produce outputs at. One of ['vertices', 'edges', 'faces']. Defaults to "vertices".
            mlp_hidden_dims (list, optional): Hidden layer sizes for MLPs. Defaults to [C_width, C_width].
            dropout (bool, optional): Whether to use dropout in MLPs. Defaults to True.
        """
        super(DiffusionNet, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ["vertices", "edges", "faces"]:
            raise ValueError("invalid setting for outputs_at")

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)

        self.blocks = nn.ModuleList()
        for _ in range(self.N_block):
            self.blocks.append(
                DiffusionNetBlock(
                    C_width=C_width,
                    mlp_hidden_dims=mlp_hidden_dims,
                    dropout=dropout,
                )
            )

    def forward(
        self,
        x_in: torch.Tensor,
        mass: torch.Tensor,
        L: torch.Tensor = None,
        evals: torch.Tensor = None,
        evecs: torch.Tensor = None,
        gradX: torch.Tensor = None,
        gradY: torch.Tensor = None,
        edges: torch.Tensor = None,
        faces: torch.Tensor = None,
    ) -> torch.Tensor:
        appended_batch_dim = False
        if x_in.shape[-1] != self.C_in:
            raise ValueError(
                f"DiffusionNet C_in={self.C_in}, but x_in has last dim={x_in.shape[-1]}"
            )

        if len(x_in.shape) == 2:
            appended_batch_dim = True
            x_in, mass = x_in.unsqueeze(0), mass.unsqueeze(0)
            if L is not None:
                L = L.unsqueeze(0)
            if evals is not None:
                evals = evals.unsqueeze(0)
            if evecs is not None:
                evecs = evecs.unsqueeze(0)
            if gradX is not None:
                gradX = gradX.unsqueeze(0)
            if gradY is not None:
                gradY = gradY.unsqueeze(0)
            if edges is not None:
                edges = edges.unsqueeze(0)
            if faces is not None:
                faces = faces.unsqueeze(0)

        elif len(x_in.shape) != 3:
            raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")

        x = self.first_lin(x_in)
        for block in self.blocks:
            x = block(x, mass, L, evals, evecs, gradX, gradY, faces=faces)
        x = self.last_lin(x)

        if self.outputs_at == "vertices":
            x_out = x
        elif self.outputs_at == "edges":
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        elif self.outputs_at == "faces":
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)

        if self.last_activation is not None:
            x_out = self.last_activation(x_out)
        if appended_batch_dim:
            x_out = x_out.squeeze(0)
        return x_out


class FaceNet(nn.Module):
    """
    A network that operates on face features.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        C_out: int = 128,
        C_width: int = 64,
        N_block: int = 4,
        inject_dim: int = 9,
        dropout: bool = False,
        out_dim: int = 9,
    ):
        super(FaceNet, self).__init__()
        self.MLP_C = 2 * C_out
        self.mlp_hidden_dims = [C_width, C_width]
        self.dropout = dropout
        self.C_width = C_width
        self.C_out = C_out
        self.first_lin = nn.Linear(inject_dim, 2 * C_out)
        self.jac_decoder = nn.Sequential(
            MiniMLP(
                [self.MLP_C] + self.mlp_hidden_dims + [self.C_width],
                dropout=self.dropout,
            ),
            MiniMLP(
                [self.C_width] + self.mlp_hidden_dims + [self.C_width],
                dropout=self.dropout,
            ),
            MiniMLP(
                [self.C_width] + self.mlp_hidden_dims + [self.C_width],
                dropout=self.dropout,
            ),
            MiniMLP([self.C_width] + self.mlp_hidden_dims + [self.C_out], dropout=True),
        )
        self.conv1 = nn.Conv1d(C_out, C_out, 1)
        self.conv2 = nn.Conv1d(C_out, out_dim, 1)
        self.out_dim = out_dim
        self.relu = nn.ReLU()

    def forward(self, batch_dict: dict) -> torch.Tensor:
        x_0 = self.first_lin(batch_dict["sm_J"].reshape(1, -1, 9))
        x = self.jac_decoder(x_0)
        x_pn = self.relu(self.conv1(x.transpose(1, 2)))
        out_dim_sqrt = int(self.out_dim**0.5)
        Jac = self.conv2(x_pn).transpose(1, 2).reshape(-1, out_dim_sqrt, out_dim_sqrt)
        return Jac


class JacNet(nn.Module):
    """
    A network for decoding Jacobians using DiffusionNet.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        C_out: int = 128,
        C_width: int = 64,
        N_block: int = 4,
        inject_dim: int = 12,
        dropout: bool = False,
        out_dim: int = 9,
        outputs_at: str = "faces",
    ):
        super(JacNet, self).__init__()
        self.jac_decoder = DiffusionNet(
            C_in=latent_dim + inject_dim,
            C_out=C_out,
            C_width=C_width,
            N_block=N_block,
            dropout=dropout,
            outputs_at=outputs_at,
        )
        self.conv1a = nn.Conv1d(C_out, C_out, 1)
        self.conv1b = nn.Conv1d(C_out, C_out, 1)
        self.conv2 = nn.Conv1d(C_out, out_dim, 1)
        self.out_dim = out_dim
        self.relu = nn.ReLU()

    def forward(
        self,
        src_inp_feat: torch.Tensor,
        src_mass: torch.Tensor,
        src_Lap: torch.Tensor,
        src_evals: torch.Tensor,
        src_evecs: torch.Tensor,
        src_gradX: torch.Tensor,
        src_gradY: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        x = self.jac_decoder(
            src_inp_feat,
            src_mass,
            src_Lap,
            src_evals,
            src_evecs,
            src_gradX,
            src_gradY,
            faces=faces,
        )
        x_pn = self.relu(self.conv1b(self.relu(self.conv1a(x.transpose(1, 2)))))
        out_dim_sqrt = int(self.out_dim**0.5)
        Jac = self.conv2(x_pn).transpose(1, 2).reshape(-1, out_dim_sqrt, out_dim_sqrt)
        return Jac


class LJN(nn.Module):
    """
    The main network architecture of LJN.
    """

    def __init__(self, inject_dim: int = 9, outputs_at: str = "faces"):
        super(LJN, self).__init__()
        self.outputs_at = outputs_at
        self.jac_decoder = JacNet(
            latent_dim=0,
            inject_dim=inject_dim,
            N_block=4,
            C_width=256,
            C_out=256,
            out_dim=9,
            outputs_at=outputs_at,
        )

    def forward(self, batch_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        n_v = batch_dict["src_v"].shape[1]
        n_f = batch_dict["src_f"].shape[1]

        if self.outputs_at == "faces":
            sm_S = batch_dict["sm_J_V"].squeeze().reshape(n_v, -1).unsqueeze(0)
        else:
            sm_S = batch_dict["sm_J"].squeeze().reshape(n_f, -1).unsqueeze(0)

        pred_J = self.jac_decoder(
            sm_S,
            batch_dict["src_mass"],
            batch_dict["src_lap"],
            batch_dict["src_evals"],
            batch_dict["src_evecs"],
            batch_dict["src_gradX"],
            batch_dict["src_gradY"],
            batch_dict["src_f"],
        )

        pred_J_ = pred_J.double().cpu()
        pred_pos = self._get_pos(batch_dict, pred_J_)
        return pred_J, pred_pos

    def _get_pos(self, batch_dict: dict, pred_J: torch.Tensor) -> torch.Tensor:
        tar_v = batch_dict["tar_v"].squeeze()
        lap_mat = batch_dict["src_lap_op"][0].cpu().double()
        div_op = batch_dict["src_div"][0].cpu().double()
        lap_op_ch = spth_to_cholespy(lap_mat.cpu())
        pred_pos = (
            SPLUSolveLayer.apply(lap_op_ch, div_op @ pred_J.reshape(-1, 3).cpu())
            .cuda()
            .float()
            .squeeze()
        )

        if self.training:
            pred_pos = pred_pos - pred_pos.mean(dim=0, keepdim=True)
            pred_pos += tar_v.mean(dim=0, keepdim=True)

        return pred_pos
