import torch

# Thanks to authors of NJF


class SPLUSolveLayer(torch.autograd.Function):
    """
    Implements the SPLU solve as a differentiable layer, with a forward and backward function
    """

    @staticmethod
    def forward(ctx, solver, b):
        """
        override forward function
        :param ctx: context object (to keep the lu object for the backward pass)
        :param lu: splu object
        :param b: right hand side, could be a vector or matrix
        :return: the vector or matrix x which holds lu.solve(b) = x
        """
        assert isinstance(b, torch.Tensor)
        assert (
            b.shape[-1] >= 1 and b.shape[-1] <= 3
        ), f"got shape {b.shape} expected last dim to be in range 1-3"
        b = b.contiguous()
        ctx.solver = solver

        # st = time.time()
        vertices = SPLUSolveLayer.solve(solver, b).type_as(b)

        assert not torch.isnan(
            vertices
        ).any(), "Nan in the forward pass of the POISSON SOLVE"
        return vertices

    def backward(ctx, grad_output):
        """
        overrides backward function
        :param grad_output: the gradient to be back-propped
        :return: the outgoing gradient to be back-propped
        """

        assert isinstance(grad_output, torch.Tensor)
        assert (
            grad_output.shape[-1] >= 1 and grad_output.shape[-1] <= 3
        ), f"got shape {grad_output.shape} expected last dim to be in range 1-3"
        grad_output = grad_output.contiguous()
        grad = SPLUSolveLayer.solve(ctx.solver, grad_output)
        assert not torch.isnan(
            grad
        ).any(), "Nan in the backward pass of the POISSON SOLVE"

        return None, grad

    @staticmethod
    def solve(solver, b):
        """
        solve the linear system defined by an SPLU object for a given right hand side. if the RHS is a matrix, solution will also be a matrix.
        :param solver: the splu object (LU decomposition) or cholesky object
        :param b: the right hand side to solve for
        :return: solution x which satisfies Ax = b where A is the poisson system lu describes
        """
        b = b.double().contiguous()
        c = b
        c = c.view(c.shape[0], -1)
        x = torch.zeros_like(c)
        solver.solve(c, x)
        # x = x.view(b.shape[1], b.shape[2], b.shape[0])
        x = x.view(b.shape[0], b.shape[1])
        return x.contiguous()
