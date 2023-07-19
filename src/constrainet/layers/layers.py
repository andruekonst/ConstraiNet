import torch


class StepGradModificationFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, a, b, pivot):
        sign_arg = z.detach() @ a.detach() - b.detach()
        # when ind == True, no projection is required
        ind = (sign_arg < 0).unsqueeze(1)
        ray = (pivot - z).detach()
        alpha = (-sign_arg / (ray @ a)).detach()
        # alpha = torch.clamp(alpha, min=0.0, max=1.0)
        ctx.save_for_backward(a, ind, alpha)
        return z + (~ind) * (alpha.unsqueeze(1) * ray)

    @staticmethod
    def backward(ctx, grad_output):
        a, ind, alpha = ctx.saved_tensors
        # gradients for points that satisfy constraint
        grad_z_inside = grad_output * ind
        # gradients for points that violate constraint
        # projection and end point correction
        beta = 1 / (1 - alpha)
        # print(f'shapes: {ind.shape=}; {a.unsqueeze(0).shape=}; {beta.shape=}')
        grad_z_outside = (~ind) * (a.unsqueeze(0) * beta.unsqueeze(1))

        grad_z = grad_z_inside + grad_z_outside
        return grad_z, None, None, None


class StepModifiedGradConstrainLayer(torch.nn.Module):
    def __init__(self, pivot: torch.Tensor, a: torch.Tensor, b: float):
        super().__init__()
        self.pivot = pivot
        self.a = a
        self.b = b

    def forward(self, z):
        return StepGradModificationFn.apply(z, self.a, self.b, self.pivot)


class ShiftSimpleConstrainLayer(torch.nn.Module):
    def __init__(self, pivot: torch.Tensor, a: torch.Tensor, b: float):
        super().__init__()
        self.pivot = pivot
        self.a = a
        self.b = b

    def forward(self, z):
        sign_arg = z @ self.a - self.b
        # when ind == True, no projection is required
        ind = (sign_arg < 0).unsqueeze(1).detach()
        ray = (self.pivot - z).detach()
        alpha = (-sign_arg / torch.clamp_max(ray @ self.a, -1.e-9))

        alpha = torch.clamp(alpha, min=0.0, max=1.0)
        # coef = alpha.pow(2)  # place point inside
        # return z + (~ind) * (alpha.unsqueeze(1) * ray + coef.unsqueeze(1) * ray)
        inside = ind * z
        alpha = alpha.unsqueeze(1)
        gamma = alpha
        p = self.pivot
        outside = (~ind) * ((gamma + (1 - gamma) * alpha) * p + (1 - gamma) * (1 - alpha) * z)
        return inside + outside


class ShiftWithoutMirrorConstrainLayer(torch.nn.Module):
    def __init__(self, pivot: torch.Tensor, a: torch.Tensor, b: float):
        super().__init__()
        self.pivot = pivot
        self.a = a
        self.b = b

    def forward(self, z):
        sign_arg = z @ self.a - self.b
        # when ind == True, no projection is required
        ind = (sign_arg < 0).unsqueeze(1).detach()
        ray = (self.pivot - z)  # .detach()
        alpha = (-sign_arg / torch.clamp_max(ray @ self.a, -1.e-9))

        alpha = torch.clamp(alpha, min=0.0, max=1.0)
        # coef = alpha.pow(2)  # place point inside
        # return z + (~ind) * (alpha.unsqueeze(1) * ray + coef.unsqueeze(1) * ray)
        inside = ind * z
        alpha = alpha.unsqueeze(1)
        gamma = alpha  # .detach()
        p = self.pivot
        outside = (~ind) * ((gamma + (1 - gamma) * alpha) * p + (1 - gamma) * (1 - alpha) * z)
        return inside + outside


class NewStepGradModificationFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, a, b, pivot, lr: float):
        sign_arg = z.detach() @ a.detach() - b.detach()
        # when ind == True, no projection is required
        ind = (sign_arg < 0).unsqueeze(1)
        ray = (pivot - z).detach()
        alpha = (-sign_arg / (ray @ a)).detach()
        # alpha = torch.clamp(alpha, min=0.0, max=1.0)
        ctx.save_for_backward(z, a, b, sign_arg, ind, alpha, lr)
        return z + (~ind) * (alpha.unsqueeze(1) * ray)

    @staticmethod
    def backward(ctx, grad_output):
        z, a, b, sign_arg, ind, alpha, lr = ctx.saved_tensors
        # gradients for points that satisfy constraint
        grad_z_inside = grad_output * ind
        # gradients for points that violate constraint
        # projection and end point correction

        a_norm = a.T @ a
        beta = torch.clamp(-grad_output @ a, 0.0) + sign_arg / a_norm

        grad_z_outside = (~ind) * (grad_output + a.unsqueeze(0) * beta.unsqueeze(1))

        # modify gradients to avoid constraint violation
        next_sign_arg = (z - grad_z_inside * lr) @ a - b
        next_inv_ind = (next_sign_arg > 0).unsqueeze(1)
        # inside_gamma = next_sign_arg / torch.clamp(grad_output @ a, 1.e-9)
        grad_norm = torch.norm(grad_output, dim=1)
        smaller_step_grad_output = grad_output + grad_norm * a.unsqueeze(0) / a_norm
        grad_z_inside = (
            grad_output * ~next_inv_ind +
            smaller_step_grad_output * next_inv_ind
        ) * ind

        grad_z = grad_z_inside + grad_z_outside
        return grad_z, None, None, None


class NewStepModifiedGradConstrainLayer(torch.nn.Module):
    def __init__(self, pivot: torch.Tensor, a: torch.Tensor, b: float):
        super().__init__()
        self.pivot = pivot
        self.a = a
        self.b = b

    def forward(self, z, lr: float = 1.e-5):
        return NewStepGradModificationFn.apply(z, self.a, self.b, self.pivot, lr)


class FusedConstrainFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, At, b, pivot):
        # z shape: (n_samples, n_features)
        # At shape: (n_features, n_constraints)
        sign_arg = z @ At - b.unsqueeze(0)
        # when ind == True, no projection is required
        ind = (sign_arg < 0)
        ray = pivot.unsqueeze(0) - z
        alpha = -sign_arg / (ray @ At)
        # alpha shape: (n_samples, n_constraints)
        # alpha = torch.clamp(alpha, min=0.0, max=1.0)

        # aggregate alpha values among constraints that are violated
        # we have to select the maximum value
        common_alpha = torch.max(alpha * (~ind), dim=1)
        common_ind = torch.all(ind, dim=1)

        ctx.save_for_backward(At, ind, alpha)
        return z + (~common_ind) * (common_alpha.unsqueeze(1) * ray)

    @staticmethod
    def backward(ctx, grad_output):
        At, ind, alpha = ctx.saved_tensors
        # gradients for points that satisfy constraint
        grad_z_inside = grad_output * ind

        # gradients for points that violate constraint
        # projection and end point correction
        beta = 1 / (1 - alpha)
        # TODO: change beta, this is suboptimal

        # average directions

        grad_z_outside = (~ind) * (At.unsqueeze(0) * beta.unsqueeze(1))

        grad_z = grad_z_inside + grad_z_outside
        return grad_z, None, None, None


class FusedConstrainNetwork(torch.nn.Module):
    def __init__(self, pivot: torch.Tensor, A: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.pivot = pivot
        self.At = A.T
        self.b = b

    def forward(self, z):
        return FusedConstrainFn.apply(z, self.At, self.b, self.pivot)
