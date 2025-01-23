import numpy as np
import torch
from torch.nn import Module, Sequential

from .layers import MultiplyLayer, ShiftLayer, TempSigmoid
from ..polyhedron.project import project_onto_hyperplane
from ..polyhedron.check import check_h_nonempty


LEFT_BOUND_LAYER_TYPE = torch.nn.ReLU
BOTH_BOUND_LAYER_TYPE = torch.nn.Sigmoid
# BOTH_BOUND_LAYER_TYPE = (lambda: TempSigmoid(0.01))


class SystemSplitter:
    def __init__(self):
        pass

    def _eliminate_infinite_rows(self, A, b):
        """Assuming the inequality system Ax >= b has at least one solution,
        rows with infinite RHS can be eliminated.
        """
        finite_mask = np.isfinite(b)
        return A[finite_mask], b[finite_mask]

    def _split_system_step(self, A, b):
        # lets choose the first row as a support hyperplane
        R1, s1 = self._find_backprojection_matrix(A[0], b[0])
        A_hat = A[1:] @ R1
        if A_hat.shape[0] == 1:
            q = A[1:] @ A[0]
            if q <= 0:
                b_hat = b[1:] - A[1:] @ s1
            else:
                b_hat = -np.inf
        else:
            try:
                A_hat, b_hat = self._find_projection(
                    A_hat,
                    b=b[1:] - A[1:] @ s1,
                    q=A[1:] @ A[0]
                )
            except Exception as ex:
                print(f'{A_hat.shape=}, {R1.shape=}, {A[1:].shape=}')
                raise ex
        return R1, s1, A_hat, b_hat

    def _split_system(self, A, b):
        """Split the inequality system Ax >= b into pivot-rotation representation.

        Args:
            A: LHS matrix.
            b: RHS vector.

        Returns:
            Tuple (
                List of rotation matrices R_i;
                List of pivots s_i;
                List of LHS matrices (reduced space constraints);
                List of RHS vectors (reduced space constraints);
            )

        """
        TYPE = np.float64
        R = []
        s = []
        A_hat = []
        b_hat = []
        normals = []
        system_for_alpha_A = []
        system_for_alpha_b = []
        cur_A_hat = A
        cur_b_hat = b
        while True:
            prev_A_hat, prev_b_hat = cur_A_hat, cur_b_hat
            cur_R, cur_s, cur_A_hat, cur_b_hat = self._split_system_step(cur_A_hat, cur_b_hat)
            cur_A_hat, cur_b_hat = self._eliminate_infinite_rows(cur_A_hat, cur_b_hat)
            R.append(cur_R.astype(TYPE))
            s.append(cur_s.astype(TYPE))
            A_hat.append(cur_A_hat.astype(TYPE))
            b_hat.append(cur_b_hat.astype(TYPE))
            normals.append(prev_A_hat[0].astype(TYPE))
            system_for_alpha_A.append(prev_A_hat[1:].astype(TYPE))
            system_for_alpha_b.append(prev_b_hat[1:].astype(TYPE))
            if cur_A_hat.shape[1] == 1 or cur_A_hat.shape[0] <= 1:
                break
        return R, s, A_hat, b_hat, normals, system_for_alpha_A, system_for_alpha_b

    def _find_onedim_domain(self, A, b):
        assert A.ndim == 1 or A.shape[1] == 1, f'{A.shape=}, {b.shape=}'
        A = A.cpu().numpy()
        b = b.cpu().numpy()
        A = A.squeeze(1)
        x = b / A
        positive_constraints_mask = A > 0
        negative_constraints_mask = A < 0
        left = np.max(x[positive_constraints_mask], initial=-np.inf)
        right = np.min(x[negative_constraints_mask], initial=np.inf)
        return left, right

    def split(self, A, b):
        assert check_h_nonempty(A, b), "Seems like the system is infeasible"
        self.sys_dim = A.shape[1]
        self.R, self.s, self.A, self.b, self.normals, self.alpha_sys_A, self.alpha_sys_b = self._split_system(
            A, b
        )
        smallest_A, smallest_b = self.A[-1] , self.b[-1]
        if smallest_A.shape[1] == 1:
            start_domain = self._find_onedim_domain(self.A[-1], self.b[-1])
            assert start_domain[0] < start_domain[1], "Something went wrong with defining the initial 1d domain"
        else:
            # n-dimensional domain (underdetermined system)
            pass
        self.R = list(reversed(self.R))
        self.s = list(reversed(self.s))
        self.A = list(reversed(self.A))
        self.b = list(reversed(self.b))
        self.normals = list(reversed(self.normals))
        self.alpha_sys_A = list(reversed(self.alpha_sys_A))
        self.alpha_sys_b = list(reversed(self.alpha_sys_b))


class OrthogonalNet(Module):
    """H-representation based neural network.
    """

    # (batch, dim) -> (batch, 1) \in (0, 1)
    def _get_bounded_nn(self, dim_in):
        return Sequential(
            # torch.nn.Linear(dim_in, 4 * dim_in),
            # torch.nn.Linear(4 * dim_in, dim_in),
            # torch.nn.Linear(dim_in, max(1, dim_in // 2)),
            # torch.nn.Linear(max(1, dim_in // 2), 1),
            torch.nn.Linear(dim_in, 1),
            BOTH_BOUND_LAYER_TYPE()
        )

    # (batch, dim) -> (batch, 1) \in (0, +inf)
    def _get_right_unbounded_nn(self, dim_in):
        return Sequential(
            # torch.nn.Linear(dim_in, max(1, dim_in // 2)),
            # torch.nn.ReLU(),
            # torch.nn.Linear(max(1, dim_in // 2), 1),
            torch.nn.Linear(dim_in, 1),
            LEFT_BOUND_LAYER_TYPE()
        )

    # (batch, dim) -> (batch, 1) \in (-inf, 0)
    def _get_left_unbounded_nn(self, dim_in):
        return Sequential(
            # torch.nn.Linear(dim_in, max(1, dim_in // 2)),
            # torch.nn.ReLU(),
            # torch.nn.Linear(max(1, dim_in // 2), 1),
            torch.nn.Linear(dim_in, 1),
            LEFT_BOUND_LAYER_TYPE(),
            MultiplyLayer(-1),
        )

    # (batch, dim) -> (batch, 1) \in (-inf, +inf)
    def _get_unbounded_nn(self, dim_in):
        return Sequential(
            # torch.nn.Linear(dim_in, dim_in),
            # torch.nn.Tanh(),
            # torch.nn.Linear(dim_in, max(1, dim_in // 2)),
            # torch.nn.Linear(max(1, dim_in // 2), 1),
            torch.nn.Linear(dim_in, 1),
        )

    # (0,1)->(alpha_min, alpha_max)
    def _map_nn_output_to_range(self, nn_output, alpha_min, alpha_max):
        val_range = alpha_max - alpha_min
        nn_output = nn_output * val_range
        nn_output = nn_output + alpha_min
        return nn_output

    def _find_projection(self, A, b, q):
        """Find left and right hand side of unequalities set:
            A x >= r,
            that is equivalent to:
            x in D <=> Exists alpha >= 0: A x >= b - alpha q.
        """
        return project_onto_hyperplane(A, b, q)

    def _find_backprojection_matrix(self, a, b):
        s = b * a.T / np.linalg.norm(a) ** 2
        u, sigma, v = np.linalg.svd(a[np.newaxis])
        R = v[1:].T
        return R, s

    # returns (batch, 1) shape
    def _find_alpha_range(self, A, b, pivot, normal):
        products = A @ normal
        alphas = (b[None, ...] - torch.inner(pivot, A)) / \
            products[None, ...]  # ! same matmul problem
        left_cut = alphas.clone()  # copies gradients
        left_cut[torch.logical_or(alphas < 0, products <= 0)] = 0
        left = torch.max(left_cut, dim=-1, keepdim=True)[0]
        right_cut = alphas
        right_cut[torch.logical_or(alphas < 0, products >= 0)] = torch.inf
        right = torch.min(right_cut, dim=-1, keepdim=True)[0]
        return left, right

    def __init__(self, dim_in: int, A_lb: np.ndarray, b_lb: np.ndarray) -> None:
        super().__init__()
        self.dim_in = dim_in

        if np.isfinite(start_domain[0]) and np.isfinite(start_domain[1]):
            self.nn_first_point = self._get_bounded_nn(dim_in)
            domain_range = start_domain[1] - start_domain[0]
            self.nn_first_point.append(MultiplyLayer(domain_range.item()))
            self.nn_first_point.append(ShiftLayer(start_domain[0].item()))
        elif not np.isfinite(start_domain[0]) and not np.isfinite(start_domain[1]):
            self.nn_first_point = self._get_unbounded_nn(dim_in)
        elif not np.isfinite(start_domain[1]):
            self.nn_first_point = self._get_right_unbounded_nn(dim_in)
            self.nn_first_point.append(ShiftLayer(start_domain[0]))
        else:
            self.nn_first_point = self._get_left_unbounded_nn(dim_in)
            self.nn_first_point.append(ShiftLayer(start_domain[1]))

        self.nn_nu_list = []
        for i in range(self.sys_dim - 1):
            cur_nn_nu = self._get_bounded_nn(dim_in + i + 2)
            # `cur_nn_nu` не обязательно bounded, это зависит от того, какие будут генерироваться
            # границы (`alpha_range`)?
            self.nn_nu_list.append(cur_nn_nu)
            self.add_module(f'nn_nu_{i}', cur_nn_nu)

    def forward(self, x: torch.Tensor):
        assert self.dim_in == x.shape[1], "x has wrong shape"
        cur_x = self.nn_first_point(x)
        for i in range(0, self.sys_dim - 1):
            p = self.s[i][None, ...] + torch.inner(cur_x, self.R[i])  # !should be asked
            alpha_range = self._find_alpha_range(
                self.alpha_sys_A[i], self.alpha_sys_b[i], p, self.normals[i])
            # nu_nn_out = self.nn_nu_list[i](x)  # ! inserting p gives extra low variance
            nu_nn_out = self.nn_nu_list[i](torch.concat((x, p), dim=1))
            nu = self._map_nn_output_to_range(nu_nn_out, *alpha_range)
            cur_x = p + nu * self.normals[i][None, ...]
        return cur_x

    def predict(self, x: np.ndarray, device='cpu'):
        with torch.no_grad():
            x = torch.from_numpy(x.astype(np.double)).to(device)
            return self(x).cpu().numpy()

