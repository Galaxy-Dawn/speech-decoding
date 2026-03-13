import torch
import torch.distributed as dist
import math
from typing import List, Optional, Tuple
from torch import Tensor
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
import torch.nn as nn


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                        dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                            dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()],
                                    params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


def get_muon_param_groups(model, lr_muon=0.02, lr_adam=3e-4, weight_decay=0.01):
    linear_weights = []
    linear_biases = []
    other_params = []

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for _, param in module.named_parameters():
                if param.ndim >= 2:
                    linear_weights.append(param)
                else:
                    linear_biases.append(param)

    all_linear_params = set(linear_weights + linear_biases)
    for _, param in model.named_parameters():
        if param not in all_linear_params:
            other_params.append(param)

    adamw_params = other_params + linear_biases

    param_groups = [
        dict(params=linear_weights, use_muon=True,
             lr=lr_muon, weight_decay=weight_decay),
        dict(params=adamw_params, use_muon=False,
             lr=lr_adam, betas=(0.9, 0.95), weight_decay=weight_decay),
    ]

    return param_groups


class SafeMuonWithAuxAdam(SingleDeviceMuonWithAuxAdam):
    def step(self, closure=None):
        for group in self.param_groups:
            original_params = group["params"]
            valid_params = [p for p in original_params if p.grad is not None]

            if len(valid_params) != len(original_params):
                print(f"Warning: {len(original_params) - len(valid_params)} params have grad=None")

            group["params"] = valid_params

        result = super().step(closure)

        for group in self.param_groups:
            pass

        return result


class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        try:
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


class Adan(Optimizer):
    """ Implements a pytorch variant of Adan.

    Adan was proposed in Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models
    https://arxiv.org/abs/2208.06677

    Arguments:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        betas: Coefficients used for first- and second-order moments.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Decoupled weight decay (L2 penalty)
        no_prox: How to perform the weight decay
        caution: Enable caution from 'Cautious Optimizers'
        foreach: If True would use torch._foreach implementation. Faster but uses slightly more memory.
    """

    def __init__(self,
            params,
            lr: float = 1e-3,
            betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            no_prox: bool = False,
            caution: bool = False,
            foreach: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError('Invalid beta parameter at index 2: {}'.format(betas[2]))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            no_prox=no_prox,
            caution=caution,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)
            group.setdefault('caution', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        try:
            has_scalar_maximum = 'Scalar' in torch.ops.aten._foreach_maximum_.overloads()
        except:
            has_scalar_maximum = False

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            neg_pre_grads = []

            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily supported by making it a tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                if 'neg_pre_grad' not in state or group['step'] == 1:
                    state['neg_pre_grad'] = -p.grad.clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avg_diffs.append(state['exp_avg_diff'])
                neg_pre_grads.append(state['neg_pre_grad'])

            if not params_with_grad:
                continue

            if group['foreach'] is None:
                use_foreach = not group['caution'] or has_scalar_maximum
            else:
                use_foreach = group['foreach']

            if use_foreach:
                func = _multi_tensor_adan
            else:
                func = _single_tensor_adan

            func(
                params_with_grad,
                grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                neg_pre_grads=neg_pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(bias_correction3),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                no_prox=group['no_prox'],
                caution=group['caution'],
            )

        return loss


def _single_tensor_adan(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        exp_avg_diffs: List[Tensor],
        neg_pre_grads: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        bias_correction1: float,
        bias_correction2: float,
        bias_correction3_sqrt: float,
        lr: float,
        weight_decay: float,
        eps: float,
        no_prox: bool,
        caution: bool,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        neg_grad_or_diff = neg_pre_grads[i]

        # for memory saving, we use `neg_grad_or_diff` to get some temp variable in an inplace way
        neg_grad_or_diff.add_(grad)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
        exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff, alpha=1 - beta2)  # diff_t

        neg_grad_or_diff.mul_(beta2).add_(grad)
        exp_avg_sq.mul_(beta3).addcmul_(neg_grad_or_diff, neg_grad_or_diff, value=1 - beta3)  # n_t

        denom = (exp_avg_sq.sqrt() / bias_correction3_sqrt).add_(eps)
        step_size_diff = lr * beta2 / bias_correction2
        step_size = lr / bias_correction1

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            mask = (exp_avg * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            exp_avg = exp_avg * mask

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.addcdiv_(exp_avg, denom, value=-step_size)
            param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)
            param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
            param.div_(1 + lr * weight_decay)

        neg_grad_or_diff.zero_().add_(grad, alpha=-1.0)


def _multi_tensor_adan(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        exp_avg_diffs: List[Tensor],
        neg_pre_grads: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        bias_correction1: float,
        bias_correction2: float,
        bias_correction3_sqrt: float,
        lr: float,
        weight_decay: float,
        eps: float,
        no_prox: bool,
        caution: bool,
):
    if len(params) == 0:
        return

    # for memory saving, we use `neg_pre_grads` to get some temp variable in a inplace way
    torch._foreach_add_(neg_pre_grads, grads)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, neg_pre_grads, alpha=1 - beta2)  # diff_t

    torch._foreach_mul_(neg_pre_grads, beta2)
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs, neg_pre_grads, neg_pre_grads, value=1 - beta3)  # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    step_size_diff = lr * beta2 / bias_correction2
    step_size = lr / bias_correction1

    if caution:
        # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
        masks = torch._foreach_mul(exp_avgs, grads)
        masks = [(m > 0).to(g.dtype) for m, g in zip(masks, grads)]
        mask_scale = [m.mean() for m in masks]
        torch._foreach_maximum_(mask_scale, 1e-3)
        torch._foreach_div_(masks, mask_scale)
        exp_avgs = torch._foreach_mul(exp_avgs, masks)

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)
    else:
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)
        torch._foreach_div_(params, 1 + lr * weight_decay)

    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)


def get_optimizer(model, optim_type, lr, lr_muon, weight_decay) -> Optimizer:
    if optim_type == 'adan':
        return Adan(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'muon':
        param_groups = get_muon_param_groups(model, lr_muon=lr_muon, lr_adam=lr, weight_decay=weight_decay)
        return SafeMuonWithAuxAdam(param_groups)
    else:
        raise NotImplementedError(optim_type)