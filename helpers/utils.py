import copy
from scipy.optimize import minimize

import torch
import torch.nn.functional as F
import numpy as np

"""
Define task metrics, loss functions and model trainer here. Adapted from https://github.com/lorenmt/auto-lambda
"""


class ConfMatrix(object):
    """
    For mIoU and other pixel-level classification tasks.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def reset(self):
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item()


def get_weight_str(weight, tasks):
    """
    Record task weighting.
    """
    weight_str = 'Task Weighting | '
    for i, task_id in enumerate(tasks):
        weight_str += '{} {:.04f} '.format(task_id, weight[i])
    return weight_str


def get_weight_str_ranked(weight, tasks, rank_num):
    """
    Record top-k ranked task weighting.
    """
    rank_idx = np.argsort(weight)

    if type(tasks) == dict:
        tasks = list(tasks.keys())

    top_str = 'Top {}: '.format(rank_num)
    bot_str = 'Bottom {}: '.format(rank_num)
    for i in range(rank_num):
        top_str += '{} {:.02f} '.format(tasks[rank_idx[-i-1]].title(), weight[rank_idx[-i-1]])
        bot_str += '{} {:.02f} '.format(tasks[rank_idx[i]].title(), weight[rank_idx[i]])

    return 'Task Weighting | {}| {}'.format(top_str, bot_str)


def compute_loss(pred, gt, task_id):
    """
    Compute task-specific loss.
    """
    if task_id in ['seg', 'part_seg'] or 'class' in task_id:
        # Cross Entropy Loss with Ignored Index (values are -1)
        loss = F.cross_entropy(pred, gt, ignore_index=-1)

    if task_id in ['normal', 'depth', 'disp', 'noise']:
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = -1 if task_id == 'disp' else 0
        valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        loss = torch.sum(F.l1_loss(pred, gt, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)
    return loss


class TaskMetric:
    def __init__(self, train_tasks, pri_tasks, batch_size, epochs, include_mtl=False):
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks
        self.batch_size = batch_size
        self.include_mtl = include_mtl
        self.metric = {key: np.zeros([epochs, 2]) for key in train_tasks}  # record loss & task-specific metric
        self.data_counter = 0
        self.epoch_counter = 0
        self.conf_mtx = {}

    def reset(self):
        """
        Reset data counter and confusion matrices.
        """
        self.epoch_counter += 1
        self.data_counter = 0

        if len(self.conf_mtx) > 0:
            for i in self.conf_mtx:
                self.conf_mtx[i].reset()

    def update_metric(self, task_pred, task_gt, task_loss, tasks):
        """
        Update batch-wise metric for each task.
            :param task_pred: [TASK_PRED1, TASK_PRED2, ...]
            :param task_gt: {'TASK_ID1': TASK_GT1, 'TASK_ID2': TASK_GT2, ...}
            :param task_loss: [TASK_LOSS1, TASK_LOSS2, ...]
        """
        
        #everything is [loss_class, loss_regr]
        
        curr_bs = task_pred[0].shape[0]
        r = self.data_counter / (self.data_counter + curr_bs / self.batch_size)
        e = self.epoch_counter
        self.data_counter += 1

        with torch.no_grad():
            #calc classification performance
            self.metric[tasks[0]][e, 0] = r * self.metric[tasks[0]][e, 0] + (1 - r) * task_loss[0]
            pred_label = task_pred[0].max(1)[1]
            acc = pred_label.eq(task_gt[0]).sum().item() / pred_label.shape[0]
            self.metric[tasks[0]][e, 1] = r * self.metric[tasks[0]][e, 1] + (1 - r) * acc

            #calc regression performance
            self.metric[tasks[1]][e, 0] = r * self.metric[tasks[0]][e, 0] + (1 - r) * task_loss[1]
            abs_err = torch.mean(torch.abs(task_pred[1] - task_gt[1]))
            self.metric[tasks[1]][e, 1] = r * self.metric[tasks[1]][e, 1] + (1 - r) * abs_err


"""
Define Gradient-based frameworks here. 
Based on https://github.com/Cranial-XIX/CAGrad/blob/main/cityscapes/utils.py
"""


def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
    U = torch.rand_like(grads[:, 0])
    M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g


def pcgrad(grads, rng, num_tasks):
    grad_vec = grads.t()

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (grad_vec.norm(dim=1, keepdim=True) + 1e-8)  # num_tasks x dim
    modified_grad_vec = copy.deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[task_indices]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)   # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g


def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(
            x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1


def overwrite_grad(m, newgrad, grad_dims, num_tasks):
    newgrad = newgrad * num_tasks  # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

"""
AutoLambda adapted from https://github.com/lorenmt/auto-lambda
"""
class AutoLambda:
    def __init__(self, model, device, train_tasks, pri_tasks, weight_init=0.1):
        self.model = model
        self.model_ = copy.deepcopy(model)
        self.meta_weights = torch.tensor([weight_init] * len(train_tasks), requires_grad=True, device=device)
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks

    def virtual_step(self, train_x, train_y, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        train_pred = self.model(train_x)

        #[logits_class, logits_regr]
        train_loss = self.model_fit(train_pred, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                else:
                    m = 0

                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

    def unrolled_backward(self, train_x, train_y, val_x, val_y, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]

        # compute validation data loss on primary tasks
        if type(val_x) == list:
            val_pred = self.model_(val_x)
        else:
            val_pred = self.model_(val_x)
        val_loss = self.model_fit(val_pred, val_y)
        loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, train_x, train_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = - alpha * h

    def compute_hessian(self, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        if type(train_x) == list:
            train_pred = self.model(train_x)
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        if type(train_x) == list:
            train_pred = self.model(train_x)
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    def model_fit(self, preds, targets):
        """
        define task specific losses
        """
        #pred=[logits_class, logits_regr]
        
        #NOTE
        # because targets are together, class is put as float
        # but cross-entropy cannot handle int as target
        loss_class = F.cross_entropy(preds[0].squeeze(), targets[0].squeeze()).type(torch.float32)
        loss_regr = F.mse_loss(preds[1], targets[1]).type(torch.float32)

        return [loss_class, loss_regr]
    
def get_loss(preds, targets):
    """
    define task specific losses
    """
    #pred=[logits_class, logits_regr]
    
    #NOTE
    # because targets are together, class is put as float
    # but cross-entropy cannot handle int as target
    loss_class = F.cross_entropy(preds[0].squeeze(), targets[0].squeeze()).type(torch.float32)
    loss_regr = F.mse_loss(preds[1], targets[1]).type(torch.float32)

    return [loss_class, loss_regr]
