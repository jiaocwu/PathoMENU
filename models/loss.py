import torch
import torch.nn as nn
import torch.nn.functional as F








class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


def get_diff_loss(hg_s, hg_p, hi_s, hi_p, cross=False):
    loss_diff = DiffLoss()

    loss = loss_diff(hg_s, hg_p)
    loss += loss_diff(hi_s, hi_p)

    if cross:
        loss += loss_diff(hg_p, hi_p)

    return loss


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        m = (0.5 * (p + q)).log()

        kl_p = self.kl(m, p.log())
        kl_q = self.kl(m, q.log())

        jsd_element = 0.5 * (kl_p + kl_q)

        return jsd_element.sum(dim=1)


class DisentanglementLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.jsd = JSD()
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)

    def _abs_cos_sim(self, x, y):

        return self.alignment_cos_sim(x, y).abs()

    def _global_ortho_loss(self, x, y):
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        x_norm = F.normalize(x, p=2, dim=0)
        y_norm = F.normalize(y, p=2, dim=0)


        correlation_matrix = torch.mm(x_norm.T, y_norm)

        return correlation_matrix.pow(2).mean()



    def forward(self, shared1, private1, shared2, private2):
        local_ortho = self._abs_cos_sim(shared1, private1) + self._abs_cos_sim(
            shared2, private2
        )
        loss_ortho = local_ortho

        loss_align = self.jsd(shared1.sigmoid(), shared2.sigmoid())

        return loss_ortho, loss_align



class Objective(nn.Module):
    def __init__(self, batch_size, temperature_f=1.0):
        super(Objective, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, n):
        mask = torch.ones((n, n)).fill_diagonal_(0)
        for i in range(n // 2):
            mask[i, n // 2 + i] = 0
            mask[n // 2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, h_i, h_j):
        n = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(-1, 1)
        mask = self.mask_correlated_samples(n)
        negative_samples = sim[mask].reshape(n, -1)

        labels = torch.zeros(n).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss = loss / n
        return loss


def scale_mse(recon_x, x, alpha=0.0):
    return F.mse_loss(alpha * (x - recon_x).mean(dim=1, keepdims=True) + recon_x, x)


def contrast_loss(feat1, feat2, tau=0.1, weight=1.0):
    sim_matrix = torch.einsum("ik, jk -> ij", feat1, feat2) / torch.einsum(
        "i, j -> ij", feat1.norm(p=2, dim=1), feat2.norm(p=2, dim=1)
    )
    label = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss = F.cross_entropy(input=sim_matrix / tau, target=label) * weight
    return loss


def trivial_entropy(feat, tau=0.1, weight=1.0):
    prob_x = F.softmax(feat / tau, dim=1)
    p = prob_x / prob_x.norm(p=1).sum(0)
    loss = (p * torch.log(p + 1e-8)).sum() * weight
    return loss


def cross_instance_loss(feat1, feat2, tau=0.1, weight=1.0):
    sim_matrix = torch.einsum("ik, jk -> ij", feat1, feat2) / torch.einsum(
        "i, j -> ij", feat1.norm(p=2, dim=1), feat2.norm(p=2, dim=1)
    )
    entropy = (
            torch.distributions.Categorical(logits=sim_matrix / tau).entropy().mean()
            * weight
    )
    return entropy


def kl_loss(feat_x1, feat_x2, tau):
    return torch.nn.KLDivLoss()(
        (feat_x1 / tau).log_softmax(1), (feat_x2 / tau).softmax(1)
    )
