import torch
import torch.nn as nn
import torch.nn.functional as F

class HFMCA(nn.modules.loss._Loss):

    def __init__(self, device， eps=1e-3, hfmca_version="plus", temperature=0.07, alpha=1.0, beta=1.0):
        """
        alpha: weight for logdet (positive part)
        beta: weight for the negative contrastive part
        """
        super(HFMCA, self).__init__()
        self.device = device
        self.eps = eps
        self.version = version
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(self, multi_feature, multi_output):
        
        # ====== HFMCA ======
        
        f_proj = multi_output.unsqueeze(2).unsqueeze(2)
        g_patches = multi_feature.unsqueeze(3).unsqueeze(2).unsqueeze(2)

        flatten_1 = torch.flatten(f_proj.permute(0, 2, 3, 1), 0, -2)
        flatten_2 = g_patches.permute(0, 2, 3, 4, 5, 1).flatten(0, -2)
        flatten_3 = torch.flatten(g_patches.mean(dim=(-1, -2)).permute(0, 2, 3, 1), 0, -2)

        P = flatten_1.T@flatten_3/flatten_1.shape[0]
        RF = flatten_1.T@flatten_1/flatten_1.shape[0]
        RG = flatten_2.T@flatten_2/flatten_2.shape[0]
        
        iut_dim, output_dim = multi_feature.shape[1], multi_output.shape[1]
        
        RFG = torch.zeros((iut_dim+output_dim, input_dim+output_dim)).cuda()
        RFG[:input_dim, :input_dim] = RF
        RFG[input_dim:, input_dim:] = RG
        RFG[:input_dim, input_dim:] = P
        RFG[input_dim:, :input_dim] = P.T

        RFG = RFG + torch.eye(RFG.shape[0], device=self.device) * self.eps
        RF = RF + torch.eye(RF.shape[0], device=self.device) * self.eps
        RG = RG + torch.eye(RG.shape[0], device=self.device) * self.eps
        
        pos_loss = torch.logdet(RFG) - torch.logdet(RF) - torch.logdet(RG)

        if self.version != "plus":
            return pos_loss

        # ====== HFMCA++ Negative part (contrastive style, but no positives) ======
        N = flatten_1.shape[0]
        z_anchor = F.normalize(flatten_1, dim=1)
        z_positive = F.normalize(flatten_3, dim=1)

        # cosine similarity matrix [N, N]
        sim_matrix = torch.matmul(z_anchor, z_positive.T) / self.temperature

        # mask out diagonal (positives), keep only negatives
        neg_mask = ~torch.eye(N, dtype=torch.bool, device=self.device)

        # for each anchor i, compute log-sum-exp over negatives
        neg_loss = torch.logsumexp(sim_matrix.masked_fill(~neg_mask, -1e9), dim=1).mean()

        # ====== Combine ======
        COST = self.alpha * pos_loss + self.beta * neg_loss
        
        return COST






    
