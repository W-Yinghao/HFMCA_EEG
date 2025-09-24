import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MoCo(torch.nn.modules.loss._Loss):
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive, queue):
        
        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)
        queue = torch.mm(torch.diag(torch.sum(torch.pow(queue, 2), axis=1) ** (-0.5)), queue)

        # positive logits: Nx1, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [emb_anchor, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
class HFMCA(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=0.5, use_trace=False, hirea=False):
        """
        T: softmax temperature (default: 0.07)
        """
        super(HFMCA, self).__init__()
        self.T = T
        self.device = device
        self.use_trace = use_trace
        self.hirea = hirea

        print("HFMCA: ", use_trace)
        print("Hirea model: ", hirea)
    1
    def forward(self, emb_anchor, emb_positive, iter_num=None, norm_feats=False):

        # print(emb_anchor.shape, emb_positive.shape)

        if self.hirea:
            multi_feature = emb_anchor
            multi_output = emb_positive

            f_proj = multi_output.unsqueeze(2).unsqueeze(2)
            g_patches = multi_feature.unsqueeze(3).unsqueeze(2).unsqueeze(2)

            f_proj = multi_output.unsqueeze(2).unsqueeze(2)

            g_patches = multi_feature.unsqueeze(3).unsqueeze(2).unsqueeze(2)

            flatten_1 = torch.flatten(f_proj.permute(0, 2, 3, 1), 0, -2)

            flatten_2 = g_patches.permute(0, 2, 3, 4, 5, 1).flatten(0, -2)

            flatten_3 = torch.flatten(g_patches.mean(dim=(-1, -2)).permute(0, 2, 3, 1), 0, -2)


        else:

            flatten_1 = emb_anchor
            flatten_2 = emb_positive

            # CONSTRUCT P
            flatten_3 = emb_anchor
        P = flatten_1.T@flatten_3/flatten_1.shape[0]

        RF = flatten_1.T@flatten_1/flatten_1.shape[0]
        RG = flatten_2.T@flatten_2/flatten_2.shape[0]

        input_dim, output_dim = emb_anchor.shape[1], emb_positive.shape[1]
        RFG = torch.zeros((input_dim+output_dim, input_dim+output_dim)).cuda()
        RFG[:input_dim, :input_dim] = RF
        RFG[input_dim:, input_dim:] = RG
        RFG[:input_dim, input_dim:] = P
        RFG[input_dim:, :input_dim] = P.T


        if self.use_trace and iter_num is not None:

            # print("FMCA using trace! ")

            track_cov_final = torch.zeros((input_dim + input_dim)).cuda()
            track_cov_estimate_final = torch.zeros((input_dim + input_dim)).cuda()
            track_cov_final, track_cov_estimate_final, COST = MCA_LOSS_GIVEN_R(RFG, track_cov_final, iter_num, input_dim)
            COST, TSD = return_cost_trace(RFG, track_cov_estimate_final, input_dim)

            # print(COST)

            return COST


    
class FMCA(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=0.5, use_trace=False, hirea=False):
        """
        T: softmax temperature (default: 0.07)
        """
        super(FMCA, self).__init__()
        self.T = T
        self.device = device
        self.use_trace = use_trace
        self.hirea = hirea

        print("FMCA using trace: ", use_trace)
        print("Hirea model: ", hirea)


    def forward(self, emb_anchor, emb_positive, iter_num=None, norm_feats=False):

        # print(emb_anchor.shape, emb_positive.shape)

        if self.hirea and not norm_feats:
            multi_feature = emb_anchor 
            multi_output = emb_positive  

            f_proj = multi_output.unsqueeze(2).unsqueeze(2)
            # print(f'multi_feature:',multi_feature.shape)
            # print(f'multi_output:',multi_output.shape)
            g_patches = multi_feature.unsqueeze(3).unsqueeze(2).unsqueeze(2)
            
            f_proj = multi_output.unsqueeze(2).unsqueeze(2)
            
            g_patches = multi_feature.unsqueeze(3).unsqueeze(2).unsqueeze(2)

            flatten_1 = torch.flatten(f_proj.permute(0, 2, 3, 1), 0, -2)

            flatten_2 = g_patches.permute(0, 2, 3, 4, 5, 1).flatten(0, -2)

            flatten_3 = torch.flatten(g_patches.mean(dim=(-1, -2)).permute(0, 2, 3, 1), 0, -2)


        else:

            flatten_1 = emb_anchor 
            flatten_2 = emb_positive  

            # CONSTRUCT P
            flatten_3 = emb_anchor 

        P = flatten_1.T@flatten_3/flatten_1.shape[0]

        RF = flatten_1.T@flatten_1/flatten_1.shape[0]
        RG = flatten_2.T@flatten_2/flatten_2.shape[0]
        # print(RG) hava values
        input_dim, output_dim = emb_anchor.shape[1], emb_positive.shape[1]
        RFG = torch.zeros((input_dim+output_dim, input_dim+output_dim)).cuda()
        RFG[:input_dim, :input_dim] = RF
        RFG[input_dim:, input_dim:] = RG
        RFG[:input_dim, input_dim:] = P
        RFG[input_dim:, :input_dim] = P.T
        # print(f'RF:',RFG[:input_dim, :input_dim])
        # print(f'RG:',RFG[input_dim:, input_dim:])

        if self.use_trace and iter_num is not None:

            track_cov_final = torch.zeros((input_dim + input_dim)).cuda()
            track_cov_estimate_final = torch.zeros((input_dim + input_dim)).cuda()
            track_cov_final, track_cov_estimate_final, COST = MCA_LOSS_GIVEN_R(RFG, track_cov_final, iter_num, input_dim)
            # print(RFG)
            COST, TSD = return_cost_trace(RFG, track_cov_estimate_final,input_dim)

            # print(TSD)

            return COST

        if norm_feats and iter_num is not None:

            # track_cov_final = torch.zeros((input_dim + input_dim)).cuda()
            # track_cov_estimate_final = torch.zeros((input_dim + input_dim)).cuda()
            # track_cov_final, track_cov_estimate_final, COST = MCA_LOSS_GIVEN_R(RFG, track_cov_final, iter_num, input_dim)
            # COST_trace, TSD = return_cost_trace(RFG.copy(), track_cov_estimate_final)
            
            E1,V1 = torch.linalg.eigh(RF) # track_cov_estimate_final[:input_dim, :input_dim])
            E2,V2 = torch.linalg.eigh(RG) # track_cov_estimate_final[input_dim:, input_dim:])

            RF_NORM = V1@torch.diag(E1**(-1/2))@V1.T
            RG_NORM = V2@torch.diag(E2**(-1/2))@V2.T

            # P = track_cov_estimate_final[:input_dim, input_dim:]
            
            # P_STAR = RF_NORM@P@RG_NORM

            P_STAR = RF@P@RG

            U, S, V = torch.svd(P_STAR)
            # eig_list.append(S.detach().cpu().numpy())
            
            RF_NORM = RF**(-1/2)
            RF_NORM = RF_NORM.detach().cpu()
            U = U.detach().cpu()
            
            # print(RF_NORM.shape, U.shape)
            feats_ml = U*RF_NORM

            # RG_NORM = RG_NORM.detach().cpu()
            # V = V.detach().cpu()

            # eig_list.append(S.detach().cpu().numpy())

            # TSD = S.sum()
            # classifier_error.append(TSD.item())

        eps = 1e-3
        RFG = RFG + torch.eye((RFG.shape[0])).cuda() * eps
        RF = RF + torch.eye((RF.shape[0])).cuda() * eps
        RG = RG + torch.eye((RG.shape[0])).cuda()*  eps

        COST = torch.logdet(RFG) - torch.logdet(RF) - torch.logdet(RG)
        # COST = torch.trace(RFG) - torch.trace(RF) - torch.trace(RG)

        
        if norm_feats:
            return COST, feats_ml
        return COST


# for some reasons the adaptive filter is needed
def adaptive_estimation(v_t, beta, square_term, i):
    # print(v_t)
    # print(beta)
    # print(square_term)
    # print(i)

    v_t = beta*v_t + (1-beta)*square_term.detach()
    # print((v_t/(1-beta**i)))
    return v_t, (v_t/(1-beta**i))


def MCA_LOSS_GIVEN_R(RP, track_cov, i, dim):
    cov = RP + torch.eye((RP.shape[0])).cuda()*(.000001)
    # print(cov)
    # print(RP)
    i+=1
    track_cov, cov_estimate = adaptive_estimation(track_cov, 0.5, cov, i)
    # print(track_cov)
    # print(cov_estimate)
    cov_estimate_f = cov_estimate[:dim, :dim]
    cov_f = cov[:dim, :dim]

    cov_estimate_g = cov_estimate[dim:, dim:]
    cov_g = cov[dim:, dim:]
    # print(cov_estimate_g)
    LOSS = (torch.linalg.inv(cov_estimate)*cov).sum() - (torch.linalg.inv(cov_estimate_f)*cov_f).sum() -(torch.linalg.inv(cov_estimate_g)*cov_g).sum()
    return track_cov, cov_estimate, LOSS


def return_cost_trace(RFG, track_cov_estimate_final,dim):
    # print(RFG.shape)
    RF_E = track_cov_estimate_final[:dim, :dim]
    RG_E = track_cov_estimate_final[dim:, dim:]
    P_E = track_cov_estimate_final[:dim, dim:]
    # print(RF_E)
    RF_EI = torch.inverse(RF_E)
    RG_EI = torch.inverse(RG_E)

    RF = RFG[:dim, :dim]
    RG = RFG[dim:, dim:]
    # print(f'RF:',RF)
    # print(f'RG:',RG)
    
    P = RFG[:dim, dim:]
    # print(f'P:',P)
    COST = -RF_EI@RF@RF_EI@P_E@RG_EI@P_E.T \
            + RF_EI@P@RG_EI@P_E.T \
            - RF_EI@P_E@RG_EI@RG@RG_EI@P_E.T \
            + RF_EI@P_E@RG_EI@P.T
    
    TSD = RF_EI@P_E@RG_EI@P_E.T
    # print(RG)
    return -torch.trace(COST), -torch.trace(TSD)



class BYOL(torch.nn.modules.loss._Loss):
    """

    """
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(BYOL, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):

        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)

        # positive logits: Nxk, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        l_neg = torch.mm(emb_anchor, emb_positive.t())

        loss = - l_pos.sum()
                
        return loss


class SimSiam(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(SimSiam, self).__init__()
        self.T = T
        self.device = device

    def forward(self, p1, p2, z1, z2):

        # L2 normalize
        p1 = F.normalize(p1, p=2, dim=1)
        p2 = F.normalize(p2, p=2, dim=1)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # print(f'p1 shape:',p1.shape)
        # print(f'z1 shape:',z1.shape)
        # print(f'p2 shape:',p2.shape)
        # print(f'z2 shape:',z2.shape)

        # mutual prediction
        l_pos1 = torch.einsum('nc,nc->n', [p1, z2.detach()]).unsqueeze(-1)
        l_pos2 = torch.einsum('nc,nc->n', [p2, z1.detach()]).unsqueeze(-1)

        loss = - (l_pos1.sum() + l_pos2.sum())
                
        return loss


class OurLoss(torch.nn.modules.loss._Loss):

    def __init__(self, device, margin=0.5, sigma=2.0, T=2.0):
        """
        T: softmax temperature (default: 0.07)
        """
        super(OurLoss, self).__init__()
        self.T = T
        self.device = device
        self.margin = margin
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigma = sigma


    def forward(self, emb_anchor, emb_positive):
        
        # L2 normalize, Nxk, Nxk
        emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
        emb_positive = F.normalize(emb_positive, p=2, dim=1)

        # compute instance-aware world representation, Nx1
        sim = torch.mm(emb_anchor, emb_positive.t()) / self.T
        weight = self.softmax(sim)
        neg = torch.mm(weight, emb_positive)

        # representation similarity of pos/neg pairs
        l_pos = torch.exp(-torch.sum(torch.pow(emb_anchor - emb_positive, 2), dim=1) / (2 * self.sigma ** 2))
        l_neg = torch.exp(-torch.sum(torch.pow(emb_anchor - neg, 2), dim=1) / (2 * self.sigma ** 2))

        zero_matrix = torch.zeros(l_pos.shape).to(self.device)
        loss = torch.max(zero_matrix, l_neg - l_pos + self.margin).mean()
        
        return loss


class SimCLR(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=1.0):
        """
        T: softmax temperature (default: 0.07)
        """
        super(SimCLR, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):
        
        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)
        N = emb_anchor.shape[0]
        emb_total = torch.cat([emb_anchor, emb_positive], dim=0)

        # representation similarity matrix, NxN
        logits = torch.mm(emb_total, emb_total.t())
        logits[torch.arange(2*N), torch.arange(2*N)] =  -1e10
        logits /= self.T

        # cross entropy
        labels = torch.LongTensor(torch.cat([torch.arange(N, 2*N), torch.arange(N)])).to(self.device)
        loss = F.cross_entropy(logits, labels)
                
        return loss


class BarlowTwins(torch.nn.modules.loss._Loss):

    def __init__(self, device, lambda_coeff=5e-3):
        """
        from slefeeg
        """
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.device = device


    def forward(self, emb_anchor, emb_positive):

        z1 = emb_anchor
        z2 = emb_positive
        
        N, D = z1.shape
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)

        c_mat = (z1_norm.T @ z2_norm) / N
        c_mat2 = c_mat.pow(2)

        loss = (
            D
            - 2 * torch.trace(c_mat)
            + self.lambda_coeff * torch.sum(c_mat**2)
            + (1 - self.lambda_coeff) * torch.trace(c_mat**2)
        )
        return loss
    

class VicReg(torch.nn.modules.loss._Loss):

    def __init__(self, device, lambda_coeff=25, mu=25, nu=1, epsilon=1e-4):
        """
        from slefeeg
        """
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.device = device
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon


    def forward(self, emb_anchor, emb_positive):

        z1 = emb_anchor
        z2 = emb_positive

        N, D = z1.shape
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # invariance loss
        sim_loss = F.mse_loss(z1, z2)

        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + self.epsilon)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.epsilon)
        std_loss = (torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))) / 2

        # covariance loss
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z1[range(D), range(D)] = 0.0
        cov_z2 = (z2.T @ z2) / (N - 1)
        cov_z2[range(D), range(D)] = 0.0
        cov_loss = cov_z1.pow_(2).sum() / D + cov_z2.pow_(2).sum() / D
        loss = self.lambda_coeff * sim_loss + self.mu * std_loss + self.nu * cov_loss
        return loss
    

class CICL_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device, temperature=0.5):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, emb_anchor, emb_positive):
        features = torch.cat([emb_anchor, emb_positive], dim=0)   # (2N, d)
        N = emb_anchor.shape[0]

        # 1. correlation similarity (matches your code)
        similarity_matrix = torch.corrcoef(features)  # (2N, 2N)

        # 2. remove diagonal (matches your code)
        mask = torch.eye(2 * N, dtype=torch.bool).to(self.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * N, -1)

        # 3. build boolean label matrix (matches your code)
        labels = torch.cat([torch.arange(N) for _ in range(2)], dim=0).to(self.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels[~mask].view(2 * N, -1)

        # 4. positives and negatives (matches your code)
        positives = similarity_matrix[labels.bool()].view(2 * N, -1)
        negatives = similarity_matrix[~labels.bool()].view(2 * N, -1)

        # 5. logits = negatives + positives, pos at last col (matches your code)
        logits = torch.cat([negatives, positives], dim=1) / self.temperature

        # 6. labels = last column index (matches your code)
        target = torch.ones(logits.shape[0], dtype=torch.long).to(self.device) * (logits.shape[1] - 1)

        # 7. cross entropy loss
        loss = self.criterion(logits, target)

        return loss



class FMCA_PosNeg(torch.nn.modules.loss._Loss):
    def __init__(self, device, temperature=0.07, alpha=1.0, beta=1.0):
        """
        alpha: weight for logdet (positive part)
        beta: weight for negative contrastive part
        """
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(self, emb_anchor, emb_positive, step=None):
        # ====== Positive (your original logdet) ======
        multi_feature = emb_anchor
        multi_output = emb_positive  

        f_proj = multi_output.unsqueeze(2).unsqueeze(2)
        g_patches = multi_feature.unsqueeze(3).unsqueeze(2).unsqueeze(2)

        flatten_1 = torch.flatten(f_proj.permute(0, 2, 3, 1), 0, -2)
        flatten_2 = g_patches.permute(0, 2, 3, 4, 5, 1).flatten(0, -2)
        flatten_3 = torch.flatten(g_patches.mean(dim=(-1, -2)).permute(0, 2, 3, 1), 0, -2)

        P = flatten_1.T @ flatten_3 / flatten_1.shape[0]
        RF = flatten_1.T @ flatten_1 / flatten_1.shape[0]
        RG = flatten_2.T @ flatten_2 / flatten_2.shape[0]

        input_dim, output_dim = emb_anchor.shape[1], emb_positive.shape[1]
        RFG = torch.zeros((input_dim+output_dim, input_dim+output_dim)).to(self.device)
        RFG[:input_dim, :input_dim] = RF
        RFG[input_dim:, input_dim:] = RG
        RFG[:input_dim, input_dim:] = P
        RFG[input_dim:, :input_dim] = P.T

        eps = 1e-3
        RFG = RFG + torch.eye(RFG.shape[0], device=self.device) * eps
        RF = RF + torch.eye(RF.shape[0], device=self.device) * eps
        RG = RG + torch.eye(RG.shape[0], device=self.device) * eps

        pos_loss = torch.logdet(RFG) - torch.logdet(RF) - torch.logdet(RG)

        # ====== Negative part (contrastive style, but no positives) ======
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




    
    
def gaussian_kl(S1, S2):
    d = S1.shape[0]
    invS2 = torch.linalg.inv(S2)
    return 0.5 * (torch.trace(invS2 @ S1) - d 
                  + torch.logdet(S2) - torch.logdet(S1))

def jbld(A, B):
    return torch.logdet((A+B)/2) - 0.5*torch.logdet(A) - 0.5*torch.logdet(B)


    
