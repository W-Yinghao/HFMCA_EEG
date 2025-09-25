import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


    
