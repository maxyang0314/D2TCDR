import torch.nn as nn
import torch
from diffurec import DiffuRec
import torch.nn.functional as F
import torch as th
import random

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num+1
        self.p = args.p
        self.batch_size = args.batch_size
        self.souce_embeddings = nn.Embedding(args.source_item_num+1, self.emb_dim)
        self.target_embeddings = nn.Embedding(args.target_item_num+1, self.emb_dim)
        self.shared_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.specific_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
       
    def diffu_pre(self, item_rep, tag_emb, mask_seq):
        seq_rep_diffu, item_rep_out, weights, t  = self.diffu(item_rep, tag_emb, mask_seq)
        return seq_rep_diffu, item_rep_out, weights, t

    def reverse(self, item_rep, item_rep1, noise_x_t, mask_seq):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, item_rep1, noise_x_t, mask_seq)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))


    def loss_diffu_ce(self, rep_diffu, labels, pretrain_flag, sample_weights=None):
        if pretrain_flag:
            scores = torch.matmul(rep_diffu, self.shared_layer(self.souce_embeddings.weight).t())
        else:
            scores = torch.matmul(rep_diffu, self.target_embeddings.weight.t())
        labels_flat = labels.squeeze(-1)

        # ============================================
        # Original D2TCDR
        # Stage-1 或 D1 沒啟用時
        # ============================================
        if sample_weights is None:
            return self.loss_ce(scores, labels_flat)
        # ============================================
        # D1 Sample-wise weighted CE
        # ============================================
        per_sample_loss = self.loss_ce_rec(scores, labels_flat)
        sample_weights = (sample_weights.view(-1).to(per_sample_loss.device))
        # 所有 sample 原始 supervision 都保留
        base_loss = (per_sample_loss.mean())
        # Popular = 1
        # Unpopular > 1
        # 只取額外增加的 supervision
        extra_weights = (sample_weights - 1.0).clamp_min(0.0)
        extra_loss = (per_sample_loss * extra_weights).mean()
        weighted_loss = (base_loss + extra_loss)
        return weighted_loss

    def diffu_rep_pre(self, rep_diffu, pretrain_flag):
        if pretrain_flag:
            scores = torch.matmul(rep_diffu, self.shared_layer(self.souce_embeddings.weight).t())
        else:
            scores = torch.matmul(rep_diffu, self.target_embeddings.weight.t())
        return scores

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))


    def embedding_loss(
        self,
        item_shared_embeddings,
        con_shared_embeddings,
        item_specific_embeddings,
        con_specific_embeddings,

        source_pop_mask=None,
        source_tail_mask=None,
        target_pop_mask=None,
        target_tail_mask=None,

        group_alignment=False,
        group_alignment_mode='both',
        group_pop_weight=1.0,
        group_tail_weight=1.0,

        lambda_shared=200.0,
        lambda_private=0.2,
        lambda_orthogonal=20
    ):

        # ========================================================
        # Original Orthogonal Loss
        # ========================================================

        item_orthogonal_loss = torch.mean(
            F.cosine_similarity(
                item_shared_embeddings,
                item_specific_embeddings,
                dim=-1
            ) ** 2
        )

        con_orthogonal_loss = torch.mean(
            F.cosine_similarity(
                con_shared_embeddings,
                con_specific_embeddings,
                dim=-1
            ) ** 2
        )

        orthogonal_loss = (
            item_orthogonal_loss
            + con_orthogonal_loss
        )

        # ========================================================
        # Experiment E：
        # Group-aware Shared MMD
        # ========================================================

        if (
            group_alignment
            and source_pop_mask is not None
            and source_tail_mask is not None
            and target_pop_mask is not None
            and target_tail_mask is not None
        ):

            weighted_loss = None
            total_group_weight = 0.0

            # ----------------------------------------------------
            # Popular ↔ Popular
            # ----------------------------------------------------

            if (
                group_alignment_mode in ['both', 'popular_only']
                and group_pop_weight > 0
                and source_pop_mask.any().item()
                and target_pop_mask.any().item()
            ):

                popular_mmd = self.mmd_loss(
                    item_shared_embeddings[
                        source_pop_mask
                    ],
                    con_shared_embeddings[
                        target_pop_mask
                    ]
                )

                popular_term = (
                    group_pop_weight
                    * popular_mmd
                )

                weighted_loss = popular_term

                total_group_weight += (
                    group_pop_weight
                )

            # ----------------------------------------------------
            # Tail ↔ Tail
            # ----------------------------------------------------

            if (
                group_alignment_mode in ['both', 'tail_only']
                and group_tail_weight > 0
                and source_tail_mask.any().item()
                and target_tail_mask.any().item()
            ):

                tail_mmd = self.mmd_loss(
                    item_shared_embeddings[
                        source_tail_mask
                    ],
                    con_shared_embeddings[
                        target_tail_mask
                    ]
                )

                tail_term = (
                    group_tail_weight
                    * tail_mmd
                )

                if weighted_loss is None:
                    weighted_loss = tail_term
                else:
                    weighted_loss = (
                        weighted_loss
                        + tail_term
                    )

                total_group_weight += (
                    group_tail_weight
                )

            # ----------------------------------------------------
            # Normalized weighted average
            # ----------------------------------------------------

            if (
                weighted_loss is not None
                and total_group_weight > 0
            ):

                loss_mmd_shared = (
                    weighted_loss
                    / total_group_weight
                )

            else:

                # 原本 Group-aware 維持舊設定
                if group_alignment_mode == 'both':

                    loss_mmd_shared = self.mmd_loss(
                        item_shared_embeddings,
                        con_shared_embeddings
                    )

                # Popular-only / Tail-only：
                # 該 batch 沒有所指定的 group，就不做 shared alignment
                else:

                    loss_mmd_shared = (
                        item_shared_embeddings.sum()
                        * 0.0
                    )

        else:

            # Original D2TCDR
            loss_mmd_shared = self.mmd_loss(
                item_shared_embeddings,
                con_shared_embeddings
            )

        # ========================================================
        # Private separation：
        # 完全維持原始 D2TCDR
        # ========================================================

        loss_mmd_private = (
            1
            / self.mmd_loss(
                item_specific_embeddings,
                con_specific_embeddings
            )
        )

        # ========================================================
        # Total Alignment Loss
        # ========================================================

        total_loss = (
            lambda_shared
            * loss_mmd_shared

            + lambda_private
            * loss_mmd_private

            + lambda_orthogonal
            * orthogonal_loss
        )

        return total_loss

    def rbf_kernel(self, x, y, sigma=1.0, chunk_size=4096):
        device = x.device
        batch_size_x = x.size(0)
        batch_size_y = y.size(0)
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  
        y_norm = (y ** 2).sum(dim=1, keepdim=True)

        mean_kernel_value = 0.0
        total_elements = 0

        for i in range(0, batch_size_x, chunk_size):
            x_chunk = x[i:i + chunk_size]
            x_chunk_norm = x_norm[i:i + chunk_size]

            for j in range(0, batch_size_y, chunk_size):
                y_chunk = y[j:j + chunk_size]
                y_chunk_norm = y_norm[j:j + chunk_size]

                dist_chunk = x_chunk_norm - 2 * th.matmul(x_chunk, y_chunk.T) + y_chunk_norm.T
                kernel_chunk = th.exp(-dist_chunk / (2 * sigma ** 2)) 

                mean_kernel_value += kernel_chunk.sum()
                total_elements += kernel_chunk.numel()

        mean_kernel_value /= total_elements
        return mean_kernel_value
    
    def mmd_loss(self,x, y, sigma=1.0):
        x = x.mean(dim=1)
        y = y.mean(dim=1)
        k_xx = self.rbf_kernel(x, x, sigma)  
        k_yy = self.rbf_kernel(y, y, sigma)  
        k_xy = self.rbf_kernel(x, y, sigma)  
        mmd_loss = k_xx+ k_yy - 2 * k_xy
        return mmd_loss

    def forward(self, sequence, tag, con_seq, pretrain_flag, args, epoch, train_flag=True, source_pop_mask=None, source_tail_mask=None, target_pop_mask=None, target_tail_mask=None): 
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        mask_seq = (sequence>0).float()
        device = args.device
        em_loss = 0

        if pretrain_flag:
            item_embeddings = self.souce_embeddings(sequence)
            item_embeddings = item_embeddings + position_embeddings
            item_shared_embeddings = self.shared_layer(item_embeddings)
            item_specific_embeddings = item_embeddings - item_shared_embeddings
            item_embeddings = item_shared_embeddings
            
            if train_flag:
                
               
                con_embeddings= self.target_embeddings(con_seq)
                con_embeddings = con_embeddings + position_embeddings

                con_shared_embeddings = self.shared_layer(con_embeddings)
                con_specific_embeddings = con_embeddings - con_shared_embeddings
                em_loss = self.embedding_loss(item_shared_embeddings, con_shared_embeddings, item_specific_embeddings, con_specific_embeddings, source_pop_mask=source_pop_mask,
                    source_tail_mask=source_tail_mask, target_pop_mask=target_pop_mask, target_tail_mask=target_tail_mask,
                    group_alignment=(args.group_alignment == 1), group_alignment_mode=args.group_alignment_mode, group_pop_weight=args.group_pop_weight, group_tail_weight=args.group_tail_weight)
        else:
            item_embeddings = self.target_embeddings(sequence)
            item_embeddings = item_embeddings + position_embeddings
            if train_flag:
                item_shared_embeddings = self.shared_layer(item_embeddings)
                num = random.random()
                
                if num < self.p:
                    item_embeddings = item_shared_embeddings
            else:
                item_shared_embeddings = self.shared_layer(item_embeddings)
                item_shared_embeddings = self.embed_dropout(item_shared_embeddings)
                item_shared_embeddings = self.LayerNorm(item_shared_embeddings)
            
        item_embeddings = self.embed_dropout(item_embeddings)  

        item_embeddings = self.LayerNorm(item_embeddings)
        
        
        if train_flag:
            if pretrain_flag:
                tag_emb = self.shared_layer(self.souce_embeddings(tag.squeeze(-1))) 
            else:
                tag_emb = self.target_embeddings(tag.squeeze(-1))
 
            rep_diffu, rep_item, weights, t = self.diffu(item_embeddings, tag_emb, mask_seq)

            
            item_rep_dis = None
            seq_rep_dis = None
        else:
            noise_x_t = th.randn_like(item_embeddings[:,-1,:])
            if pretrain_flag:
                rep_diffu = self.reverse(item_embeddings, item_embeddings, noise_x_t, mask_seq)
            else:
                rep_diffu = self.reverse(item_embeddings, item_shared_embeddings, noise_x_t, mask_seq)
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None

        scores = None
        return scores, rep_diffu, weights, t, item_rep_dis, seq_rep_dis, em_loss
        

def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre
