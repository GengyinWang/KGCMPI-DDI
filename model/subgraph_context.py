import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, ffn_ratio=2.0, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by subgraph_gt_num_heads"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        ffn_hidden_dim = max(1, int(hidden_dim * ffn_ratio))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
        )

    def forward(self, node_feat, node_mask, attn_bias):
        batch_size, node_num, hidden_dim = node_feat.shape
        mask = node_mask.unsqueeze(-1).to(node_feat.dtype)

        residual = node_feat
        x = self.norm1(node_feat)
        qkv = self.qkv(x).reshape(
            batch_size,
            node_num,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn_logits = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) * self.scale
        attn_logits = attn_logits + attn_bias
        key_mask = ~node_mask.unsqueeze(1).unsqueeze(2)
        attn_logits = attn_logits.masked_fill(key_mask, -1e9)

        attn_weight = torch.softmax(attn_logits, dim=-1)
        query_mask = node_mask.unsqueeze(1).unsqueeze(-1)
        attn_weight = attn_weight.masked_fill(~query_mask, 0.0)
        attn_weight = self.attn_dropout(attn_weight)

        context = torch.matmul(attn_weight, value)
        context = context.transpose(1, 2).reshape(
            batch_size,
            node_num,
            hidden_dim,
        )
        node_feat = residual + self.residual_dropout(self.out(context))
        node_feat = node_feat * mask

        residual = node_feat
        node_feat = residual + self.residual_dropout(
            self.ffn(self.norm2(node_feat))
        )
        node_feat = node_feat * mask
        return node_feat


class SubgraphFeatureEncoderBase(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_nodes,
            num_node_types,
            dropout=0.1,
            use_node_id_embedding=True,
            max_hop_bucket=8,
            use_gradient_checkpointing=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_node_id_embedding = use_node_id_embedding
        self.max_hop_bucket = max_hop_bucket
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if use_node_id_embedding:
            self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
            nn.init.xavier_uniform_(self.node_embedding.weight)
        else:
            self.node_embedding = None

        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.hop_embedding = nn.Embedding(max_hop_bucket + 2, hidden_dim)
        self.center_embedding = nn.Embedding(2, hidden_dim)
        self.view_embedding = nn.Embedding(2, hidden_dim)
        self.structural_projection = nn.Linear(5, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.readout_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.score_attention_scale = nn.Parameter(torch.tensor(0.1))
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        nn.init.xavier_uniform_(self.node_type_embedding.weight)
        nn.init.xavier_uniform_(self.hop_embedding.weight)
        nn.init.xavier_uniform_(self.center_embedding.weight)
        nn.init.xavier_uniform_(self.view_embedding.weight)
        nn.init.xavier_uniform_(self.structural_projection.weight)
        nn.init.zeros_(self.structural_projection.bias)

    @staticmethod
    def _normalize_structural_features(batch):
        node_num = max(1, batch["node_ids"].size(1) - 1)
        rank = batch["node_ranks"] / float(node_num)
        degree = torch.log1p(batch["degree"])
        degree = degree / degree.amax(dim=1, keepdim=True).clamp_min(1.0)
        visit_count = torch.log1p(batch["visit_count"])
        visit_count = visit_count / visit_count.amax(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        first_visit_scale = batch["first_visit_step"].clamp_min(0).amax(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        first_visit = torch.where(
            batch["first_visit_step"] >= 0,
            batch["first_visit_step"] / first_visit_scale,
            torch.full_like(batch["first_visit_step"], -1.0),
        )
        return torch.stack([
            batch["node_scores"],
            rank,
            degree,
            visit_count,
            first_visit,
        ], dim=-1)

    @staticmethod
    def _select_batch(view_data, drug_ids, device):
        cpu_ids = drug_ids.detach().to(device="cpu", dtype=torch.long)
        tensor_fields = (
            "node_ids",
            "node_types",
            "node_mask",
            "center_mask",
            "node_scores",
            "node_ranks",
            "hop_distance",
            "degree",
            "visit_count",
            "first_visit_step",
        )
        batch = {
            field: view_data[field].index_select(0, cpu_ids).to(
                device,
                non_blocking=True,
            )
            for field in tensor_fields
        }

        node_num = int(view_data["node_ids"].size(1))
        edge_ptr = view_data["edge_ptr"]
        edge_parts = []
        relation_parts = []
        walk_flag_parts = []
        for batch_index, drug_id in enumerate(cpu_ids.tolist()):
            start = int(edge_ptr[drug_id].item())
            end = int(edge_ptr[drug_id + 1].item())
            if start == end:
                continue
            edge_parts.append(
                view_data["edge_index"][:, start:end] + batch_index * node_num
            )
            relation_parts.append(view_data["edge_types"][start:end])
            walk_flag_parts.append(view_data["edge_walk_flags"][start:end])

        if edge_parts:
            batch["edge_index"] = torch.cat(edge_parts, dim=1).to(
                device,
                non_blocking=True,
            )
            batch["edge_types"] = torch.cat(relation_parts).to(
                device,
                non_blocking=True,
            )
            batch["edge_walk_flags"] = torch.cat(walk_flag_parts).to(
                device,
                non_blocking=True,
            )
        else:
            batch["edge_index"] = torch.zeros(
                (2, 0),
                dtype=torch.long,
                device=device,
            )
            batch["edge_types"] = torch.zeros(
                (0,),
                dtype=torch.long,
                device=device,
            )
            batch["edge_walk_flags"] = torch.zeros(
                (0,),
                dtype=torch.bool,
                device=device,
            )
        return batch


class GraphTransformerSubgraphEncoder(SubgraphFeatureEncoderBase):
    def __init__(
            self,
            hidden_dim,
            num_nodes,
            num_node_types,
            num_edge_types,
            num_layer=2,
            num_heads=4,
            ffn_ratio=2.0,
            dropout=0.1,
            use_node_id_embedding=True,
            max_hop_bucket=8,
            use_edge_bias=True,
            use_gradient_checkpointing=True,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_node_types=num_node_types,
            dropout=dropout,
            use_node_id_embedding=use_node_id_embedding,
            max_hop_bucket=max_hop_bucket,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.num_edge_types = num_edge_types
        self.num_heads = num_heads
        self.use_edge_bias = use_edge_bias
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            for _ in range(num_layer)
        ])

        if use_edge_bias:
            self.edge_type_bias = nn.Embedding(num_edge_types, num_heads)
            self.walk_edge_bias = nn.Embedding(2, num_heads)
            nn.init.zeros_(self.edge_type_bias.weight)
            nn.init.zeros_(self.walk_edge_bias.weight)
        else:
            self.edge_type_bias = None
            self.walk_edge_bias = None

    def forward(self, drug_ids, view_data, view_id, device):
        batch = self._select_batch(view_data, drug_ids, device)
        node_ids = batch["node_ids"]
        node_types = batch["node_types"]
        node_mask = batch["node_mask"]
        center_mask = batch["center_mask"]

        if self.node_embedding is None:
            node_feat = torch.zeros(
                (*node_ids.shape, self.hidden_dim),
                device=device,
            )
        else:
            node_ids = node_ids.clamp(
                min=0,
                max=self.node_embedding.num_embeddings - 1,
            )
            node_feat = self.node_embedding(node_ids)

        node_types = node_types.clamp(
            min=0,
            max=self.node_type_embedding.num_embeddings - 1,
        )
        hop_bucket = batch["hop_distance"].clone()
        hop_bucket[hop_bucket < 0] = self.max_hop_bucket + 1
        hop_bucket = hop_bucket.clamp(max=self.max_hop_bucket + 1)

        structural_features = self._normalize_structural_features(batch)
        view_ids = torch.full_like(node_ids, int(view_id))
        node_feat = (
            node_feat
            + self.node_type_embedding(node_types)
            + self.hop_embedding(hop_bucket)
            + self.center_embedding(center_mask.long())
            + self.view_embedding(view_ids)
            + self.structural_projection(structural_features)
        )
        node_feat = self.input_norm(node_feat)
        node_feat = self.input_dropout(node_feat)
        node_feat = node_feat * node_mask.unsqueeze(-1).to(node_feat.dtype)

        attn_bias = self._build_attention_bias(
            batch,
            node_mask,
            dtype=node_feat.dtype,
        )
        for layer in self.layers:
            if self.training and self.use_gradient_checkpointing:
                node_feat = checkpoint(
                    layer,
                    node_feat,
                    node_mask,
                    attn_bias,
                    use_reentrant=False,
                )
            else:
                node_feat = layer(
                    node_feat,
                    node_mask,
                    attn_bias,
                )
            node_feat = node_feat * node_mask.unsqueeze(-1).to(node_feat.dtype)

        center_weight = center_mask.unsqueeze(-1).to(node_feat.dtype)
        center_emb = (node_feat * center_weight).sum(dim=1)
        center_emb = center_emb / center_weight.sum(dim=1).clamp_min(1.0)

        attention_logits = self.readout_attention(node_feat).squeeze(-1)
        attention_logits = (
            attention_logits
            + self.score_attention_scale * batch["node_scores"]
        )
        attention_logits = attention_logits.masked_fill(~node_mask, -1e9)
        attention_weight = torch.softmax(attention_logits, dim=1)
        graph_emb = (
            attention_weight.unsqueeze(-1) * node_feat
        ).sum(dim=1)
        return self.output(torch.cat([center_emb, graph_emb], dim=-1))

    def _build_attention_bias(self, batch, node_mask, dtype):
        batch_size, node_num = node_mask.shape
        device = node_mask.device
        attn_bias = torch.zeros(
            (batch_size, self.num_heads, node_num, node_num),
            device=device,
            dtype=dtype,
        )
        if (
            not self.use_edge_bias
            or batch["edge_types"].numel() == 0
        ):
            return attn_bias

        source = batch["edge_index"][0]
        destination = batch["edge_index"][1]
        batch_index = torch.div(source, node_num, rounding_mode="floor")
        local_source = source % node_num
        local_destination = destination % node_num
        valid = (
            (batch_index >= 0)
            & (batch_index < batch_size)
            & (local_source >= 0)
            & (local_source < node_num)
            & (local_destination >= 0)
            & (local_destination < node_num)
        )
        if not bool(valid.any()):
            return attn_bias

        edge_types = batch["edge_types"][valid].clamp(
            min=0,
            max=self.num_edge_types - 1,
        )
        edge_bias = self.edge_type_bias(edge_types)
        edge_bias = edge_bias + self.walk_edge_bias(
            batch["edge_walk_flags"][valid].long()
        )
        edge_bias = edge_bias.to(dtype)

        flat_index = (
            (batch_index[valid] * node_num + local_source[valid])
            * node_num
            + local_destination[valid]
        )
        flat_bias = torch.zeros(
            (batch_size * node_num * node_num, self.num_heads),
            device=device,
            dtype=dtype,
        )
        flat_bias.index_add_(0, flat_index, edge_bias)
        return flat_bias.reshape(
            batch_size,
            node_num,
            node_num,
            self.num_heads,
        ).permute(0, 3, 1, 2).contiguous()


class SubgraphContextEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_nodes,
            num_node_types,
            num_edge_types,
            num_layer=2,
            dropout=0.1,
            use_node_id_embedding=True,
            view_fusion="gate",
            kg_subgraph_mode="gate",
            gt_num_heads=4,
            gt_ffn_ratio=2.0,
            gt_use_edge_bias=True,
            use_gradient_checkpointing=True,
            branch_scale_init=0.02,
    ):
        super().__init__()
        if view_fusion not in {"mean", "attention", "gate"}:
            raise ValueError(
                "subgraph view_fusion must be 'mean', 'attention', or 'gate'"
            )
        if kg_subgraph_mode not in {"kg_only", "subgraph_only", "gate"}:
            raise ValueError(
                "kg_subgraph_mode must be 'kg_only', 'subgraph_only', or 'gate'"
            )

        self.view_fusion = view_fusion
        self.kg_subgraph_mode = kg_subgraph_mode
        branch_scale_init = min(max(float(branch_scale_init), 1e-6), 0.99)
        self.branch_scale_logit = nn.Parameter(torch.tensor(
            math.log(branch_scale_init / (1.0 - branch_scale_init))
        ))
        self.shared_encoder = GraphTransformerSubgraphEncoder(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            num_layer=num_layer,
            num_heads=gt_num_heads,
            ffn_ratio=gt_ffn_ratio,
            dropout=dropout,
            use_node_id_embedding=use_node_id_embedding,
            use_edge_bias=gt_use_edge_bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.view_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.view_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid(),
        )
        self.branch_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.branch_gate_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, kg_emb, drug_ids, drug_subgraphs):
        if self.kg_subgraph_mode == "kg_only":
            if kg_emb is None:
                raise ValueError("kg_emb is required in kg_only mode")
            return kg_emb[drug_ids]

        device = (
            kg_emb.device
            if kg_emb is not None
            else self.shared_encoder.node_type_embedding.weight.device
        )
        ppr_emb = self.shared_encoder(
            drug_ids,
            drug_subgraphs["ppr"],
            view_id=0,
            device=device,
        )
        rw_emb = self.shared_encoder(
            drug_ids,
            drug_subgraphs["rw"],
            view_id=1,
            device=device,
        )
        context_emb = self._fuse_views(ppr_emb, rw_emb)

        if self.kg_subgraph_mode == "subgraph_only":
            return context_emb
        if kg_emb is None:
            raise ValueError("kg_emb is required for KG/subgraph gated fusion")

        raw_emb = kg_emb[drug_ids]
        gate_input = torch.cat(
            [raw_emb, context_emb, raw_emb * context_emb],
            dim=-1,
        )
        gate = torch.sigmoid(
            self.branch_gate(gate_input) + self.branch_gate_bias
        )
        scale = torch.sigmoid(self.branch_scale_logit)
        return raw_emb + scale * gate * context_emb

    def _fuse_views(self, ppr_emb, rw_emb):
        if self.view_fusion == "mean":
            return 0.5 * (ppr_emb + rw_emb)
        if self.view_fusion == "attention":
            view_emb = torch.stack([ppr_emb, rw_emb], dim=1)
            weight = torch.softmax(self.view_attention(view_emb), dim=1)
            return (weight * view_emb).sum(dim=1)

        gate_input = torch.cat(
            [ppr_emb, rw_emb, ppr_emb * rw_emb],
            dim=-1,
        )
        gate = self.view_gate(gate_input)
        return gate * ppr_emb + (1.0 - gate) * rw_emb
