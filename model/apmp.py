import random
import json
import os
import torch
import torch.nn as nn

from utils.functional_groups import (
    build_functional_group_lookup,
    score_functional_group_path,
)


class APMP(nn.Module):
    """
    Association Pattern-aware Message Passing for molecular graph branch.

    输入:
        atom_feat: [num_atoms, hidden_dim]
    输出:
        role=enhance 时返回增强后的原子表示；role=delta 时返回路径辅助增量。
    """

    def __init__(
        self,
        g,
        hidden_dim,
        role="enhance",
        num_paths=24,
        path_len=6,
        top_ratio=0.4,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        chunk_size=2048,
        use_path_attention=True,
        seed=42,
        use_fg_path_score=False,
        fg_path_weight=0.0,
        functional_group_matches=None,
        save_fg_path_annotations=False,
        fg_path_annotation_path=None,
        use_path_mask_padding=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.role = role
        self.num_paths = num_paths
        self.path_len = path_len
        self.top_ratio = top_ratio
        self.top_k = max(1, int(num_paths * top_ratio))
        self.chunk_size = chunk_size
        self.use_path_attention = use_path_attention
        self.use_path_mask_padding = use_path_mask_padding
        self.use_fg_path_score = use_fg_path_score
        self.fg_path_weight = fg_path_weight
        self.save_fg_path_annotations = save_fg_path_annotations
        self.fg_path_annotation_path = fg_path_annotation_path
        if self.role not in {"enhance", "delta"}:
            raise ValueError(
                "role must be 'enhance' or 'delta'"
            )

        self.path_proj = nn.Linear(path_len * hidden_dim, hidden_dim)

        if self.use_path_attention:
            self.path_attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
        else:
            self.path_attn = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.cpi = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # 小门控，防止一开始 APMP 扰乱原始 HGNN 表示
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.fg_lookup = None
        if (
            self.use_fg_path_score
            or self.save_fg_path_annotations
        ) and functional_group_matches:
            self.fg_lookup = build_functional_group_lookup(
                functional_group_matches,
                g.num_nodes(),
            )

        path_index, path_mask, path_annotations = self._build_path_index(g, seed=seed)
        self.path_annotations = path_annotations
        self.register_buffer("path_index", path_index, persistent=False)
        self.register_buffer("path_mask", path_mask, persistent=False)
        if self.save_fg_path_annotations:
            self._save_path_annotations(path_annotations)

    def _build_path_index(self, g, seed=42):
        """
        为每个原子预采样 top-k 条高置信路径。
        返回:
            path_index: [num_atoms, top_k, path_len]
        """
        random.seed(seed)

        g_cpu = g.cpu()
        num_nodes = g_cpu.num_nodes()

        src, dst = g_cpu.edges()
        src = src.tolist()
        dst = dst.tolist()

        neighbors = [set() for _ in range(num_nodes)]
        for u, v in zip(src, dst):
            if u == v:
                continue
            neighbors[u].add(v)
        neighbors = [sorted(nbrs) for nbrs in neighbors]

        # degree centrality
        deg = torch.zeros(num_nodes)
        for i in range(num_nodes):
            deg[i] = len(set(neighbors[i]))

        if num_nodes > 1:
            centrality = deg / float(num_nodes - 1)
        else:
            centrality = deg

        all_paths = []
        all_path_masks = []
        path_annotations = []

        for start in range(num_nodes):
            candidate_paths = []
            candidate_centrality_scores = []
            candidate_fg_scores = []
            candidate_scores = []
            candidate_annotations = []
            candidate_walk_stats = []
            candidate_path_masks = []

            for _ in range(self.num_paths):
                path, path_mask = self._sample_non_redundant_path(start, neighbors)
                valid_path = [
                    atom for atom, is_valid in zip(path, path_mask)
                    if is_valid
                ]

                centrality_score = float(centrality[valid_path].sum().item())
                if (
                    self.use_fg_path_score
                    or self.save_fg_path_annotations
                ):
                    if self.save_fg_path_annotations:
                        fg_score, annotation = score_functional_group_path(
                            valid_path,
                            self.fg_lookup,
                            return_annotation=True,
                        )
                    else:
                        fg_score = score_functional_group_path(
                            valid_path,
                            self.fg_lookup,
                            return_annotation=False,
                        )
                        annotation = None
                else:
                    fg_score = 0.0
                    annotation = None

                candidate_paths.append(path)
                candidate_centrality_scores.append(centrality_score)
                candidate_fg_scores.append(float(fg_score))
                candidate_annotations.append(annotation)
                candidate_walk_stats.append(self._path_walk_stats(path, path_mask))
                candidate_path_masks.append(path_mask)

            centrality_norm = self._normalize_scores(candidate_centrality_scores)
            fg_norm = self._normalize_scores(candidate_fg_scores)
            if self.use_fg_path_score and self.fg_path_weight > 0:
                candidate_scores = [
                    c_score + self.fg_path_weight * fg_score
                    for c_score, fg_score in zip(centrality_norm, fg_norm)
                ]
            else:
                candidate_scores = candidate_centrality_scores

            order = sorted(
                range(len(candidate_scores)),
                key=lambda i: candidate_scores[i],
                reverse=True,
            )

            selected_ids = order[:self.top_k]
            selected = [candidate_paths[i] for i in selected_ids]
            selected_path_masks = [candidate_path_masks[i] for i in selected_ids]
            all_paths.append(selected)
            all_path_masks.append(selected_path_masks)

            if self.save_fg_path_annotations:
                for rank, candidate_id in enumerate(selected_ids):
                    annotation = candidate_annotations[candidate_id]
                    if annotation is None:
                        _, annotation = score_functional_group_path(
                            [
                                atom
                                for atom, is_valid in zip(
                                    candidate_paths[candidate_id],
                                    candidate_path_masks[candidate_id],
                                )
                                if is_valid
                            ],
                            self.fg_lookup,
                            return_annotation=True,
                        )
                    annotation.update(
                        {
                            "start_atom": int(start),
                            "rank": int(rank),
                            "centrality_score": float(candidate_centrality_scores[candidate_id]),
                            "centrality_score_norm": float(centrality_norm[candidate_id]),
                            "fg_score_norm": float(fg_norm[candidate_id]),
                            "final_score": float(candidate_scores[candidate_id]),
                            "use_path_mask_padding": bool(self.use_path_mask_padding),
                            "path_mask": candidate_path_masks[candidate_id],
                            "valid_path_length": int(sum(candidate_path_masks[candidate_id])),
                            "padding_steps": int(self.path_len - sum(candidate_path_masks[candidate_id])),
                            **candidate_walk_stats[candidate_id],
                        }
                    )
                    path_annotations.append(annotation)

        path_index = torch.LongTensor(all_paths)
        path_mask = torch.FloatTensor(all_path_masks)
        return path_index, path_mask, path_annotations

    def _sample_non_redundant_path(self, start, neighbors):
        cur = start
        prev = None
        path = [start]
        path_mask = [1]
        visited = {start}

        for _step in range(self.path_len - 1):
            nbrs = neighbors[cur]
            candidates = [
                node for node in nbrs
                if node != prev and node not in visited
            ]

            if not candidates:
                if self.use_path_mask_padding:
                    path.extend([cur] * (self.path_len - len(path)))
                    path_mask.extend([0] * (self.path_len - len(path_mask)))
                    break
                candidates = [node for node in nbrs if node != prev]
            if not candidates and not self.use_path_mask_padding:
                candidates = [node for node in nbrs if node not in visited]
            if not candidates and not self.use_path_mask_padding:
                candidates = list(nbrs)
            if not candidates:
                path.extend([cur] * (self.path_len - len(path)))
                path_mask.extend([0] * (self.path_len - len(path_mask)))
                break

            nxt = random.choice(candidates)

            path.append(nxt)
            path_mask.append(1)
            visited.add(nxt)
            prev, cur = cur, nxt

        return path, path_mask

    @staticmethod
    def _path_walk_stats(path, path_mask=None):
        if path_mask is not None:
            path = [
                atom for atom, is_valid in zip(path, path_mask)
                if is_valid
            ]
        self_loop_steps = 0
        immediate_backtracks = 0
        for idx in range(1, len(path)):
            if path[idx] == path[idx - 1]:
                self_loop_steps += 1
            if idx >= 2 and path[idx] == path[idx - 2]:
                immediate_backtracks += 1

        return {
            "self_loop_steps": int(self_loop_steps),
            "immediate_backtracks": int(immediate_backtracks),
            "repeat_visits": int(len(path) - len(set(path))),
        }

    @staticmethod
    def _normalize_scores(scores):
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if abs(max_score - min_score) < 1e-12:
            return [0.0 for _ in scores]
        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    def _save_path_annotations(self, path_annotations):
        if self.fg_path_annotation_path is None:
            return

        output_dir = os.path.dirname(self.fg_path_annotation_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        payload = {
            "num_paths": int(self.num_paths),
            "top_k": int(self.top_k),
            "path_len": int(self.path_len),
            "role": self.role,
            "use_fg_path_score": bool(self.use_fg_path_score),
            "fg_path_weight": float(self.fg_path_weight),
            "use_path_mask_padding": bool(self.use_path_mask_padding),
            "annotations": path_annotations,
        }
        with open(self.fg_path_annotation_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def forward(self, atom_feat):
        """
        atom_feat: [num_atoms, hidden_dim]
        """
        num_atoms = atom_feat.size(0)
        outputs = []

        path_index = self.path_index.to(atom_feat.device)
        path_mask = self.path_mask.to(atom_feat.device)

        for st in range(0, num_atoms, self.chunk_size):
            ed = min(st + self.chunk_size, num_atoms)

            idx = path_index[st:ed]  # [chunk, top_k, path_len]

            # [chunk, top_k, path_len, hidden_dim]
            path_feat = atom_feat[idx]
            path_feat = path_feat * path_mask[st:ed].unsqueeze(-1)

            # [chunk, top_k, path_len * hidden_dim]
            path_feat = path_feat.reshape(
                path_feat.size(0),
                path_feat.size(1),
                self.path_len * self.hidden_dim,
            )

            # [chunk, top_k, hidden_dim]
            path_emb = self.path_proj(path_feat)

            # Common Pattern Interaction
            path_emb = self.cpi(path_emb)

            # 聚合 top-k association patterns
            if self.path_attn is not None:
                attn_logits = self.path_attn(path_emb).squeeze(-1)
                path_weight = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
                pattern_emb = (path_weight * path_emb).sum(dim=1)
            else:
                pattern_emb = path_emb.mean(dim=1)  # [chunk, hidden_dim]

            outputs.append(pattern_emb)

        pattern_emb = torch.cat(outputs, dim=0)

        path_delta = torch.tanh(self.alpha) * self.dropout(pattern_emb)
        if self.role == "delta":
            return path_delta

        enhanced = atom_feat + path_delta
        enhanced = self.norm(enhanced)

        return enhanced
