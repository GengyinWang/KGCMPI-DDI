import torch
import torch.nn as nn

from model.hgnn import HGNN
from model.decoder import Mlp
from model.mol import Mol
from model.subgraph_context import SubgraphContextEncoder


class HetDDI(nn.Module):
    @staticmethod
    def _make_projection(hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

    def __init__(
            self,
            kg_g,
            smiles,
            hidden_dim,
            num_layer,
            mode,
            class_num,
            condition,
            drug_subgraphs=None,
            use_subgraph_context=True,
            subgraph_num_layer=1,
            subgraph_view_fusion="gate",
            kg_subgraph_mode="gate",
            subgraph_gt_num_heads=4,
            subgraph_gt_ffn_ratio=2.0,
            subgraph_gt_use_edge_bias=True,
            use_subgraph_node_id_embedding=False,
            subgraph_branch_scale_init=0.02,
            subgraph_encode_chunk_size=128,
            subgraph_gradient_checkpointing=True,
            use_apmp=False,
            apmp_input_source="hgnn",
            apmp_role="enhance",
            apmp_num_paths=24,
            apmp_path_len=6,
            apmp_top_ratio=0.4,
            apmp_num_layers=1,
            apmp_num_heads=4,
            apmp_chunk_size=2048,
            use_apmp_parallel=True,
            apmp_parallel_fusion="gate",
            use_apmp_path_attention=True,
            use_fg_path_score=False,
            fg_path_weight=0.0,
            save_fg_path_annotations=False,
            fg_path_annotation_path=None,
            use_path_mask_padding=True,
    ):
        super().__init__()
        self.smiles = smiles
        self.device = kg_g.device
        self.mode = mode
        self.drug_num = len(smiles)
        self.drug_subgraphs = drug_subgraphs
        self.use_subgraph_context = (
            use_subgraph_context
            and drug_subgraphs is not None
            and mode in {"only_kg", "concat"}
        )
        self.kg_subgraph_mode = (
            kg_subgraph_mode if self.use_subgraph_context else "kg_only"
        )
        self.subgraph_encode_chunk_size = max(
            1,
            int(subgraph_encode_chunk_size),
        )

        if self.kg_subgraph_mode not in {"kg_only", "subgraph_only", "gate"}:
            raise ValueError(
                "kg_subgraph_mode must be 'kg_only', 'subgraph_only', or 'gate'"
            )

        dropout = 0.1
        if self.mode in {"only_kg", "concat"}:
            self.kg_size = hidden_dim
            needs_global_kg = self.kg_subgraph_mode != "subgraph_only"
            if needs_global_kg:
                self.kg = HGNN(
                    kg_g,
                    kg_g.edata["edges"],
                    kg_g.ndata["nodes"],
                    hidden_dim,
                    num_layer,
                )
                self.kg_size = self.kg.get_output_size()
                self.kg_fc = self._make_projection(self.kg_size, dropout)
            else:
                self.kg = None
                self.kg_fc = None

            if self.use_subgraph_context:
                meta = self.drug_subgraphs.get("meta", {})
                num_nodes = int(meta.get(
                    "num_nodes",
                    int(kg_g.ndata["nodes"][:, 0].max().item()) + 1,
                ))
                num_node_types = int(meta.get(
                    "num_node_types",
                    int(kg_g.ndata["nodes"][:, 1].max().item()) + 1,
                ))
                num_edge_types = int(meta.get(
                    "num_edge_types",
                    int(kg_g.edata["edges"].max().item()) + 1,
                ))
                self.subgraph_context = SubgraphContextEncoder(
                    hidden_dim=self.kg_size,
                    num_nodes=num_nodes,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    num_layer=subgraph_num_layer,
                    dropout=dropout,
                    use_node_id_embedding=use_subgraph_node_id_embedding,
                    view_fusion=subgraph_view_fusion,
                    kg_subgraph_mode=self.kg_subgraph_mode,
                    gt_num_heads=subgraph_gt_num_heads,
                    gt_ffn_ratio=subgraph_gt_ffn_ratio,
                    gt_use_edge_bias=subgraph_gt_use_edge_bias,
                    use_gradient_checkpointing=subgraph_gradient_checkpointing,
                    branch_scale_init=subgraph_branch_scale_init,
                )
            else:
                self.subgraph_context = None

        if self.mode in {"only_mol", "concat"}:
            self.mol = Mol(
                smiles,
                hidden_dim,
                num_layer,
                self.device,
                condition,
                use_apmp=use_apmp,
                apmp_input_source=apmp_input_source,
                apmp_role=apmp_role,
                apmp_num_paths=apmp_num_paths,
                apmp_path_len=apmp_path_len,
                apmp_top_ratio=apmp_top_ratio,
                apmp_num_layers=apmp_num_layers,
                apmp_num_heads=apmp_num_heads,
                apmp_chunk_size=apmp_chunk_size,
                use_apmp_parallel=use_apmp_parallel,
                apmp_parallel_fusion=apmp_parallel_fusion,
                use_apmp_path_attention=use_apmp_path_attention,
                use_fg_path_score=use_fg_path_score,
                fg_path_weight=fg_path_weight,
                save_fg_path_annotations=save_fg_path_annotations,
                fg_path_annotation_path=fg_path_annotation_path,
                use_path_mask_padding=use_path_mask_padding,
            )
            self.mol_size = self.mol.gnn.get_output_size()
            self.mol_fc = self._make_projection(self.mol_size, dropout)

        if self.mode == "only_kg":
            self.decoder = Mlp(self.kg_size, 0, class_num=class_num)
        elif self.mode == "only_mol":
            self.decoder = Mlp(0, self.mol_size, class_num=class_num)
        else:
            self.decoder = Mlp(
                self.kg_size,
                self.mol_size,
                class_num=class_num,
            )

    def forward(self, left, right):
        left = self._as_drug_ids(left)
        right = self._as_drug_ids(right)

        if self.mode == "only_kg":
            left_kg_emb, right_kg_emb = self._encode_kg_pairs(left, right)
            return self.decoder(left_kg_emb, right_kg_emb)

        if self.mode == "only_mol":
            mol_emb = self._encode_mol_branch()
            return self.decoder(mol_emb[left], mol_emb[right])

        left_kg_emb, right_kg_emb = self._encode_kg_pairs(left, right)
        mol_emb = self._encode_mol_branch()
        left_emb = torch.cat([left_kg_emb, mol_emb[left]], dim=-1)
        right_emb = torch.cat([right_kg_emb, mol_emb[right]], dim=-1)
        return self.decoder(left_emb, right_emb)

    def _encode_kg_pairs(self, left, right):
        if self.kg is None:
            kg_emb_all = None
        else:
            kg_emb_all = self.kg_fc(self.kg())

        if self.subgraph_context is None:
            return kg_emb_all[left], kg_emb_all[right]

        pair_ids = torch.cat([left, right], dim=0)
        unique_ids, inverse = torch.unique(
            pair_ids,
            sorted=True,
            return_inverse=True,
        )
        unique_emb = torch.cat([
            self.subgraph_context(
                kg_emb_all,
                unique_ids[start:start + self.subgraph_encode_chunk_size],
                self.drug_subgraphs,
            )
            for start in range(
                0,
                unique_ids.numel(),
                self.subgraph_encode_chunk_size,
            )
        ], dim=0)
        batch_size = left.size(0)
        return (
            unique_emb[inverse[:batch_size]],
            unique_emb[inverse[batch_size:]],
        )

    def _encode_mol_branch(self):
        mol_emb = self.mol()
        return self.mol_fc(mol_emb)

    def _as_drug_ids(self, drug_ids):
        device = next(self.parameters()).device
        if torch.is_tensor(drug_ids):
            return drug_ids.to(
                device=device,
                dtype=torch.long,
            )
        return torch.as_tensor(
            drug_ids,
            device=device,
            dtype=torch.long,
        )
