import os

import torch
import torch.nn as nn
import dgl
from model.hgnn import HGNN
from rdkit import Chem
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, WeightAndSum, SumPooling
from model.apmp import APMP
from utils.functional_groups import identify_functional_groups


class Mol(nn.Module):
    def __init__(
        self,
        smiles,
        num_hidden,
        num_layer,
        device='cuda:0',
        condition='S1',
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
        super(Mol, self).__init__()
        self.device = device
        self.use_apmp = use_apmp
        self.apmp_input_source = apmp_input_source
        self.apmp_role = apmp_role
        self.use_apmp_parallel = use_apmp_parallel
        self.apmp_parallel_fusion = apmp_parallel_fusion
        if self.apmp_input_source not in {"hgnn", "raw"}:
            raise ValueError(
                "apmp_input_source must be 'hgnn' or 'raw'"
            )
        if self.apmp_role not in {"enhance", "delta"}:
            raise ValueError(
                "apmp_role must be 'enhance' or 'delta'"
            )
        if self.apmp_parallel_fusion not in {"gate", "mean"}:
            raise ValueError(
                "apmp_parallel_fusion must be 'gate' or 'mean'"
            )

        self.readout = AvgPooling()

        build_fg_info = use_apmp and (
            use_fg_path_score
            or save_fg_path_annotations
        )
        mol_g, functional_group_matches = graph_construction(
            smiles,
            build_fg_info=build_fg_info,
        )
        self.functional_group_matches = functional_group_matches
        self.mol_g = dgl.batch(mol_g).to(self.device)

        nodes_type = self.mol_g.ndata['atomic_number'].tolist()
        nodes = []
        for i in range(len(nodes_type)):
            if condition == 'S1':
                nodes.append([i, nodes_type[i]])
            else:
                nodes.append([nodes_type[i], 0])

        nodes = torch.tensor(nodes).to(self.device)

        self.gnn = HGNN(
            self.mol_g,
            self.mol_g.edata['bond_type'],
            nodes,
            num_hidden,
            num_layer
        ).to(device)

        if not condition == 'S1':
            self.gnn.node_embedding = nn.Embedding(
                self.gnn.nodes[:, 0].max() + 2,
                num_hidden
            )

        if self.use_apmp:
            self.apmp = APMP(
                g=self.mol_g,
                hidden_dim=num_hidden,
                role=apmp_role,
                num_paths=apmp_num_paths,
                path_len=apmp_path_len,
                top_ratio=apmp_top_ratio,
                num_layers=apmp_num_layers,
                num_heads=apmp_num_heads,
                chunk_size=apmp_chunk_size,
                use_path_attention=use_apmp_path_attention,
                use_fg_path_score=use_fg_path_score,
                fg_path_weight=fg_path_weight,
                functional_group_matches=functional_group_matches,
                save_fg_path_annotations=save_fg_path_annotations,
                fg_path_annotation_path=fg_path_annotation_path,
                use_path_mask_padding=use_path_mask_padding,
            ).to(self.device)
            self.apmp_gate = None
            self.apmp_branch_scale = None
            self.apmp_fuse_norm = None
            if (
                self.use_apmp_parallel
                and self.apmp_parallel_fusion == "gate"
            ):
                self.apmp_gate = nn.Sequential(
                    nn.Linear(num_hidden * 2, num_hidden),
                    nn.Sigmoid()
                )
                self.apmp_branch_scale = nn.Parameter(torch.tensor(0.1))
                self.apmp_fuse_norm = nn.LayerNorm(num_hidden)
        else:
            self.apmp = None
            self.apmp_gate = None

    def forward(self):
        atom_h = self.gnn()
        base_mol = self.readout(self.mol_g, atom_h)

        if self.apmp is not None:
            apmp_input = self._get_apmp_input(atom_h)
            apmp_atom_h = self.apmp(apmp_input)
            apmp_mol = self.readout(self.mol_g, apmp_atom_h)

            if self.apmp_role == "delta":
                return self._fuse_apmp_delta(base_mol, apmp_mol)

            if self.use_apmp_parallel:
                if self.apmp_parallel_fusion == "mean":
                    result = 0.5 * (base_mol + apmp_mol)
                else:
                    gate = self.apmp_gate(
                        torch.cat([base_mol, apmp_mol], dim=-1)
                    )
                    result = (
                        base_mol
                        + torch.tanh(self.apmp_branch_scale)
                        * gate
                        * apmp_mol
                    )
                    result = self.apmp_fuse_norm(result)
                return result

            return apmp_mol

        return base_mol

    def _get_apmp_input(self, atom_h):
        if self.apmp_input_source == "hgnn":
            return atom_h
        return self.gnn.node_embedding(self.gnn.nodes[:, 0])

    def _fuse_apmp_delta(self, base_mol, apmp_delta_mol):
        if (
            self.use_apmp_parallel
            and self.apmp_parallel_fusion == "gate"
        ):
            gate = self.apmp_gate(
                torch.cat([base_mol, apmp_delta_mol], dim=-1)
            )
            result = (
                base_mol
                + torch.tanh(self.apmp_branch_scale)
                * gate
                * apmp_delta_mol
            )
            return self.apmp_fuse_norm(result)

        return base_mol + apmp_delta_mol

def graph_construction(smiles, build_fg_info=False):
    graphs = []
    functional_group_matches = []
    atom_offset = 0
    for mol_id, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        g = mol_to_bigraph(mol, add_self_loop=True,
                           node_featurizer=PretrainAtomFeaturizer(),
                           edge_featurizer=PretrainBondFeaturizer(),
                           canonical_atom_order=False)
        graphs.append(g)
        if build_fg_info:
            functional_group_matches.extend(
                identify_functional_groups(
                    smi,
                    mol_id=mol_id,
                    atom_offset=atom_offset,
                )
            )
        atom_offset += g.num_nodes()

    return graphs, functional_group_matches
