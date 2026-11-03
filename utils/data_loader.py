import os
import heapq
import random
from collections import Counter, deque
import numpy as np
import dgl
import tqdm
import torch
import pickle
from sklearn.model_selection import train_test_split
from utils.KFold import KFold
import pandas as pd


def load_data(
        data_path='./data',
        device='cpu',
        subgraph_size=32,
        use_subgraph_context=False,
        subgraph_rw_steps=64,
        subgraph_ppr_alpha=0.85,
        subgraph_max_ego_nodes=4096,
        subgraph_ppr_max_pushes=1024,
        subgraph_ppr_epsilon=1e-4,
        subgraph_excluded_edge_types=None,
):
    if os.path.exists(os.path.join(data_path, 'kg_data.pkl')):
        with open(os.path.join(data_path, 'kg_data.pkl'), 'rb') as save_data:
            kg_g, e_feat = pickle.load(save_data)
    else:
        # g
        edges = []
        e_feat = []
        with open(os.path.join(data_path, 'edges.tsv'), 'r') as file:
            for line in tqdm.tqdm(file.readlines(), desc='loading kg'):
                h, r, t = line.strip().split('\t')
                [h, r, t] = [int(h), int(r), int(t)]
                if h == t:
                    continue
                edges.append([h, t])
                edges.append([t, h])
                e_feat.append(r + 1)
                e_feat.append(r + 1)
                # edge_type[(h, t)] = r + 1
                # edge_type[(t, h)] = r + 1

        kg_g = dgl.graph(edges)
        kg_g = dgl.remove_self_loop(kg_g)  # 消除自环
        kg_g = dgl.add_self_loop(kg_g)  # 给孤立的点加自环

        e_feat = np.concatenate([np.array(e_feat), np.zeros(kg_g.num_edges() - len(e_feat), ).astype(np.int64)])

        nodes_list = []
        with open(os.path.join(data_path, 'nodes.tsv'), 'r') as file:
            for line in file.readlines():
                i, node_name, node_type = line.strip().split('\t')
                nodes_list.append([int(i), int(node_type)])
                node_ids = [node[0] for node in nodes_list]

        kg_g.ndata.update({'nodes': torch.tensor(nodes_list)})
        kg_g.edata.update({'edges': torch.from_numpy(e_feat)})

        with open(os.path.join(data_path, 'kg_data.pkl'), 'wb') as file:
            pickle.dump([kg_g, e_feat], file)

    smiles_list = []
    with open(os.path.join(data_path, 'smiles.tsv'), 'r') as file:
        for line in file.readlines():
            smiles = line.strip().split('\t')[1]
            smiles_list.append(smiles)

    if use_subgraph_context:
        subgraph_cache_path = os.path.join(
            data_path,
            f'drug_subgraphs_structured_ppr_rw_v3_k{subgraph_size}.pkl'
        )

        drug_subgraphs = build_or_load_drug_subgraphs(
            kg_g=kg_g,
            drug_num=len(smiles_list),
            cache_path=subgraph_cache_path,
            subgraph_size=subgraph_size,
            rw_steps=subgraph_rw_steps,
            ppr_alpha=subgraph_ppr_alpha,
            max_ego_nodes=subgraph_max_ego_nodes,
            ppr_max_pushes=subgraph_ppr_max_pushes,
            ppr_epsilon=subgraph_ppr_epsilon,
            excluded_edge_types=subgraph_excluded_edge_types,
        )
    else:
        drug_subgraphs = None

    # Structured subgraphs stay on CPU. The encoder transfers only the
    # subgraphs used by the current batch.
    kg_g = kg_g.to(device)

    return kg_g, smiles_list, drug_subgraphs


def get_train_test(data_path='./data', fold_num=5, condition='s1'):
    sample = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values

    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=42, condition=condition)

    final_train_samples = []
    final_valid_samples = []
    final_test_samples = []

    print(f"Start spliting data (Condition: {condition})...")

    for fold_idx, (train_val_data, test_data) in enumerate(kfold.split(sample)):  # kfold.split(sample)生成训练和测试数据，这个循环生成5-折交叉验证的训练、测试数据
        y = train_val_data[:, 2]
        counts = Counter(y)
        invalid_classes = [cls for cls, count in counts.items() if count < 2]

        if invalid_classes:
            mask = np.array([label in invalid_classes for label in y])
            rare_data = train_val_data[mask]
            train_val_data = train_val_data[~mask]
            y = train_val_data[:, 2]
        else:
            rare_data = None

        real_train_data, valid_data = train_test_split(
            train_val_data,
            test_size=0.125,
            random_state=42,
            shuffle=True,
            stratify=y
        )

        if rare_data is not None and len(rare_data) > 0:
            real_train_data = np.concatenate([real_train_data, rare_data])
            
        final_train_samples.append(real_train_data)
        final_valid_samples.append(valid_data)
        final_test_samples.append(test_data)

        print(f"Fold {fold_idx}: Train {len(real_train_data)} | Valid {len(valid_data)} | Test {len(test_data)}")
    print("Data spliting completed")

    return final_train_samples, final_valid_samples, final_test_samples

def build_or_load_drug_subgraphs(
        kg_g,
        drug_num,
        cache_path,
        subgraph_size=32,
        rw_steps=64,
        ppr_alpha=0.85,
        max_ego_nodes=4096,
        ppr_max_pushes=1024,
        ppr_epsilon=1e-4,
        excluded_edge_types=None,
):
    """
    Build structured PPR and random-walk subgraphs.

    Each view stores node identity/type/masks, structural features, all
    induced typed edges in ragged form, and the actual random-walk trace.
    """
    excluded_edge_types = tuple(sorted(int(value) for value in (excluded_edge_types or [])))
    graph = kg_g if str(kg_g.device) == "cpu" else kg_g.to("cpu")
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    max_edge_type = int(graph.edata['edges'].max().item())
    cache_config = {
        "subgraph_size": subgraph_size,
        "rw_steps": rw_steps,
        "ppr_alpha": ppr_alpha,
        "max_ego_nodes": max_ego_nodes,
        "ppr_max_pushes": ppr_max_pushes,
        "ppr_epsilon": ppr_epsilon,
        "excluded_edge_types": excluded_edge_types,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "max_edge_type": max_edge_type,
    }

    if os.path.exists(cache_path):
        print(f"Loading cached drug subgraphs from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if (
            isinstance(cached, dict)
            and cached.get("meta", {}).get("format") == "structured_ppr_rw_v3"
            and cached.get("meta", {}).get("config") == cache_config
        ):
            return cached
        print("Cached subgraph configuration changed; rebuilding.")

    (
        indptr,
        indices,
        typed_indptr,
        typed_indices,
        typed_relations,
    ) = _build_subgraph_adjacency(graph, excluded_edge_types)
    node_types = graph.ndata['nodes'][:, 1].cpu().long().tolist()
    rng = random.Random(42)

    ppr_items = []
    rw_items = []

    print(
        "Building structured drug subgraphs: "
        f"drug_num={drug_num}, subgraph_size={subgraph_size}, "
        f"max_ego_nodes={max_ego_nodes}, ppr_max_pushes={ppr_max_pushes}"
    )

    for drug_id in tqdm.tqdm(
            range(drug_num),
            desc=f"extracting structured subgraphs k={subgraph_size}",
            total=drug_num
    ):
        ego_nodes, _ = _bounded_two_hop_nodes(
            drug_id,
            indptr,
            indices,
            max_nodes=max_ego_nodes,
        )
        ppr_nodes, ppr_scores = _approximate_local_ppr(
            drug_id,
            ego_nodes,
            indptr,
            indices,
            top_k=subgraph_size,
            alpha=ppr_alpha,
            epsilon=ppr_epsilon,
            max_pushes=ppr_max_pushes,
        )
        ppr_items.append(
            _build_structured_subgraph(
                center=drug_id,
                nodes=ppr_nodes,
                node_scores=ppr_scores,
                node_types=node_types,
                indptr=indptr,
                indices=indices,
                typed_indptr=typed_indptr,
                typed_indices=typed_indices,
                typed_relations=typed_relations,
                walk_edges=[],
                visit_counts={},
                first_visit_steps={},
            )
        )

        rw_data = _sample_random_walk_view(
            center=drug_id,
            indptr=indptr,
            indices=indices,
            target_nodes=subgraph_size,
            rw_steps=rw_steps,
            rng=rng,
        )
        rw_items.append(
            _build_structured_subgraph(
                center=drug_id,
                nodes=rw_data["nodes"],
                node_scores=rw_data["node_scores"],
                node_types=node_types,
                indptr=indptr,
                indices=indices,
                typed_indptr=typed_indptr,
                typed_indices=typed_indices,
                typed_relations=typed_relations,
                walk_edges=rw_data["walk_edges"],
                visit_counts=rw_data["visit_counts"],
                first_visit_steps=rw_data["first_visit_steps"],
            )
        )

    subgraphs = {
        "ppr": _pack_structured_view(ppr_items, drug_num, subgraph_size),
        "rw": _pack_structured_view(rw_items, drug_num, subgraph_size),
        "meta": {
            "format": "structured_ppr_rw_v3",
            "config": cache_config,
            "num_nodes": num_nodes,
            "num_node_types": int(max(node_types)) + 1,
            "num_edge_types": max_edge_type + 1,
            "self_loop_edge_type": 0,
        },
    }

    with open(cache_path, "wb") as f:
        pickle.dump(subgraphs, f)

    print(f"Saved structured drug subgraphs to {cache_path}")

    return subgraphs


def _build_subgraph_adjacency(kg_g, excluded_edge_types):
    """Build topology CSR and typed CSR while keeping relation identities."""
    from scipy.sparse import csr_matrix

    src_tensor, dst_tensor = kg_g.edges()
    relation_tensor = kg_g.edata['edges'].long()
    src = src_tensor.cpu().numpy()
    dst = dst_tensor.cpu().numpy()
    relations = relation_tensor.cpu().numpy()
    num_nodes = kg_g.num_nodes()

    valid = src != dst
    if excluded_edge_types:
        valid &= ~np.isin(relations, np.asarray(excluded_edge_types, dtype=relations.dtype))

    filtered_src = src[valid]
    filtered_dst = dst[valid]

    adjacency = csr_matrix(
        (
            np.ones(filtered_src.shape[0], dtype=np.uint8),
            (filtered_src, filtered_dst),
        ),
        shape=(num_nodes, num_nodes),
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1
    adjacency.sort_indices()

    order = np.argsort(filtered_src, kind="stable")
    sorted_src = filtered_src[order]
    typed_indices = filtered_dst[order]
    typed_relations = relations[valid][order]
    counts = np.bincount(sorted_src, minlength=num_nodes)
    typed_indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))

    return (
        adjacency.indptr,
        adjacency.indices,
        typed_indptr,
        typed_indices,
        typed_relations,
    )


def _bounded_two_hop_nodes(center, indptr, indices, max_nodes):
    """Collect a deterministic 2-hop ego neighborhood and sampling hops."""
    ego_nodes = {center}
    hop_distance = {center: 0}
    one_hop = indices[indptr[center]:indptr[center + 1]]

    for node in one_hop:
        node = int(node)
        ego_nodes.add(node)
        hop_distance[node] = 1
        if len(ego_nodes) >= max_nodes:
            return ego_nodes, hop_distance

    for node in one_hop:
        neighbors = indices[indptr[node]:indptr[node + 1]]
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor not in ego_nodes:
                ego_nodes.add(neighbor)
                hop_distance[neighbor] = 2
            if len(ego_nodes) >= max_nodes:
                return ego_nodes, hop_distance

    return ego_nodes, hop_distance


def _approximate_local_ppr(
        center,
        ego_nodes,
        indptr,
        indices,
        top_k,
        alpha,
        epsilon,
        max_pushes,
):
    """
    Approximate personalized PageRank with a residual-push algorithm.

    Work is bounded by max_pushes, so high-degree hubs cannot trigger dozens
    of full PageRank passes over a large induced subgraph.
    """
    scores = {}
    residual = {center: 1.0}
    local_neighbors = {}

    def neighbors_in_ego(node):
        if node not in local_neighbors:
            row = indices[indptr[node]:indptr[node + 1]]
            local_neighbors[node] = [
                int(neighbor) for neighbor in row
                if int(neighbor) in ego_nodes
            ]
        return local_neighbors[node]

    def global_degree(node):
        return max(1, int(indptr[node + 1] - indptr[node]))

    center_degree = global_degree(center)
    queue = [(-1.0 / center_degree, center)]
    pushes = 0

    while queue and pushes < max_pushes:
        neg_priority, node = heapq.heappop(queue)
        mass = residual.get(node, 0.0)
        priority = mass / global_degree(node)

        if priority <= epsilon and pushes > 0:
            break
        if -neg_priority > priority + 1e-15:
            continue

        neighbors = neighbors_in_ego(node)
        residual[node] = 0.0
        scores[node] = scores.get(node, 0.0) + (1.0 - alpha) * mass
        pushes += 1

        if not neighbors:
            residual[center] = residual.get(center, 0.0) + alpha * mass
            center_priority = residual[center] / center_degree
            heapq.heappush(queue, (-center_priority, center))
            continue

        share = alpha * mass / len(neighbors)
        for neighbor in neighbors:
            residual[neighbor] = residual.get(neighbor, 0.0) + share
            neighbor_priority = residual[neighbor] / global_degree(neighbor)
            if neighbor_priority > epsilon:
                heapq.heappush(queue, (-neighbor_priority, neighbor))

    for node, mass in residual.items():
        if mass > 0:
            scores[node] = scores.get(node, 0.0) + mass

    ranked = heapq.nlargest(
        top_k,
        scores.items(),
        key=lambda item: (item[1], -item[0]),
    )
    nodes = [center]
    for node, _ in ranked:
        if node != center:
            nodes.append(int(node))
        if len(nodes) == top_k:
            break

    selected_scores = [float(scores.get(node, 0.0)) for node in nodes]
    score_sum = sum(selected_scores)
    if score_sum > 0:
        selected_scores = [score / score_sum for score in selected_scores]
    elif selected_scores:
        selected_scores[0] = 1.0
    return nodes, selected_scores


def _sample_random_walk_view(center, indptr, indices, target_nodes, rw_steps, rng):
    nodes = [center]
    seen = {center}
    visit_counts = Counter({center: 1})
    first_visit_steps = {center: 0}
    walk_edges = []
    current = center

    for step in range(1, rw_steps + 1):
        start = int(indptr[current])
        end = int(indptr[current + 1])
        if start == end:
            break
        next_node = int(indices[rng.randrange(start, end)])
        walk_edges.append((current, next_node))
        visit_counts[next_node] += 1
        if next_node not in seen:
            nodes.append(next_node)
            seen.add(next_node)
            first_visit_steps[next_node] = step
            if len(nodes) == target_nodes:
                current = next_node
                break
        current = next_node

    total_visits = max(1, sum(visit_counts[node] for node in nodes))
    node_scores = [visit_counts[node] / total_visits for node in nodes]
    return {
        "nodes": nodes,
        "node_scores": node_scores,
        "visit_counts": visit_counts,
        "first_visit_steps": first_visit_steps,
        "walk_edges": walk_edges,
    }


def _build_structured_subgraph(
        center,
        nodes,
        node_scores,
        node_types,
        indptr,
        indices,
        typed_indptr,
        typed_indices,
        typed_relations,
        walk_edges,
        visit_counts,
        first_visit_steps,
):
    unique_nodes = [center]
    seen_nodes = {center}
    score_by_node = {int(node): float(score) for node, score in zip(nodes, node_scores)}
    for node in nodes:
        node = int(node)
        if node not in seen_nodes:
            unique_nodes.append(node)
            seen_nodes.add(node)

    local_index = {node: index for index, node in enumerate(unique_nodes)}
    walk_local_edges = [
        (local_index[source], local_index[destination])
        for source, destination in walk_edges
        if source in local_index and destination in local_index
    ]
    walked_pairs = set(walk_local_edges)

    edge_index = []
    edge_types = []
    edge_walk_flags = []
    seen_edges = set()
    for source in unique_nodes:
        local_source = local_index[source]
        start = int(typed_indptr[source])
        end = int(typed_indptr[source + 1])
        for edge_position in range(start, end):
            destination = int(typed_indices[edge_position])
            relation = int(typed_relations[edge_position])
            if destination not in local_index or source == destination:
                continue
            local_destination = local_index[destination]
            edge_key = (local_source, local_destination, relation)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_index.append((local_source, local_destination))
            edge_types.append(relation)
            edge_walk_flags.append((local_source, local_destination) in walked_pairs)

    for local_node in range(len(unique_nodes)):
        edge_index.append((local_node, local_node))
        edge_types.append(0)
        edge_walk_flags.append(False)

    hop_distance = _induced_hop_distances(
        center_index=0,
        node_count=len(unique_nodes),
        edge_index=edge_index,
    )
    degrees = [
        int(indptr[node + 1] - indptr[node])
        for node in unique_nodes
    ]

    return {
        "node_ids": unique_nodes,
        "node_types": [int(node_types[node]) for node in unique_nodes],
        "center_mask": [index == 0 for index in range(len(unique_nodes))],
        "node_scores": [score_by_node.get(node, 0.0) for node in unique_nodes],
        "node_ranks": list(range(len(unique_nodes))),
        "hop_distance": hop_distance,
        "degree": degrees,
        "visit_count": [int(visit_counts.get(node, 0)) for node in unique_nodes],
        "first_visit_step": [
            int(first_visit_steps.get(node, -1))
            for node in unique_nodes
        ],
        "edge_index": edge_index,
        "edge_types": edge_types,
        "edge_walk_flags": edge_walk_flags,
        "walk_edge_index": walk_local_edges,
    }


def _induced_hop_distances(center_index, node_count, edge_index):
    neighbors = [[] for _ in range(node_count)]
    for source, destination in edge_index:
        if source != destination:
            neighbors[source].append(destination)
            neighbors[destination].append(source)

    distances = [-1] * node_count
    distances[center_index] = 0
    queue = deque([center_index])
    while queue:
        source = queue.popleft()
        for destination in neighbors[source]:
            if distances[destination] >= 0:
                continue
            distances[destination] = distances[source] + 1
            queue.append(destination)
    return distances


def _pack_structured_view(items, drug_num, target_nodes):
    node_ids = torch.zeros((drug_num, target_nodes), dtype=torch.long)
    node_types = torch.zeros((drug_num, target_nodes), dtype=torch.long)
    node_mask = torch.zeros((drug_num, target_nodes), dtype=torch.bool)
    center_mask = torch.zeros((drug_num, target_nodes), dtype=torch.bool)
    node_scores = torch.zeros((drug_num, target_nodes), dtype=torch.float32)
    node_ranks = torch.zeros((drug_num, target_nodes), dtype=torch.float32)
    hop_distance = torch.full((drug_num, target_nodes), -1, dtype=torch.long)
    degree = torch.zeros((drug_num, target_nodes), dtype=torch.float32)
    visit_count = torch.zeros((drug_num, target_nodes), dtype=torch.float32)
    first_visit_step = torch.full((drug_num, target_nodes), -1, dtype=torch.float32)

    all_edges = []
    all_edge_types = []
    all_edge_walk_flags = []
    edge_ptr = [0]
    all_walk_edges = []
    walk_edge_ptr = [0]

    for drug_id, item in enumerate(items):
        nodes = item["node_ids"][:target_nodes]
        node_count = len(nodes)
        if node_count == 0:
            nodes = [drug_id]
            node_count = 1

        node_ids[drug_id, :] = int(nodes[0])
        node_types[drug_id, :] = int(item["node_types"][0])
        node_ids[drug_id, :node_count] = torch.tensor(nodes, dtype=torch.long)
        node_types[drug_id, :node_count] = torch.tensor(
            item["node_types"][:node_count],
            dtype=torch.long,
        )
        node_mask[drug_id, :node_count] = True
        center_mask[drug_id, :node_count] = torch.tensor(
            item["center_mask"][:node_count],
            dtype=torch.bool,
        )
        node_scores[drug_id, :node_count] = torch.tensor(
            item["node_scores"][:node_count],
            dtype=torch.float32,
        )
        node_ranks[drug_id, :node_count] = torch.tensor(
            item["node_ranks"][:node_count],
            dtype=torch.float32,
        )
        hop_distance[drug_id, :node_count] = torch.tensor(
            item["hop_distance"][:node_count],
            dtype=torch.long,
        )
        degree[drug_id, :node_count] = torch.tensor(
            item["degree"][:node_count],
            dtype=torch.float32,
        )
        visit_count[drug_id, :node_count] = torch.tensor(
            item["visit_count"][:node_count],
            dtype=torch.float32,
        )
        first_visit_step[drug_id, :node_count] = torch.tensor(
            item["first_visit_step"][:node_count],
            dtype=torch.float32,
        )

        valid_edges = [
            (edge, relation, walk_flag)
            for edge, relation, walk_flag in zip(
                item["edge_index"],
                item["edge_types"],
                item["edge_walk_flags"],
            )
            if edge[0] < node_count and edge[1] < node_count
        ]
        all_edges.extend(edge for edge, _, _ in valid_edges)
        all_edge_types.extend(relation for _, relation, _ in valid_edges)
        all_edge_walk_flags.extend(walk_flag for _, _, walk_flag in valid_edges)
        edge_ptr.append(len(all_edges))

        valid_walk_edges = [
            edge for edge in item["walk_edge_index"]
            if edge[0] < node_count and edge[1] < node_count
        ]
        all_walk_edges.extend(valid_walk_edges)
        walk_edge_ptr.append(len(all_walk_edges))

    if all_edges:
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    if all_walk_edges:
        walk_edge_index = torch.tensor(all_walk_edges, dtype=torch.long).t().contiguous()
    else:
        walk_edge_index = torch.zeros((2, 0), dtype=torch.long)

    return {
        "node_ids": node_ids,
        "node_types": node_types,
        "node_mask": node_mask,
        "center_mask": center_mask,
        "node_scores": node_scores,
        "node_ranks": node_ranks,
        "hop_distance": hop_distance,
        "degree": degree,
        "visit_count": visit_count,
        "first_visit_step": first_visit_step,
        "edge_index": edge_index,
        "edge_types": torch.tensor(all_edge_types, dtype=torch.long),
        "edge_walk_flags": torch.tensor(all_edge_walk_flags, dtype=torch.bool),
        "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long),
        "walk_edge_index": walk_edge_index,
        "walk_edge_ptr": torch.tensor(walk_edge_ptr, dtype=torch.long),
    }
