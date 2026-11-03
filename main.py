import os
import sys
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
from model.HetDDI import HetDDI
from utils.data_loader import load_data, get_train_test
from train_test import train_one_epoch, test
from utils.pytorchtools import EarlyStopping
from utils.logger import Logger
from utils.losses import Tailed_FocalLoss, get_class_counts, CE_TailedFocalLoss


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected: True or False')


def parse_int_list(value):
    if value is None or str(value).strip() == '':
        return []
    try:
        return [int(item.strip()) for item in str(value).split(',') if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'Expected a comma-separated integer list.'
        ) from error


def format_args_one_line(args):
    """将运行参数整理为单行，便于日志检索和实验复现。"""
    return "运行参数：" + " | ".join(
        f"{name}={value}" for name, value in vars(args).items()
    )


def format_parameter_relations():
    """集中说明 main.py 中关键参数之间的启用关系。"""
    return (
        "参数关联说明：\n"
        "1. data_path、kg_name、ddi_name 共同组成数据目录：data_path/(kg_name+ddi_name)。\n"
        "2. condition 决定 S1/S2/S3 数据划分和分子节点初始化方式；fold_num 决定交叉验证折数。\n"
        "3. mode 控制使用 KG 分支、分子分支或二者拼接；only_kg 下 APMP 分子参数不生效，only_mol 下 KG 子图参数不生效。\n"
        "4. use_subgraph_context 仅在 mode 为 only_kg/concat 且 kg_subgraph_mode 不为 kg_only 时触发子图加载和编码。\n"
        "5. subgraph_size、subgraph_rw_steps、subgraph_ppr_* 和 subgraph_excluded_edge_types 共同控制药物 KG 子图抽取。\n"
        "6. 子图固定使用 Graph Transformer 编码，subgraph_gt_num_heads 需要整除 hidden_dim。\n"
        "7. kg_subgraph_mode 决定全局 KG 与子图表示的关系：kg_only 只用全局 KG，subgraph_only 只用子图，gate 做门控融合。\n"
        "8. use_apmp 仅在 mode 为 only_mol/concat 时启用；apmp_num_paths 与 apmp_top_ratio 共同决定规则模式下 top-k 路径数量。\n"
        "9. apmp_input_source 控制 APMP 使用 HGNN 后特征或原始节点 embedding；apmp_role 控制 APMP 输出完整增强表示或辅助增量。\n"
        "10. apmp_path_len 控制路径长度；use_path_mask_padding 控制路径不足时使用 mask padding 还是回退/重复补齐。\n"
        "11. apmp_num_layers、apmp_num_heads 与 hidden_dim 共同决定 APMP Transformer 编码器结构，apmp_num_heads 需要能整除 hidden_dim。\n"
        "12. use_fg_path_score 与 fg_path_weight 影响规则路径选择分数；save_fg_path_annotations 只保存解释信息，不直接改变预测。"
    )




def run(args):
    # np.random.seed(42)
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(42)

    data_path = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name)
    use_kg_subgraph_context = (
        args.use_subgraph_context
        and args.mode != 'only_mol'
        and args.kg_subgraph_mode != 'kg_only'
    )
    # kg_g, smiles = load_data(data_path, device=device)  # STEP 2、返回加载的知识图谱和SMILES列表
    # kg_g, smiles, drug_subgraphs = load_data(data_path, device=device)
    kg_g, smiles, drug_subgraphs = load_data(
        data_path,
        device=device,
        subgraph_size=args.subgraph_size,
        use_subgraph_context=use_kg_subgraph_context,
        subgraph_rw_steps=args.subgraph_rw_steps,
        subgraph_ppr_alpha=args.subgraph_ppr_alpha,
        subgraph_max_ego_nodes=args.subgraph_max_ego_nodes,
        subgraph_ppr_max_pushes=args.subgraph_ppr_max_pushes,
        subgraph_ppr_epsilon=args.subgraph_ppr_epsilon,
        subgraph_excluded_edge_types=args.subgraph_excluded_edge_types,
    )
    train_sample, valid_sample, test_sample = get_train_test(
        data_path,
        fold_num=args.fold_num,
        condition=args.condition)    # STEP 3、返回5折训练样本和测试样本，并且每折生成了相同数量的负训练样本和负测试样本

    # # save train and test sample
    # for i in range(5):
    #     np.savetxt(f'train_{i}.txt', train_sample[i], fmt='%d')
    #     np.savetxt(f'valid_{i}.txt', valid_sample[i], fmt='%d')
    #     np.savetxt(f'test_{i}.txt', test_sample[i], fmt='%d')

    scores = []
    run_timestamp = getattr(args, 'run_timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
    for i in range(0, args.fold_num):
        print(f"============================ Fold {i} Start ============================")
        model_save_dir = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name, args.condition, f'Fold{i}')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        best_model_path = os.path.join(
            model_save_dir,
            f'best_model_{args.num_layer}_{args.hidden_dim}_{run_timestamp}.pth'
        )
        fg_path_annotation_path = None
        if args.save_fg_path_annotations:
            fg_path_annotation_path = os.path.join(
                model_save_dir,
                f'fg_path_annotations_{run_timestamp}.json'
            )

        train_x_left = train_sample[i][:, 0]
        train_x_right = train_sample[i][:, 1]
        train_y = train_sample[i][:, 2:]

        valid_x_left = valid_sample[i][:, 0]
        valid_x_right = valid_sample[i][:, 1]
        valid_y = valid_sample[i][:, 2:]

        test_x_left = test_sample[i][:, 0]
        test_x_right = test_sample[i][:, 1]
        test_y = test_sample[i][:, 2:]

        train_y = torch.from_numpy(train_y).long()
        valid_y = torch.from_numpy(valid_y).long()
        test_y = torch.from_numpy(test_y).long()

        # STEP 4、load model
        class_num = 65 if args.ddi_name == 'Deng' else 86
        # model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, class_num, args.condition).to(device)
        model = HetDDI(
            kg_g,
            smiles,
            args.hidden_dim,
            args.num_layer,
            args.mode,
            class_num,
            args.condition,
            drug_subgraphs=drug_subgraphs,
            use_subgraph_context=use_kg_subgraph_context,
            subgraph_num_layer=args.subgraph_num_layer,
            subgraph_view_fusion=args.subgraph_view_fusion,
            kg_subgraph_mode=args.kg_subgraph_mode,
            subgraph_gt_num_heads=args.subgraph_gt_num_heads,
            subgraph_gt_ffn_ratio=args.subgraph_gt_ffn_ratio,
            subgraph_gt_use_edge_bias=args.subgraph_gt_use_edge_bias,
            use_subgraph_node_id_embedding=args.use_subgraph_node_id_embedding,
            subgraph_branch_scale_init=args.subgraph_branch_scale_init,
            subgraph_encode_chunk_size=args.subgraph_encode_chunk_size,
            subgraph_gradient_checkpointing=args.subgraph_gradient_checkpointing,
            use_apmp=args.use_apmp,
            apmp_input_source=args.apmp_input_source,
            apmp_role=args.apmp_role,
            apmp_num_paths=args.apmp_num_paths,
            apmp_path_len=args.apmp_path_len,
            apmp_top_ratio=args.apmp_top_ratio,
            apmp_num_layers=args.apmp_num_layers,
            apmp_num_heads=args.apmp_num_heads,
            apmp_chunk_size=args.apmp_chunk_size,
            use_apmp_parallel=args.use_apmp_parallel,
            apmp_parallel_fusion=args.apmp_parallel_fusion,
            use_apmp_path_attention=args.use_apmp_path_attention,
            use_fg_path_score=args.use_fg_path_score,
            fg_path_weight=args.fg_path_weight,
            save_fg_path_annotations=args.save_fg_path_annotations,
            fg_path_annotation_path=fg_path_annotation_path,
            use_path_mask_padding=args.use_path_mask_padding,
        ).to(device)
        # loss_func = nn.CrossEntropyLoss()
        class_counts = get_class_counts(train_sample[i], num_classes=class_num)
        # loss_func = Tailed_FocalLoss(
        #     class_counts=class_counts,
        #     n_classes=class_num,
        #     gamma=2.0,
        #     beta=3.0,
        #     ts=0.9,
        #     device=device,
        # )
        loss_func = CE_TailedFocalLoss(
            class_counts=class_counts,
            n_classes=class_num,
            gamma=2.0,
            beta=1.5,
            ts=0.85,
            ce_weight=0.7,
            tail_weight=0.3,
            device=device,
        )

        if i == 0:
            print(model)

        # divide parameters into two parts, weight_p has l2_norm but bias_bn_emb_p not
        weight_p, bias_bn_emb_p = [], []
        for name, p in model.named_parameters():
            if 'subgraph_context.shared_encoder.node_embedding' in name:
                weight_p += [p]
            elif 'bias' in name or 'bn' in name or 'embedding' in name:
                bias_bn_emb_p += [p]
            else:
                weight_p += [p]
        model_parameters = [
            {'params': weight_p, 'weight_decay': args.weight_decay},
            {'params': bias_bn_emb_p, 'weight_decay': 0},
        ]
        # train setting
        optimizer = optim.AdamW(model_parameters, lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        best_val_score = None
        for epoch in range(args.epoch):
            train_one_epoch(model, loss_func, optimizer, train_x_left, train_x_right, train_y,
                            i, epoch, args.batch_size, device)
            
            val_score = test(model, loss_func, valid_x_left, valid_x_right, valid_y, 
                            i, epoch, args.batch_size, device)

            val_f1 = val_score[1]   # best score set to F1
            # test_acc = test_score[1]    # best score set to ACC
            if epoch > 50:
                early_stopping(val_f1, model)
                if early_stopping.counter == 0:
                    # best_test_score = test_score
                    best_val_score = val_score
                    # save model
                    torch.save(model.state_dict(), best_model_path)
                    print(
                        "\n========== Fold {} Epoch {} Validation Improved. "
                        "Evaluating Current Model on Test Set. ==========".format(i, epoch)
                    )
                    improved_test_score = test(model, loss_func, test_x_left, test_x_right, test_y,
                                               i, epoch, args.batch_size, device)
                    test_acc, test_f1, test_precision, test_recall, test_kappa = improved_test_score
                    DARK_BLUE = "\033[1;34m"  # 加粗深蓝色
                    RESET = "\033[0m"

                    print(
                        DARK_BLUE +
                        "improved-val test acc:{:.5f}, test f1:{:.5f}, test precision:{:.5f}, "
                        "test recall:{:.5f}, test kappa:{:.5f}\n".format(
                            test_acc, test_f1, test_precision, test_recall, test_kappa
                        )
                        + RESET
                    )
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    break
            # one epoch end
            # print(best_test_score)
            if best_val_score is None:
                print(best_val_score)
            else:
                val_acc, val_f1, val_precision, val_recall, val_kappa = best_val_score
                print(
                    "val acc:{:.5f}, val f1:{:.5f}, val precision:{:.5f}, val recall:{:.5f}, val kappa:{:.5f}".format(
                        val_acc, val_f1, val_precision, val_recall, val_kappa
                    ))

        print(f"\n========== Fold {i} Training Finished. Evaluating on Test Set. ==========")
        model.load_state_dict(torch.load(best_model_path))

        final_test_score = test(model, loss_func, test_x_left, test_x_right, test_y,
                                i, 0, args.batch_size, device)

        # one fold end
        scores.append(final_test_score)
        print('Test set score:', scores)

    # all fold end, output the final result
    scores = np.array(scores)
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)

    print(f"\033[1;31mFinal DDI Result ({args.condition}):\n"
            f"Acc: {mean_scores[0]:.4f} ± {std_scores[0]:.4f}\n"
            f"F1 : {mean_scores[1]:.4f} ± {std_scores[1]:.4f}\n"
            f"Pre: {mean_scores[2]:.4f} ± {std_scores[2]:.4f}\n"
            f"Rec: {mean_scores[3]:.4f} ± {std_scores[3]:.4f}\n"
            f"Kap: {mean_scores[4]:.4f} ± {std_scores[4]:.4f}\033[0m")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    ap = argparse.ArgumentParser(description='KGSC-MAP-DDI 训练入口参数', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 训练与优化参数
    ap.add_argument('--batch_size', type=int, default=2 ** 15, help='每个批次的样本数量。')  # 批大小，越大显存占用越高
    ap.add_argument('--epoch', type=int, default=1000, help='最大训练轮数。')  # 与 patience 共同决定训练最长持续时间
    ap.add_argument('--patience', type=int, default=50, help='早停等待轮数。')  # 验证 F1 连续未提升到该轮数后停止
    ap.add_argument('--lr', type=float, default=5e-4, help='AdamW 学习率。')  # 控制优化器参数更新步长
    ap.add_argument('--weight_decay', type=float, default=2e-4, help='权重衰减系数。')  # 只作用于非 bias、非 BN、非 embedding 参数

    # 实验设置与基础模型结构
    ap.add_argument('--fold_num', type=int, default=5, help='交叉验证折数。')  # 控制 get_train_test 返回的训练/验证/测试折数
    ap.add_argument('--hidden_dim', type=int, default=256, help='模型隐藏层维度。')  # KG、分子和 APMP 分支共享的主要表示维度
    ap.add_argument('--num_layer', type=int, default=3, help='HGNN 主干网络层数。')  # 同时影响 KG 分支和分子 HGNN 分支深度
    ap.add_argument('--condition', type=str, choices=['S1', 'S2', 'S3'], default='S3', help='实验数据划分场景。')  # 同时影响数据划分和分子节点初始化方式
    ap.add_argument('--mode', type=str, choices=['only_kg', 'only_mol', 'concat'], default='concat', help='模型使用的分支模式。')  # 控制 KG 分支、分子分支或拼接分支是否参与预测

    # 数据路径与数据集参数
    ap.add_argument('--data_path', type=str, default='./data', help='数据根目录。')  # 与 kg_name、ddi_name 拼成实际数据目录
    ap.add_argument('--kg_name', type=str, default='DRKG', help='知识图谱数据集名称。')  # 用于定位 data_path/(kg_name+ddi_name)
    ap.add_argument('--ddi_name', type=str, choices=['Deng', 'Ryu'], default='Deng', help='DDI 数据集名称。')  # 影响数据目录和输出类别数

    # KG 药物子图抽取参数
    ap.add_argument('--use_subgraph_context', type=str2bool, default=True, help='是否启用 PageRank/RW 药物子图上下文模块。')  # 只在 KG 分支实际存在时生效
    ap.add_argument('--subgraph_size', type=int, default=32, help='每个药物子图保留的节点数量。')  # 控制 PPR/RW 子图抽取规模
    ap.add_argument('--subgraph_rw_steps', type=int, default=64, help='每个 KG 子图的最大随机游走步数。')  # 控制 RW 视图候选节点探索深度
    ap.add_argument('--subgraph_ppr_alpha', type=float, default=0.85, help='局部近似 PPR 的阻尼因子。')  # 控制 PPR 视图的传播衰减
    ap.add_argument('--subgraph_max_ego_nodes', type=int, default=4096, help='两跳 PPR ego 图最大候选节点数。')  # 限制子图抽取候选范围和内存
    ap.add_argument('--subgraph_ppr_max_pushes', type=int, default=1024, help='每个 PPR 子图最大 residual-push 次数。')  # 控制 PPR 近似计算成本
    ap.add_argument('--subgraph_ppr_epsilon', type=float, default=1e-4, help='近似 PPR residual 阈值。')  # 阈值越小 PPR 越精细但越慢
    ap.add_argument('--subgraph_excluded_edge_types', type=parse_int_list, default=[], help='子图抽取时排除的 KG 边类型 ID，逗号分隔。')  # 示例：--subgraph_excluded_edge_types 1,2,3

    # KG 子图编码与融合参数
    ap.add_argument('--subgraph_num_layer', type=int, default=1, help='Graph Transformer 子图编码器层数。')  # 控制药物 KG 子图编码器深度
    ap.add_argument('--subgraph_gt_num_heads', type=int, default=4, help='Graph Transformer 子图编码器注意力头数。')  # 需整除 hidden_dim
    ap.add_argument('--subgraph_gt_ffn_ratio', type=float, default=2.0, help='Graph Transformer FFN 隐藏层相对 hidden_dim 的倍率。')  # 控制子图 Transformer 前馈层容量
    ap.add_argument('--subgraph_gt_use_edge_bias', type=str2bool, default=True, help='Graph Transformer 是否使用 KG 边类型和 RW 边标记作为 attention bias。')  # 让注意力感知关系类型和随机游走轨迹边
    ap.add_argument('--subgraph_view_fusion', type=str, choices=['mean', 'attention', 'gate'], default='mean', help='PPR 视图与随机游走视图的融合方式。')  # mean/attention/gate 三种子图视图融合策略
    ap.add_argument('--kg_subgraph_mode', type=str, choices=['kg_only', 'subgraph_only', 'gate'], default='gate', help='全局 KG 与子图表示的融合模式。')  # kg_only 不用子图，subgraph_only 只用子图，gate 做门控融合
    ap.add_argument('--use_subgraph_node_id_embedding', type=str2bool, default=False, help='是否在子图分支使用独立可训练节点 ID embedding。')  # 增强节点身份信息但会增加参数量
    ap.add_argument('--subgraph_branch_scale_init', type=float, default=0.02, help='子图特征注入 KG embedding 的初始残差尺度。')  # 控制训练初期子图分支影响强度
    ap.add_argument('--subgraph_encode_chunk_size', type=int, default=128, help='每次编码的唯一药物子图数量。')  # 用于按块编码以控制显存
    ap.add_argument('--subgraph_gradient_checkpointing', type=str2bool, default=True, help='是否对子图 Graph Transformer 启用梯度检查点。')  # 以额外计算换取更低显存占用

    # APMP 路径采样与编码参数
    ap.add_argument('--use_apmp', type=str2bool, default=True, help='是否在分子分支启用 APMP 路径增强模块。')  # 仅在 mode 为 only_mol/concat 时生效
    ap.add_argument('--apmp_input_source', type=str, choices=['hgnn', 'raw'], default='raw', help='APMP 输入特征来源。')  # hgnn 使用 HGNN 后特征，raw 使用原始节点 embedding
    ap.add_argument('--apmp_role', type=str, choices=['enhance', 'delta'], default='delta', help='APMP 输出角色。')  # enhance 输出完整增强表示，delta 只输出路径辅助增量
    ap.add_argument('--apmp_num_paths', type=int, default=24, help='每个原子随机游走采样的候选路径数量。')  # 与 apmp_top_ratio 共同决定规则 top-k 数量
    ap.add_argument('--apmp_path_len', type=int, default=10, help='每条 APMP 关联路径的固定长度。')  # 与 use_path_mask_padding 共同决定路径补齐方式
    ap.add_argument('--apmp_top_ratio', type=float, default=0.4, help='规则路径选择时保留候选路径的比例。')  # top_k=int(apmp_num_paths*apmp_top_ratio)
    ap.add_argument('--apmp_num_layers', type=int, default=1, help='APMP 中 CPI TransformerEncoder 的层数。')  # 控制路径编码器深度
    ap.add_argument('--apmp_num_heads', type=int, default=4, help='APMP 中 CPI TransformerEncoder 的注意力头数。')  # 需要能整除 hidden_dim
    ap.add_argument('--apmp_chunk_size', type=int, default=2048, help='APMP forward 按原子分块处理的大小。')  # 用于控制 APMP 显存占用

    # APMP 分支融合与路径聚合参数
    ap.add_argument('--use_apmp_parallel', type=str2bool, default=True, help='是否将 APMP 表示作为并行分子分支。')  # 开启后与原始分子表示融合
    ap.add_argument('--apmp_parallel_fusion', type=str, choices=['gate', 'mean'], default='gate', help='原始分子分支与 APMP 并行分支的融合方式。')  # gate 使用门控，mean 使用均值
    ap.add_argument('--use_apmp_path_attention', type=str2bool, default=False, help='是否对 APMP 选中路径使用 attention 聚合。')  # 关闭时默认使用 mean pooling
    ap.add_argument('--use_path_mask_padding', type=str2bool, default=True, help='路径不足固定长度时是否使用 mask padding 补齐。')  # 关闭时允许回退/重复节点补齐

    # 官能团路径先验与解释参数
    ap.add_argument('--use_fg_path_score', type=str2bool, default=True, help='是否在 APMP 规则路径选择中加入 RDKit 官能团路径分数。')  # 只影响规则路径打分
    ap.add_argument('--fg_path_weight', type=float, default=1.25, help='官能团路径分数在规则路径选择中的权重。')  # 仅在 use_fg_path_score=True 且大于 0 时影响 top-k
    ap.add_argument('--save_fg_path_annotations', type=str2bool, default=True, help='是否保存选中路径的官能团匹配和解释信息。')  # 只保存解释文件，不直接改变预测

    args = ap.parse_args()

    args.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')



    terminal = sys.stdout
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        '{}_{}_{}_{}_{}_{}.txt'.format(
            args.ddi_name, args.mode, args.condition, args.hidden_dim, args.num_layer, args.run_timestamp
        )
    )
    sys.stdout = Logger(log_file, terminal)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    print(format_args_one_line(args))
    print(format_parameter_relations())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on', device)

    run(args)   # STEP 1
