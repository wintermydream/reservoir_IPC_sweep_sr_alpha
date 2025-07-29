"""
绘制不同degree下容量与alpha的关系图。

该脚本从debug_capacity_data/目录加载预先计算的容量数据，
并创建两种图表：
1. 1x4的子图布局，分别展示degree=1、3、5、7的容量与alpha的关系，每个子图中显示sr=0.6、1.0、1.4三条曲线。
2. 柱状图，展示不同degree的capacity叠加在一起，随alpha变化的趋势，每个alpha对应sr=0.6、1.0、1.4三个柱状图。
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_capacity_data(out_dir: Path, num_nodes: int, sr: float, omega: float = 1.0) -> tuple:
    """加载指定参数的容量数据。"""
    out_npz = out_dir / f"d1_d3_d5_d7_capacity_vs_alpha_N{num_nodes}_sr{sr:.3f}_om{omega:.3f}.npz"
    
    if not out_npz.exists():
        print(f"[Error] 找不到数据文件: {out_npz}")
        return None, None, None, None, None
    
    data = np.load(out_npz)
    alpha_values = data['alpha_values']
    d1_scores = data['d1_scores']
    d3_scores = data['d3_scores']
    d5_scores = data['d5_scores']
    d7_scores = data['d7_scores']
    
    return alpha_values, d1_scores, d3_scores, d5_scores, d7_scores

def plot_sr_comparison_vs_alpha(ax, alpha_values, d1_scores, d3_scores, d5_scores, d7_scores, sr, degree_idx, title):
    """在指定的轴上绘制特定sr值下的容量与alpha的关系图。"""
    markers = ['o-', 's--', 'd:', '^-']
    
    # 根据degree_idx选择对应的分数
    if degree_idx == 0:
        scores = d1_scores  # d1_scores
    elif degree_idx == 1:
        scores = d3_scores  # d3_scores
    elif degree_idx == 2:
        scores = d5_scores  # d5_scores
    else:
        scores = d7_scores  # d7_scores
            
    ax.plot(alpha_values, scores, markers[degree_idx], label=f'sr={sr:.1f}', markersize=4)
    
    ax.set_xlabel('Alpha (α)')
    ax.set_ylabel('Capacity')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

def plot_stacked_bar_capacity(alpha_values, sr_data_dict, selected_alpha_indices=None):
    """绘制叠加柱状图，展示不同degree的capacity随alpha变化的趋势。
    
    Args:
        alpha_values: alpha值数组
        sr_data_dict: 包含不同sr值的数据字典，格式为{sr: (d1_scores, d3_scores, d5_scores, d7_scores)}
        selected_alpha_indices: 选择的alpha索引，如果为None则使用所有alpha值
    """
    # 未指定则自动抽取 ~12 个 alpha 点
    if selected_alpha_indices is None:
        desired = 12
        step = max(1, int(np.ceil(len(alpha_values) / desired)))
        selected_alpha_indices = list(range(0, len(alpha_values), step))
        if (len(alpha_values) - 1) not in selected_alpha_indices:
            selected_alpha_indices.append(len(alpha_values) - 1)
    
    selected_alphas = [alpha_values[i] for i in selected_alpha_indices]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Nature-style: narrow bars with small inter-group gaps
    num_srs = len(sr_data_dict)
    group_gap = 0.05  # fraction of total group width left empty
    bar_width = (0.8 - group_gap) / num_srs  # leave gap at right of group
    hatches = ['////', '\\\\', '..']  # distinguish sr by hatch
    
    # 颜色设置
    degree_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 绘制柱状图
    for i, (sr, data) in enumerate(sr_data_dict.items()):
        d1_scores, d3_scores, d5_scores, d7_scores = data
        
        # 获取选定alpha值对应的scores
        selected_d1 = [d1_scores[idx] for idx in selected_alpha_indices]
        selected_d3 = [d3_scores[idx] for idx in selected_alpha_indices]
        selected_d5 = [d5_scores[idx] for idx in selected_alpha_indices]
        selected_d7 = [d7_scores[idx] for idx in selected_alpha_indices]
        
        # 计算 x 位置，给不同 sr 留出微小间隙
        x = np.arange(len(selected_alphas), dtype=float)
        bar_positions = x - 0.4 + i * bar_width + (i * group_gap / num_srs)
        
        # 绘制叠加柱状图
        bottom = np.zeros(len(selected_alphas))
        # 绘制 d1 层（仅给 sr 设置图例一次）
        bottom = np.zeros_like(selected_d1)
        ax.bar(bar_positions, selected_d1, bar_width, color=degree_colors[0],
               hatch=hatches[i % len(hatches)], edgecolor='black', linewidth=0.8,
               label=f'sr={sr}' )

        bottom += selected_d1
        
        # 绘制d3
        ax.bar(bar_positions, selected_d3, bar_width, bottom=bottom, 
               color=degree_colors[1], alpha=0.7, hatch=hatches[i % len(hatches)], edgecolor='black', linewidth=0.8)
        bottom += selected_d3
        
        # 绘制d5
        ax.bar(bar_positions, selected_d5, bar_width, bottom=bottom, 
               color=degree_colors[2], alpha=0.7, hatch=hatches[i % len(hatches)], edgecolor='black', linewidth=0.8)
        bottom += selected_d5
        
        # 绘制d7
        ax.bar(bar_positions, selected_d7, bar_width, bottom=bottom, 
               color=degree_colors[3], alpha=0.7, hatch=hatches[i % len(hatches)], edgecolor='black', linewidth=0.8)
        
        # 添加sr标签
        for j, pos in enumerate(bar_positions):
            ax.text(pos, 0.5, f'sr={sr}', ha='center', va='bottom', rotation=90, fontsize=8)
    
    # 设置x轴标签为alpha值
    ax.set_xticks(x)
    ax.set_xticklabels([f'{alpha:.3f}' for alpha in selected_alphas])
    
    # 添加图表标题和轴标签
    ax.set_title('Capacity vs Alpha for Different Degrees and SR Values', fontsize=16)
    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('Capacity', fontsize=12)
    
    # 仅显示 sr 图例（已在 d1 层添加）
    ax.legend(title='Spectral radius', fontsize=10)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()

    return fig

def main() -> None:
    # 参数设置
    num_nodes = 100
    omega = 1.0
    srs = [0.6, 1.0, 1.4]  # 三个sr值
    out_dir = Path('debug_capacity_data')
    
    # 创建1x4的子图布局
    fig1, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig1.suptitle('Capacity vs Alpha for Different Degrees', fontsize=16)
    
    degrees = [1, 3, 5, 7]
    degree_names = ['Degree 1', 'Degree 3', 'Degree 5', 'Degree 7']
    
    # 存储不同sr值的数据，用于绘制柱状图
    sr_data_dict = {}
    
    # 遍历所有degree
    for j, degree_idx in enumerate(range(4)):
        # 为每个sr加载数据并绘制
        for i, sr in enumerate(srs):
            # 加载数据
            alpha_values, d1_scores, d3_scores, d5_scores, d7_scores = load_capacity_data(
                out_dir, num_nodes, sr, omega
            )
            
            if alpha_values is None:
                print(f"[Warning] 缺少数据: sr={sr}")
                continue
            
            # 存储数据用于柱状图
            if sr not in sr_data_dict:
                sr_data_dict[sr] = (d1_scores, d3_scores, d5_scores, d7_scores)
            
            # 绘制子图
            title = f"{degree_names[j]}"
            plot_sr_comparison_vs_alpha(
                axes[j], alpha_values, d1_scores, d3_scores, d5_scores, d7_scores, sr, degree_idx, title
            )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为总标题留出空间
    
    # 保存图像
    output_file = 'capacity_vs_alpha_subplots.png'
    fig1.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"线图已保存到: {output_file}")
    
    # 绘制柱状图
    if alpha_values is not None and len(sr_data_dict) > 0:
        # 选择一些有代表性的alpha值点
        # 每隔一个 alpha 取一个点
        # 仅保留首尾 alpha 点
        # 每隔一个 alpha 取样，并确保首尾都在
        selected_indices = list(range(0, len(alpha_values), 2))
        if (len(alpha_values) - 1) not in selected_indices:
            selected_indices.append(len(alpha_values) - 1)
        fig2 = plot_stacked_bar_capacity(alpha_values, sr_data_dict, selected_indices)
    
    plt.show()

if __name__ == '__main__':
    main()