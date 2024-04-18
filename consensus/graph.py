# -*- coding: utf-8 -*-
# @Time    : 2024/3/25 10:46
# @Author  : Lumen
# @File    : graph.py
import json
import math
from collections import defaultdict
from enum import Enum
from typing import List, Tuple, Union, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.ion()


class EdgeType(Enum):
    ROOT_CONNECTION = 'root_connection'
    SUPPORT = 'support'
    OPPOSE = 'oppose'
    SUPPLY = 'supply'
    DISPUTE = 'dispute'


class NodeType(Enum):
    ROOT = 'root'
    IDEA = 'idea'
    ARGUMENTATION = 'argumentation'
    QUESTION = 'question'
    INFORMATION = 'information'


class DynamicDiscussionGraph:
    def __init__(self,
                 nodes: List[Tuple[int, NodeType, Union[datetime, str], int, str]] = None,
                 edges: List[Tuple[int, int, EdgeType]] = None,
                 granularity: str = 'D',
                 number: int = 1):
        """
        nodes format: Tuple[node_id, node_type, timestamp, user_weight, text], for instance:
                        [(1, NodeType.IDEA, '2022-01-01', 2, 'text1'),
                        (2, NodeType.ARGUMENTATION, '2022-01-01', 3, 'text2')]
        edges format: Tuple[src_node, dst_node, edge_type], for instance:
                        [(1, 2, EdgeType.SUPPORT), (1, 3, EdgeType.SUPPORT)]
        """
        self._graphs: List[nx.Graph] = []
        self._timestamp: List[datetime] = []
        self._root: Optional[int] = None
        self.unique_timestamps: List[datetime] = []

        if nodes is not None and edges is not None:
            self.add_graphs(nodes, edges, granularity, number)

    def add_graphs(self,
                   nodes: List[Tuple[int, NodeType, Union[datetime, str], int, str]],
                   edges: List[Tuple[int, int, EdgeType]],
                   granularity: str = 'D',
                   number: int = 1,
                   draw: bool = False):

        if isinstance(nodes[0][2], str):
            nodes = [(node_id, node_type, datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'), weight, text) for
                     node_id, node_type, timestamp, weight, text in nodes]

        unique_timestamps = set()
        roots = set()
        for node_id, node_type, timestamp, weight, text in nodes:
            unique_timestamps.add(timestamp)
            if node_type == NodeType.ROOT:
                roots.add(node_id)

        if len(roots) >= 2:
            raise ValueError("Two or more root nodes")
        else:
            self._root = roots.pop()

        self.unique_timestamps = [datetime.strftime(timestamp, '%Y-%m-%d %H:%M:%S') for timestamp in sorted(unique_timestamps)]
        time_bins = self.time_binning(self.unique_timestamps, granularity, number, draw)
        for start_time, end_time in time_bins:
            nodes_at_timestamp = [(node_id, node_type, ts, weight, text)
                                  for node_id, node_type, ts, weight, text in nodes
                                  if ts < end_time]
            valid_nodes_at_timestamp = set(node_id for node_id, _, _, _, _ in nodes_at_timestamp)
            edges_at_timestamp = [(src, dst, edge_type)
                                  for src, dst, edge_type in edges
                                  if src in valid_nodes_at_timestamp and dst in valid_nodes_at_timestamp]

            if nodes_at_timestamp and edges_at_timestamp:
                self._add_graph(start_time, nodes_at_timestamp, edges_at_timestamp)

    def _add_graph(self,
                   timestamp: datetime,
                   nodes: List[Tuple[int, NodeType, Union[datetime, str], int, str]],
                   edges: List[Tuple[int, int, EdgeType]]):
        g = nx.Graph()
        nodes_with_attrs = [(node, {'type': node_type, 'timestamp': timestamp, 'weight': weight, 'text': text}) for
                            node, node_type, timestamp, weight, text in nodes]
        g.add_nodes_from(nodes_with_attrs)

        edges_with_attrs = [(src, dst, {'type': edge_type}) for src, dst, edge_type in edges]
        g.add_edges_from(edges_with_attrs)
        self._graphs.append(g)
        self._timestamp.append(timestamp)

    @staticmethod
    def _calculate_consensus(graph: nx.Graph, root: int) -> Dict[int, float]:
        consensus = {node: 0 for node in graph.nodes}
        visited = set()

        def dfs(node: int, parent: int):
            visited.add(node)
            current_weight = graph.nodes[node]['weight']
            consensus_value = current_weight

            for neighbor, edge_attr in graph[node].items():
                if neighbor in visited or neighbor == parent:
                    continue
                neighbor_consensus = dfs(neighbor, node)

                if (edge_attr['type'] == EdgeType.SUPPORT
                        or edge_attr['type'] == EdgeType.SUPPLY
                        or edge_attr['type'] == EdgeType.ROOT_CONNECTION):
                    consensus_value += neighbor_consensus
                elif edge_attr['type'] == EdgeType.OPPOSE:
                    consensus_value -= neighbor_consensus

            consensus[node] = consensus_value
            return consensus_value

        dfs(root, -1)

        return consensus

    def get_consensus(self, node_type: NodeType = NodeType.IDEA) -> List[Dict[int, float]]:
        self._check_graphs()

        graphs_consensus = []
        for g in self._graphs:
            all_nodes_consensus = self._calculate_consensus(g, self._root)
            target_nodes = [i for i in g.nodes if g.nodes[i].get('type') == node_type]
            target_nodes_consensus = {i: all_nodes_consensus[i] for i in target_nodes}
            graphs_consensus.append(target_nodes_consensus)

        return graphs_consensus

    def get_skewness(self, node_type: NodeType = NodeType.IDEA, draw: bool = True) -> List[float]:
        graphs_consensus = self.get_consensus(node_type)
        graphs_skewness = [self._calculate_skewness(list(consensus.values())) for consensus in graphs_consensus]
        if draw:
            plt.plot(graphs_skewness, marker="o")
            plt.xlabel("时间点")
            plt.ylabel("偏度")
            plt.show()

        return graphs_skewness

    @staticmethod
    def _calculate_skewness(arr: Union[List, np.ndarray]) -> float:
        if isinstance(arr, List):
            arr = np.array(arr)

        n = len(arr)
        if n < 3:
            return 0

        mean = np.mean(arr)
        deviations = arr - mean
        cubed_deviations = deviations ** 3
        squared_deviations = deviations ** 2
        sum_cubed_deviations = np.sum(cubed_deviations)
        sum_squared_deviations = np.sum(squared_deviations)
        adj_factor = np.sqrt(n * (n - 1)) / (n - 2)

        skewness = (adj_factor * (1.0 / n * sum_cubed_deviations)) / ((1.0 / n * sum_squared_deviations) ** 1.5)

        return skewness

    def _check_graphs(self):
        if not self._graphs:
            raise ValueError("empty graphs, please add nodes and edges using method 'add_graphs()'")

    def load_graphs_from_json(self, json_path: str,
                              granularity: str = 'D',
                              number: int = 1,
                              draw: bool = False) -> 'DynamicDiscussionGraph':
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file).get('data')

        # 递归遍历节点的函数
        def dfs(node, parent_id=None):
            node_id = node['data']['id']
            text = node['data']['text']
            replytime = node['data']['replytime']
            resource = node['data'].get('resource', [])
            user_weight = node['data']['user_weight']

            if parent_id is None:  # 根节点没有 parent_id
                node_type = NodeType.ROOT
            else:
                if len(resource) == 1 and '主意' in resource:
                    node_type = NodeType.IDEA
                elif len(resource) == 2 and '论证' in resource:
                    node_type = NodeType.ARGUMENTATION
                elif len(resource) == 2 and '疑问' in resource:
                    node_type = NodeType.QUESTION
                elif len(resource) == 2 and '资料' in resource:
                    node_type = NodeType.INFORMATION
                else:
                    raise ValueError("标注有错误，请检查")

            nodes.append((node_id, node_type, replytime, user_weight, text))

            if parent_id is not None:
                if len(resource) == 2 and '支持' in resource:
                    edge_type = EdgeType.SUPPORT
                elif len(resource) == 2 and '反对' in resource:
                    edge_type = EdgeType.OPPOSE
                elif len(resource) == 2 and '补充' in resource:
                    edge_type = EdgeType.SUPPLY
                elif len(resource) == 2 and '质疑' in resource:
                    edge_type = EdgeType.DISPUTE
                elif len(resource) == 1 and '主意' in resource:
                    edge_type = EdgeType.ROOT_CONNECTION
                else:
                    raise ValueError("标注有错误，请检查")

                edges.append((parent_id, node_id, edge_type))

            # 递归遍历子节点
            for child in node.get('children', []):
                dfs(child, parent_id=node_id)

        nodes = []
        edges = []

        # 从根节点开始递归遍历
        dfs(data['root'])

        # 使用 add_graphs 添加图形
        self.add_graphs(nodes, edges, granularity, number, draw)

        return self

    def _draw_graph(self, index: int):
        G = self._graphs[index]

        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')

        node_types = nx.get_node_attributes(G, 'type')

        default_node_opts = {'node_size': 400, 'alpha': 0.9}

        type_node_list = defaultdict(list)
        for node, node_type in node_types.items():
            type_node_list[node_type].append(node)

        for node_type, nodes in type_node_list.items():
            if node_type == NodeType.IDEA:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='skyblue', **default_node_opts)
            elif node_type == NodeType.ARGUMENTATION:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='green', **default_node_opts)
            elif node_type == NodeType.QUESTION:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='red', **default_node_opts)
            elif node_type == NodeType.INFORMATION:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='black', **default_node_opts)
            elif node_type == NodeType.ROOT:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='grey', **default_node_opts)

        edge_types = nx.get_edge_attributes(G, 'type')

        for (u, v), edge_type in edge_types.items():
            if edge_type == EdgeType.OPPOSE:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='red', style='dotted')
            elif edge_type == EdgeType.SUPPORT:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='blue', style='dashed')
            elif edge_type == EdgeType.SUPPLY:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='black', style='dashed')
            elif edge_type == EdgeType.DISPUTE:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='yellow', style='dashed')
            elif edge_type == EdgeType.ROOT_CONNECTION:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='grey', style='dashed')

        nx.draw_networkx_labels(G, pos)
        edge_labels = dict(
            ((u, v), d['type'].value) for u, v, d in G.edges(data=True) if d['type'] != EdgeType.ROOT_CONNECTION)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    def draw(self, index: Optional[int] = None):
        self._check_graphs()
        if index is not None:
            self._draw_graph(index)
            plt.show()
        else:
            total_graphs = len(self._graphs)
            if total_graphs == 0:
                return

            cols = int(math.ceil(math.sqrt(total_graphs)))
            rows = int(math.ceil(total_graphs / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            if total_graphs == 1:
                axes = [axes]
            axes = axes.flatten()

            for i, g in enumerate(self._graphs):
                plt.sca(axes[i])  # 设置当前激活的Axes对象
                self._draw_graph(i)
                plt.title(self._timestamp[i])

            # 隐藏空余的子图位置
            for j in range(len(self._graphs), len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.show()

    @staticmethod
    def time_binning(time_list: List[Union[str, datetime]], granularity: str = 'D', number: int = 1, draw: bool = False) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        对时间列表进行分箱。
        :param time_list: 时间字符串的列表。
        :param granularity: 分箱的粒度，可以是'H', 'D', 'W', 'M', 或 'Y'。
        :param number: 分箱的数量。
        :param draw: 是否画图。
        :return: 每个分箱的起始和结束时间（左闭右开）。
        """
        # 将时间字符串转换为datetime objects
        times = pd.to_datetime(time_list)

        # 根据粒度设置时间偏移
        if granularity == 'H':
            offset = pd.DateOffset(hours=number)
        elif granularity == 'D':
            offset = pd.DateOffset(days=number)
        elif granularity == 'W':
            offset = pd.DateOffset(weeks=number)
        elif granularity == 'M':
            offset = pd.DateOffset(months=number)
        elif granularity == 'Y':
            offset = pd.DateOffset(years=number)
        else:
            raise ValueError("Granularity must be 'H', 'D', 'W', 'M', or 'Y'.")

        # 创建分箱的起始时间列表
        start_time = times.min()
        end_time = times.max()
        bin_edges = [start_time]
        while bin_edges[-1] <= end_time:
            bin_edges.append(bin_edges[-1] + offset)

        # 计算每个时间点落在哪个分箱中
        time_bins = pd.cut(times, bins=bin_edges, right=False)
        bin_counts = time_bins.value_counts().sort_index()

        if draw:
            # 绘制折线图和柱状图
            fig, ax = plt.subplots()

            ax.set_xlabel('时间')
            ax.set_ylabel('发帖数量', color='tab:red')
            # ax.bar(bin_counts.index.categories.left, bin_counts, color='tab:red', alpha=0.6, label='Count')
            # ax.tick_params(axis='y', labelcolor='tab:red')

            ax.plot(bin_counts.index.categories.left, bin_counts, color='tab:blue', marker="o", label='Count (Line)')
            ax.tick_params(axis='y', labelcolor='tab:blue')

            fig.tight_layout()  # 调整整体空白
            fig.autofmt_xdate()
            plt.xticks(rotation=45)
            plt.show()

        # 返回分箱的起始和结束时间
        return [(bin.left, bin.right) for bin in bin_counts.index]

    @property
    def graphs(self) -> List[nx.Graph]:
        return self._graphs

    def __getitem__(self, item):
        return self._graphs[item]


if __name__ == '__main__':
    ddg = DynamicDiscussionGraph()
    # nodes = [(0, NodeType.ROOT, '2017-09-06 00:00:00', 0, '求助'),
    #          (1, NodeType.IDEA, '2017-09-06 00:00:00', 3, '多跑几家医院多问几个医生'),
    #          (2, NodeType.ARGUMENTATION, '2017-09-07 00:00:00', 2, '好的！跑了好几家了，说是先化疗'),
    #          (3, NodeType.ARGUMENTATION, '2017-09-12 00:00:00', 4, '多上网看看'),
    #          (4, NodeType.IDEA, '2017-09-06 00:00:00', 5, '搞清楚为什么直接9291，不考虑一代TKI'),
    #          (5, NodeType.ARGUMENTATION, '2017-09-07 00:00:00', 3,
    #           '要谨慎，如果从9291起，一旦耐药，目前来讲可选择的靶向药不是很多，正常的顺序都是从一代靶向药吃起'),
    #          (6, NodeType.ARGUMENTATION, '2017-09-08 00:00:00', 2, '我准备先吃靶向药，穿插进行化疗，进行复敏'),
    #          (7, NodeType.ARGUMENTATION, '2017-09-09 00:00:00', 4, '有基因突变患者，靶向为一线方案为佳，这得到多数认可'),
    #          (8, NodeType.ARGUMENTATION, '2017-09-11 00:00:00', 2,
    #           '19.20突变，是黄金突变，首先应该用易瑞沙或者凯美纳，9291为后者'),
    #          (9, NodeType.ARGUMENTATION, '2017-09-12 00:00:00', 3, '有突变化疗的效果可能不好，还是从一代的易特开始吧'),
    #          (10, NodeType.ARGUMENTATION, '2017-09-09 00:00:00', 2, '是啊，毕竟这个副作用很小而且效果明显'),
    #          (11, NodeType.IDEA, '2017-09-16 00:00:00', 3, '化疗，进行复敏，加油'),
    #          (12, NodeType.IDEA, '2017-09-08 00:00:00', 2, '谢谢，我也打算从易瑞沙开始吃'),
    #          (13, NodeType.ARGUMENTATION, '2017-09-14 00:00:00', 4, '现在吃易瑞沙呢'),
    #          (14, NodeType.ARGUMENTATION, '2017-09-16 00:00:00', 3, '首先应该用易瑞沙')]
    # edges = [(0, 1, EdgeType.ROOT_CONNECTION),
    #          (1, 2, EdgeType.SUPPORT),
    #          (1, 3, EdgeType.SUPPLY),
    #          (0, 4, EdgeType.ROOT_CONNECTION),
    #          (4, 5, EdgeType.SUPPORT),
    #          (4, 6, EdgeType.SUPPLY),
    #          (4, 7, EdgeType.SUPPORT),
    #          (4, 8, EdgeType.SUPPORT),
    #          (4, 9, EdgeType.SUPPORT),
    #          (7, 10, EdgeType.SUPPORT),
    #          (0, 11, EdgeType.ROOT_CONNECTION),
    #          (0, 12, EdgeType.ROOT_CONNECTION),
    #          (12, 13, EdgeType.SUPPORT),
    #          (13, 14, EdgeType.SUPPORT)]
    # nodes = [(1, NodeType.IDEA, '2016-03-27 00:00:00', 3, '直接上9291把'),
    #          (2, NodeType.ARGUMENTATION, '2016-03-30 00:00:00', 2, '9291肯定是放到最后一个考虑关键是有误脑转要搞清楚'),
    #          (3, NodeType.ARGUMENTATION, '2016-04-02 00:00:00', 3, '4002/9291/3759什么的，现在肯定不考虑'),
    #          (4, NodeType.ARGUMENTATION, '2016-04-03 00:00:00', 4,
    #           '2992,4002,9291都有790靶点，只是2992对790是弱效，4002,9291对790打击力大些，但是4002不入脑，9291入脑。右脑转，最好用9291'),
    #          (5, NodeType.ARGUMENTATION, '2016-04-03 00:00:00', 3, '我家一声也建议直接上9291'),
    #          (6, NodeType.ARGUMENTATION, '2016-04-04 00:00:00', 2,
    #           '群里人一般都建议暂不把9291用上，好药留后面，是怕耐药了没有后续药，所以拖时间')]
    # edges = [(1, 2, EdgeType.OPPOSE), (1, 3, EdgeType.OPPOSE), (1, 4, EdgeType.SUPPORT),
    #          (1, 5, EdgeType.SUPPORT), (5, 6, EdgeType.OPPOSE)]
    # ddg.add_graphs(nodes, edges)
    ddg.load_graphs_from_json('../script/data/tmjy/389727634454528/labeled.json', 'D', 1)
    ddg.draw()
    print(ddg.get_consensus())
    print(ddg.get_skewness())
    time_bins = ddg.time_binning(ddg.unique_timestamps, granularity='D', number=1, draw=True)
    print(time_bins[0][0])
    consensus = pd.DataFrame(ddg.get_consensus())
    consensus['t1'] = [t1 for t1, t2 in time_bins]
    consensus['t2'] = [t2 for t1, t2 in time_bins]
    consensus = consensus.set_index(['t1', 't2'])
    print(consensus)

