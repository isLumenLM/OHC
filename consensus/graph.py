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

import colorama
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
                 nodes: List[Tuple[str, NodeType, Union[datetime, str], str, int, str]] = None,
                 edges: List[Tuple[str, str, EdgeType]] = None,
                 granularity: str = 'D',
                 number: int = 1):
        """
        nodes format: Tuple[node_id, node_type, timestamp, user, user_weight, text], for instance:
                        [('1', NodeType.IDEA, '2022-01-01', '123', 2, 'text1'),
                        ('2', NodeType.ARGUMENTATION, '2022-01-01', '234', 3, 'text2')]
        edges format: Tuple[src_node, dst_node, edge_type], for instance:
                        [('1', '2', EdgeType.SUPPORT), (1, 3, EdgeType.SUPPORT)]
        """
        self._graphs: List[nx.Graph] = []
        self._timestamp: List[datetime] = []
        self._root: Optional[int] = None
        self._unique_timestamps: List[datetime] = []
        self._time_bins: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        if nodes is not None and edges is not None:
            self.add_graphs(nodes, edges, granularity, number)

    def add_graphs(self,
                   nodes: List[Tuple[str, NodeType, Union[datetime, str], str, int, str]],
                   edges: List[Tuple[str, str, EdgeType]],
                   granularity: str = 'D',
                   number: int = 1,
                   draw: bool = False):

        if isinstance(nodes[0][2], str):
            nodes = [(node_id, node_type, datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'), user, weight, text) for
                     node_id, node_type, timestamp, user, weight, text in nodes]

        # 一个id只能有一个节点
        if len(set([node_id for node_id, _, _, _, _, _ in nodes])) != len(nodes):
            raise ValueError('duplicate node_id')

        # 一个节点只能有一条边
        if len(nodes) != len(edges) + 1:
            raise ValueError('There are node connected by multiple edges.')

        unique_timestamps = set()
        roots = set()
        for node_id, node_type, timestamp, user, weight, text in nodes:
            unique_timestamps.add(timestamp)
            if node_type == NodeType.ROOT:
                roots.add(node_id)

        if len(roots) >= 2:
            raise ValueError("Two or more root nodes")
        else:
            self._root = roots.pop()

        self._unique_timestamps = [datetime.strftime(timestamp, '%Y-%m-%d %H:%M:%S') for timestamp in sorted(unique_timestamps)]
        self._time_bins = self.time_binning(self.unique_timestamps, granularity, number, draw)
        for start_time, end_time in self._time_bins:
            nodes_at_timestamp = [(node_id, node_type, ts, user, weight, text)
                                  for node_id, node_type, ts, user, weight, text in nodes
                                  if ts < end_time]
            valid_nodes_at_timestamp = set(node_id for node_id, _, _, _, _, _ in nodes_at_timestamp)
            edges_at_timestamp = [(src, dst, edge_type)
                                  for src, dst, edge_type in edges
                                  if src in valid_nodes_at_timestamp and dst in valid_nodes_at_timestamp]

            if nodes_at_timestamp and edges_at_timestamp:
                self._add_graph(start_time, nodes_at_timestamp, edges_at_timestamp)

    def _add_graph(self,
                   timestamp: datetime,
                   nodes: List[Tuple[str, NodeType, Union[datetime, str], str, int, str]],
                   edges: List[Tuple[str, str, EdgeType]]):
        g = nx.Graph()
        nodes_with_attrs = [(node, {'type': node_type, 'timestamp': timestamp,
                                    'user': user, 'weight': weight, 'text': text}) for
                            node, node_type, timestamp, user, weight, text in nodes]
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

                if edge_attr['type'] == EdgeType.SUPPORT or edge_attr['type'] == EdgeType.ROOT_CONNECTION:
                    consensus_value += neighbor_consensus
                elif edge_attr['type'] == EdgeType.OPPOSE:
                    consensus_value -= neighbor_consensus
                elif edge_attr['type'] == EdgeType.SUPPLY:
                    consensus_value += 0.5 * neighbor_consensus

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
            plt.show(block=True)

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
            node_id: str = str(node['data']['id'])
            text: str = node['data']['source_text']
            replytime: str = node['data']['replytime']
            resource: List = node['data'].get('resource', [])
            user: str = str(node['data']['user'])
            user_weight: int = node['data']['user_weight']

            allowed_combinations = [
                frozenset(["主意"]),
                frozenset(["支持", "论证"]),
                frozenset(["反对", "论证"]),
                frozenset(["补充", "论证"]),
                frozenset(["补充", "资料"]),
                frozenset(["质疑", "疑问"]),
            ]
            resource_set = frozenset(resource)
            if (resource_set and resource_set not in allowed_combinations) or (not resource_set and parent_id is not None):
                raise ValueError("标注有错误，请检查")

            # 如果是分裂的节点 再分配一个id
            if node_id in unique_nodes:
                node_id = node_id + '_1'
            unique_nodes.add(node_id)

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

            nodes.append((node_id, node_type, replytime, user, user_weight, text))

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

        unique_nodes = set()
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
            plt.show(block=True)
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
            plt.show(block=True)

    @staticmethod
    def time_binning(time_list: List[Union[str, datetime]],
                     granularity: str = 'D',
                     number: int = 1,
                     draw: bool = False,
                     verbose: bool = False) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        对时间列表进行分箱。
        :param time_list: 时间字符串的列表。
        :param granularity: 分箱的粒度，可以是'H', 'D', 'W', 'M', 或 'Y'。
        :param number: 分箱的数量。
        :param draw: 是否画图。
        :param verbose: 是否输出时间单元数
        :return: 每个分箱的起始和结束时间（左闭右开）。
        """
        # 将时间字符串转换为datetime objects
        times = pd.to_datetime(time_list)

        # 根据粒度设置时间偏移
        if granularity.upper() == 'H':
            offset = pd.DateOffset(hours=number)
        elif granularity.upper() == 'D':
            offset = pd.DateOffset(days=number)
        elif granularity.upper() == 'W':
            offset = pd.DateOffset(weeks=number)
        elif granularity.upper() == 'M':
            offset = pd.DateOffset(months=number)
        elif granularity.upper() == 'Y':
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

        if verbose:
            print(colorama.Fore.RED + f"时间单元数量：{len(bin_counts)}")

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
            plt.show(block=True)

        # 返回分箱的起始和结束时间
        return [(bin.left, bin.right) for bin in bin_counts.index]

    @property
    def graphs(self) -> List[nx.Graph]:
        return self._graphs

    @property
    def unique_timestamps(self) -> List[datetime]:
        return self._unique_timestamps

    @property
    def time_bins(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        return self._time_bins

    def number_of_nodes(self, idx: Optional[int] = None):
        if idx is None:
            idx = -1
        return self._graphs[idx].number_of_nodes()

    def to_sentence_pair(self) -> pd.DataFrame:
        self._check_graphs()

        graph = self._graphs[-1]

        sentence_pairs = []
        for edge in graph.edges(data=True):
            src, dst, data = edge
            src_data = graph.nodes[src]
            dst_data = graph.nodes[dst]
            edge_type = data['type'].value

            if src_data['type'] == NodeType.ROOT or dst_data['type'] == NodeType.ROOT:
                continue

            sentence_pairs.append({
                'text1': src_data['text'],
                'text2': dst_data['text'],
                'text1_type': src_data['type'].value,
                'text2_type': dst_data['type'].value,
                'edge_type': edge_type
            })

        return pd.DataFrame(sentence_pairs)

    def to_graph_info(self) -> pd.DataFrame:
        self._check_graphs()

        prev_users = set()
        prev_nodes = set()

        data = {
            '用户数量': [],
            '发言数量': [],
            '主意数量': []
        }

        for idx, graph in enumerate(self._graphs):

            current_users = {graph.nodes[node]['user'] for node in graph if graph.nodes[node]['type'] != NodeType.ROOT}
            current_nodes = {node for node in graph if graph.nodes[node]['type'] != NodeType.ROOT}

            new_users = current_users - prev_users
            new_nodes = current_nodes - prev_nodes

            prev_users = current_users
            prev_nodes = current_nodes

            new_idea_count = sum(1 for node in new_nodes if graph.nodes[node]['type'] == NodeType.IDEA)

            data["用户数量"].append(len(new_users))
            data["发言数量"].append(len(new_nodes))
            data["主意数量"].append(new_idea_count)

        time_intervals = [f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                          for start_time, end_time in self._time_bins]
        df = pd.DataFrame(data, index=time_intervals)
        df['开始时间'] = [t1 for t1, t2 in self._time_bins]
        df['结束时间'] = [t2 for t1, t2 in self._time_bins]
        df = df.set_index(['开始时间', '结束时间'])

        return df

    def max_nodes_in_a_level(self, index: int = -1) -> int:
        self._check_graphs()

        graph = self._graphs[index]

        # 检查是否设置了根节点
        if self._root is None:
            raise ValueError("Root node is not set.")

        levels = {}  # {node: level}
        queue = [(self._root, 0)]  # (node, level)
        while queue:
            node, level = queue.pop(0)
            if node not in levels:
                levels[node] = level
                neighbors = list(graph[node])
                for next_node in neighbors:
                    queue.append((next_node, level + 1))

        # 计算每个层级的节点数量
        level_counts = {}
        for level in levels.values():
            if level in level_counts:
                level_counts[level] += 1
            else:
                level_counts[level] = 1

        # 找到节点数量最多的层级
        max_nodes_count = max(level_counts.values())

        return max_nodes_count

    def max_depth(self, index: int = -1):
        self._check_graphs()

        if self._root is None:
            raise ValueError("Root node is not set.")

        def dfs(node, current_depth):
            visited.add(node)
            max_depth = current_depth
            for neighbor in graph[node]:
                if neighbor not in visited:
                    depth = dfs(neighbor, current_depth + 1)
                    max_depth = max(max_depth, depth)
            return max_depth

        graph = self._graphs[index]

        visited = set()
        start_depth = 0
        ans = dfs(self._root, start_depth)

        return ans

    def __getitem__(self, item):
        return self._graphs[item]

