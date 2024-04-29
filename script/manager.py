# -*- coding: utf-8 -*-
# @Time    : 2024/4/17 15:08
# @Author  : Lumen
# @File    : manager.py
import abc
import json
import os
import pickle
import sys
import time
import webbrowser
from pprint import pprint
from typing import Dict, List, Optional

import colorama
import numpy as np
import pandas as pd
import requests

sys.path.append('..')
from consensus.graph import DynamicDiscussionGraph, NodeType

colorama.init(autoreset=True)


class PostManager(metaclass=abc.ABCMeta):

    def __init__(self):
        self.login_session = None
        self.pid = None
        self.posts = None
        self.idx = None

        self.init_json = None
        self.labeled_json = None

        self.work_path = 'data/'

    def _check(self):
        if self.pid is None:
            print("请先切换到一个帖子。")
            return False
        return True

    def switch_post(self):

        while True:
            try:
                print(colorama.Fore.RED + "可以输入自定义序号+帖子ID，也可以只输入帖子id")

                inputs = input("请输入新的帖子ID: ")
                if '+' in inputs:
                    self.pid = int(inputs.split('+')[1])
                    self.idx = int(inputs.split('+')[0])
                else:
                    self.pid = int(inputs)
                    self.idx = int(input("请输入自定义序号："))
                print(f"已切换到帖子ID: {self.pid}, 自定义序号：{self.idx}")
                break
            except ValueError:
                print("输入错误，请输入有效的帖子ID或自定义序号。")

        print("正在爬取帖子...")
        self._crawl_post()

        if not os.path.exists(os.path.join(self.work_path, f'{self.idx}+{self.pid}')):
            os.makedirs(os.path.join(self.work_path, f'{self.idx}+{self.pid}'))

        pd.DataFrame(self.posts).to_csv(
            os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+source.csv'),
            index=False,
            encoding='utf-8_sig',
            sep='\t'
        )

        self._posts_to_json(self.posts)
        self._save_init_json()

        print(f"帖子爬取完毕, 数据保存到{os.path.join(self.work_path, f'{self.idx}+{self.pid}')}")

    @staticmethod
    def open_flow():
        webbrowser.open('https://www.baidufe.com/fehelper/naotu/index.html')

    def view_post_counts(self, labeled: bool = False):
        if not self._check():
            return

        if labeled:
            ddg = DynamicDiscussionGraph()
            ddg.load_graphs_from_json(
                json_path=os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+labeled.json'),
                granularity='D',
                number=1,
                draw=False
            )
            reply_times = ddg.unique_timestamps
            nodes_num = ddg.number_of_nodes()
        else:
            reply_times = [post.get('replytime') for post in self.posts]
            nodes_num = len(self.posts)

        print(colorama.Fore.RED + f"发帖时间：{reply_times[0]}")
        print(colorama.Fore.RED + f"最后回复时间：{reply_times[-1]}")
        print(colorama.Fore.RED + f"节点数：{nodes_num - 1}")

        print("可选时间粒度：")
        print("1.小时：H")
        print("2.天：D")
        print("3.周：W")
        print("4.月：M")
        print("5.年：Y")
        print("例如，以25天进行分箱，时间粒度输入D，周期输入25")
        granularity = input("请输入时间粒度（'H', 'D', 'W', 'M', 'Y'）：")
        number = int(input("请输入周期（正整数）："))
        time_bins = DynamicDiscussionGraph.time_binning(reply_times, granularity=granularity,
                                                        number=number, draw=True, verbose=True)
        print(colorama.Fore.RED + "分箱时间区间：")
        pprint(time_bins)

    def calculate_skewness_and_consensus(self):
        if not self._check():
            return

        print("可选时间粒度：")
        print("1.小时：H")
        print("2.天：D")
        print("3.周：W")
        print("4.月：M")
        print("5.年：Y")
        print("例如，以25天进行分箱，时间粒度输入D，周期输入25")
        print("请确定已经标注好数据")
        print(f"并将已经标注的数据放入{os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+labeled.json')}")
        granularity = input("请输入时间粒度（'H', 'D', 'W', 'M', 'Y'）：").upper()
        number = int(input("请输入周期（正整数）："))
        ddg = DynamicDiscussionGraph()
        ddg.load_graphs_from_json(
            json_path=os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+labeled.json'),
            granularity=granularity,
            number=number,
            draw=True
        )
        # ddg.draw()

        skewness = ddg.get_skewness(draw=True)
        pd.DataFrame(skewness).to_csv(
            os.path.join(self.work_path,
                         f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+skewness+{number}{granularity}.csv'),
            index=False,
            encoding='utf-8_sig',
            header=['偏度值']
        )

        time_bins = ddg.time_binning(ddg.unique_timestamps, granularity=granularity, number=number, draw=False)
        consensus = pd.DataFrame(ddg.get_consensus()).fillna(0)
        consensus['开始时间'] = [t1 for t1, t2 in time_bins]
        consensus['结束时间'] = [t2 for t1, t2 in time_bins]
        consensus = consensus.set_index(['开始时间', '结束时间'])
        consensus.to_csv(
            os.path.join(self.work_path,
                         f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+consensus+{number}{granularity}.csv'),
            encoding='utf-8_sig'
        )

        sentence_pair = ddg.to_sentence_pair()
        sentence_pair.to_csv(
            os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+pairs.csv'),
            index=False,
            encoding='utf-8_sig'
        )

        graph_info = ddg.to_graph_info()
        graph_info.to_csv(
            os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+count+{number}{granularity}.csv'),
            encoding='utf-8_sig'
        )

        ls_graph = ddg.graphs[-1]
        features_dict = {
            '偏度值总体平均值': np.average(skewness),
            '发言类型的数量': len({ls_graph.nodes[node]['type'] for node in ls_graph if ls_graph.nodes[node]['type'] != NodeType.ROOT}),
            '主意个数': len({node for node in ls_graph if ls_graph.nodes[node]['type'] == NodeType.IDEA}),
            '线程广度': ddg.max_nodes_in_a_level(),
            '线程深度': ddg.max_depth(),
            '载体丰富度': self.posts[0].get('imgs') + self.posts[0].get('external_links_count'),
            '第一条评论时间': self.posts[0].get('replytime'),
            '最后一条评论时间': self.posts[-1].get('replytime'),
            '发帖者发表主题数': self.posts[0].get('posts'),
            '发帖者回帖数': self.posts[0].get('replys'),
            '有效回复数': ddg.number_of_nodes() - 1,
            '发帖者等级': self._convert_user_weight(self.posts[0].get('group'))
        }
        pd.DataFrame([features_dict]).to_csv(
            os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+features+{number}{granularity}.csv'),
            encoding='utf-8_sig',
            index=False
        )

    def prompt_user(self):
        user, password = self._get_config()
        self.login_session = self._load_session(user)
        if self.login_session is None:
            self.login_session = self._login(user, password)
            self._save_session(user)
        else:
            print(colorama.Fore.RED + f"从本地登录用户：{user}")

        while True:
            try:
                if self.pid is None:
                    self.switch_post()

                print("=" * 30)
                print(colorama.Fore.RED + f"目前正在处理的帖子id: {self.idx}+{self.pid}")
                print("请选择功能：")
                print("1. 切换帖子")
                print("2. 重新登录")
                print("3. 打开脑图")
                print("4. 查看时间分箱后的帖子数量（标注前）")
                print("5. 查看时间分箱后的帖子数量（标注后）")
                print("6. 计算偏度和共识度（标注后）")
                print("7. 退出")
                print("=" * 30)

                choice = input("请输入你的选择（1-7）: ")

                if choice == '1':
                    self.switch_post()
                elif choice == '2':
                    user, password = self._get_config()
                    self.login_session = self._login(user, password)
                    self._save_session(user)
                elif choice == '3':
                    self.open_flow()
                elif choice == '4':
                    self.view_post_counts()
                elif choice == '5':
                    self.view_post_counts(labeled=True)
                elif choice == '6':
                    self.calculate_skewness_and_consensus()
                elif choice == '7':
                    print("退出程序.")
                    break
                else:
                    print("无效的输入，请重新输入。")
            except Exception as e:
                print(colorama.Fore.RED + f"发生错误：{e}")

    def _posts_to_json(self, posts):
        self.init_json = self._read_json('config/init.json')
        children = []
        for idx, post in enumerate(posts):
            if idx == 0:
                self.init_json['title'] = post.get('title')
                self.init_json['data']['root']['data'] = {
                    'id': str(post.get('rid')),
                    'created': int(time.time() * 1000),
                    'text': (f"【rid】：{post.get('rid')}\n"
                             f"【标题】：{post.get('title')}"
                             f"【时间】：{post.get('replytime')}\n"
                             f"【作者】：{post.get('uname')}\n"
                             f"【用户组】：{post.get('group')}\n"
                             f"【糖尿病类型】：{post.get('disease')}\n"
                             f"【楼层】：{post.get('floor')}\n"
                             f"【文本内容】：{post.get('text', '')[0:20]}"),
                    'replytime': post.get('replytime'),
                    'rid': str(post.get('rid')),
                    'user': str(post.get('uid')),
                    'user_weight': self._convert_user_weight(post.get('group')),
                    'source_text': post.get('text'),
                }
            else:
                children.append({
                    'data': {
                        "id": str(post.get('rid')),
                        "created": int(time.time() * 1000),
                        "text": (f"【rid】：{post.get('rid')}\n"
                                 f"【时间】：{post.get('replytime')}\n"
                                 f"【作者】：{post.get('uname')}\n"
                                 f"【用户组】：{post.get('group')}\n"
                                 f"【糖尿病类型】：{post.get('disease')}\n"
                                 f"【楼层】：{post.get('floor')}\n"
                                 f"【文本内容】：{self._wrap_text(post.get('text', ''), 30)}"),
                        "resource": ["主意", "论证", "疑问", "资料", "支持", "反对", "补充", "质疑"],
                        'replytime': post.get('replytime'),
                        'rid': str(post.get('rid')),
                        'user': str(post.get('uid')),
                        'user_weight': self._convert_user_weight(post.get('group')),
                        'source_text': post.get('text'),
                    },
                    "children": []
                })

        self.init_json['data']['root']['children'] = children

    def _save_init_json(self):
        init_json = json.dumps(self.init_json, ensure_ascii=False)
        with open(os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+source.json'), 'w',
                  encoding='utf-8') as f:
            f.write(init_json)

    @staticmethod
    def _read_json(json_path: str) -> dict:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        timestamp = int(time.time() * 1000)

        if data['created_at'] == 0:
            data['created_at'] = timestamp
            data['data']['root']['data']['created'] = timestamp

        data['updated_at'] = timestamp

        return data

    @staticmethod
    def _wrap_text(text: str, limit: int):
        wrapped_text = ""
        current_line = ""
        for word in text:
            if len(current_line) + len(word) <= limit:
                current_line += word
            else:
                wrapped_text += current_line.strip() + "\n"
                current_line = word
        wrapped_text += current_line.strip()
        return wrapped_text

    def _posts_is_exists(self) -> bool:
        if os.path.exists(os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+source.csv')):
            return True
        else:
            return False

    def _read_local_posts(self) -> List[Dict]:
        print(colorama.Fore.RED + "检测到本地文件，读取本地文件")
        df = pd.read_csv(os.path.join(self.work_path, f'{self.idx}+{self.pid}/{self.idx}+{self.pid}+source.csv'),
                         sep='\t')
        return [row.to_dict() for ind, row in df.iterrows()]

    def _save_session(self, user: str):
        if not os.path.exists(os.path.join(self.work_path, f'session')):
            os.makedirs(os.path.join(self.work_path, f'session'))

        with open(os.path.join(self.work_path, f'session/{user}.pkl'), 'wb') as f:
            pickle.dump(self.login_session, f)

    def _load_session(self, user: str) -> Optional[requests.Session]:
        if not os.path.exists(os.path.join(self.work_path, f'session/{user}.pkl')):
            return None
        with open(os.path.join(self.work_path, f'session/{user}.pkl'), 'rb') as f:
            session = pickle.load(f)
        return session

    @staticmethod
    def _convert_user_weight(group: str) -> int:
        raise NotImplementedError

    def _login(self, user, password):
        raise NotImplementedError

    def _get_config(self):
        raise NotImplementedError

    def _crawl_post(self):
        raise NotImplementedError
