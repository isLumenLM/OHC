# -*- coding: utf-8 -*-
# @Time    : 2024/4/17 15:08
# @Author  : Lumen
# @File    : manager.py
import abc
import json
import os
import time
import webbrowser
from pprint import pprint

import pandas as pd

import sys

sys.path.append('..')
from consensus.graph import DynamicDiscussionGraph


class PostManager(metaclass=abc.ABCMeta):

    def __init__(self):
        self.pid = None
        self.posts = None

        user, password = self._get_config()
        self.login_session = self._login(user, password)
        self.init_json = None
        self.labeled_json = None

        self.work_path = 'data/'

    def _check(self):
        if self.pid is None:
            print("请先切换到一个帖子。")
            return False
        return True

    def switch_post(self):
        try:
            new_pid = input("请输入新的帖子ID: ")
            self.pid = int(new_pid)
            print(f"已切换到帖子ID: {self.pid}")
        except ValueError:
            print("输入错误，请输入有效的帖子ID。")

        print("正在爬取帖子...")
        self._crawl_post()
        self._posts_to_json(self.posts)
        if not os.path.exists(os.path.join(self.work_path, f'{self.pid}')):
            os.makedirs(os.path.join(self.work_path, f'{self.pid}'))
        self._save_init_json()

        pd.DataFrame(self.posts).to_csv(
            os.path.join(self.work_path, f'{self.pid}/{self.pid}.csv'),
            index=False,
            encoding='utf-8_sig',
            sep='\t'
        )
        print(f"帖子爬取完毕, 数据保存到{os.path.join(self.work_path, f'{self.pid}')}")

    @staticmethod
    def open_flow():
        webbrowser.open('https://www.baidufe.com/fehelper/naotu/index.html')

    def view_post_counts(self):
        if not self._check():
            return
        reply_times = [post.get('replytime') for post in self.posts]
        print(f"发帖时间：{reply_times[0]}")
        print(f"最后回复时间：{reply_times[-1]}")

        print("可选时间粒度：")
        print("1.小时：H")
        print("2.天：D")
        print("3.周：W")
        print("4.月：M")
        print("5.年：Y")
        print("例如，以25天进行分箱，时间粒度输入D，周期输入25")
        granularity = input("请输入时间粒度（'H', 'D', 'W', 'M', 'Y'）：")
        number = int(input("请输入周期（正整数）："))
        time_bins = DynamicDiscussionGraph.time_binning(reply_times, granularity=granularity, number=number, draw=True)
        print("分箱时间区间：")
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
        print(f"并将已经标注的数据放入{os.path.join(self.work_path, f'{self.pid}/{self.pid}_labeled.json')}")
        granularity = input("请输入时间粒度（'H', 'D', 'W', 'M', 'Y'）：")
        number = int(input("请输入周期（正整数）："))
        ddg = DynamicDiscussionGraph()
        ddg.load_graphs_from_json(
            json_path=os.path.join(self.work_path, f'{self.pid}/{self.pid}_labeled.json'),
            granularity=granularity,
            number=number
        )
        print(ddg.get_consensus())
        ddg.draw()
        skewness = ddg.get_skewness(draw=True)
        pd.DataFrame([skewness]).to_csv(os.path.join(self.work_path, f'{self.pid}/{self.pid}_skewness.csv'), index=False)

    def prompt_user(self):
        while True:
            print(f"目前正在处理的帖子id: {self.pid}")
            print("请选择功能：")
            print("1. 切换帖子")
            print("2. 打开脑图")
            print("3. 查看时间分箱后的帖子数量")
            print("4. 计算偏度和共识度")
            print("5. 退出")

            choice = input("请输入你的选择（1-5）: ")

            if choice == '1':
                self.switch_post()
            elif choice == '2':
                self.open_flow()
            elif choice == '3':
                self.view_post_counts()
            elif choice == '4':
                self.calculate_skewness_and_consensus()
            elif choice == '5':
                print("退出程序.")
                break
            else:
                print("无效的输入，请重新输入。")

    def _posts_to_json(self, posts):
        self.init_json = self._read_json('config/init.json')
        children = []
        for idx, post in enumerate(posts):
            if idx == 0:

                self.init_json['data']['root']['data'] = {
                    'id': post.get('rid'),
                    'created': int(time.time() * 1000),
                    'text': (f"【rid】：{post.get('rid')}\n"
                             f"【标题】：{post.get('title')}"
                             f"【时间】：{post.get('replytime')}\n"
                             f"【作者】：{post.get('uname')}\n"
                             f"【用户组】：{post.get('group')}\n"
                             f"【糖尿病类型】：{post.get('disease')}\n"
                             f"【楼层】：{post.get('floor')}\n"
                             f"【文本内容】：{post.get('text')[0:20]}"),
                    'replytime': post.get('replytime'),
                    'rid': post.get('rid'),
                    'user_weight': self._convert_user_weight(post.get('group')),
                }
            else:
                children.append({
                    'data': {
                        "id": post.get('rid'),
                        "created": int(time.time() * 1000),
                        "text": (f"【rid】：{post.get('rid')}\n"
                                 f"【时间】：{post.get('replytime')}\n"
                                 f"【作者】：{post.get('uname')}\n"
                                 f"【用户组】：{post.get('group')}\n"
                                 f"【糖尿病类型】：{post.get('disease')}\n"
                                 f"【楼层】：{post.get('floor')}\n"
                                 f"【文本内容】：{self._wrap_text(post.get('text'), 30)}"),
                        "resource": ["主意", "论证", "疑问", "资料", "支持", "反对", "补充", "质疑"],
                        'replytime': post.get('replytime'),
                        'rid': post.get('rid'),
                        'user_weight': self._convert_user_weight(post.get('group')),
                    },
                    "children": []
                })

        self.init_json['data']['root']['children'] = children

    def _save_init_json(self):
        init_json = json.dumps(self.init_json, ensure_ascii=False)
        with open(os.path.join(self.work_path, f'{self.pid}/{self.pid}.json'), 'w', encoding='utf-8') as f:
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

    @staticmethod
    def _convert_user_weight(group: str) -> int:
        raise NotImplementedError

    def _login(self, user, password):
        raise NotImplementedError

    def _get_config(self):
        raise NotImplementedError

    def _crawl_post(self):
        raise NotImplementedError

    def _make_data_folder(self):
        raise NotImplementedError
