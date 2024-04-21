# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 15:44
# @Author  : Lumen
# @File    : tmjy.py

import os
from configparser import ConfigParser

import pandas as pd

import sys
sys.path.append('..')
from script.manager import PostManager
from spider.tmjy import post_parse, login


class TMJYPostManager(PostManager):

    def __init__(self):
        super().__init__()
        self.work_path = 'data/tmjy/'

    @staticmethod
    def _convert_user_weight(group: str) -> int:
        if group in ('学习班', '禁止发言', '禁止访问'):
            return 0
        elif group in ('未晋级', '等待验证会员'):
            return 1
        elif group in ('托儿所', '观察审核', '幼儿园', '小学'):
            return 2
        elif group in ('初中', '高中'):
            return 3
        elif group in ('大专', '甜蜜义工'):
            return 4
        elif group == '预科':
            return 5
        elif group == '本科':
            return 6
        elif group == '硕士':
            return 7

        # TODO
        elif group == '博士':
            return 9
        elif group in ('版主', '荣誉版主', '特邀嘉宾', '特邀客服'):
            return 10
        elif group in ('超级版主', '管理员'):
            return 11
        else:
            raise ValueError(f'用户组：【{group}】未知，不能转换成权重')

    def _login(self, user, password):
        return login(user, password)

    def _get_config(self):
        config = ConfigParser()
        config.read('config/config.ini')
        user = config.get('tmjy', 'user')
        password = config.get('tmjy', 'password')
        return user, password

    def _crawl_post(self):
        if os.path.exists(os.path.join(self.work_path, f'{self.pid}/source.csv')):
            print("检测到本地文件，读取本地文件")
            df = pd.read_csv(os.path.join(self.work_path, f'{self.pid}/source.csv'), sep='\t')
            self.posts = [row.to_dict() for ind, row in df.iterrows()]
        else:
            self.posts = post_parse(self.pid, self.login_session, delay=[0, 1])


if __name__ == '__main__':
    manager = TMJYPostManager()
    manager.prompt_user()
