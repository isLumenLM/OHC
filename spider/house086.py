# -*- coding: utf-8 -*-
# @Time    : 2024/3/26 16:36
# @Author  : Lumen
# @File    : house086.py
import re
from datetime import datetime
from typing import List, Optional, Dict

import requests
from lxml import etree

from spider.untils import headers
from spider.untils.requester import get
from spider.untils.selector import Selector


def user_parse(uid: str, delay: Optional[List[int]] = None):
    # TODO 用户等级
    pass


def post_parse(pid: int, delay: Optional[List[int]] = None) -> List[Dict]:
    if delay is None:
        delay = [1, 3]

    url = 'https://www.house086.com/thread-{}-{}-1.html'
    page = 0
    session = requests.Session()
    posts = []
    while True:
        page += 1

        response = get(url=url.format(pid, page), headers=headers, session=session, time_delay=delay)
        root = etree.HTML(response.text)
        selector = Selector(root=root)

        # 获取总共页码
        regx = '//div[@class="pgs mtm mbm cl"]/div[@class="pg"]/label/span/text()'
        total_page = selector.xpath(regx).get()
        if total_page:
            total_page = int(total_page.split(' ')[2])
        else:
            total_page = 1
        if page > total_page:
            break

        # 帖子基本信息
        if page == 1:
            regx = '//span[@id="thread_subject"]/text()'
            title = selector.xpath(regx).get()

            regx = '//div[@class="pti"]/div[@class="authi"]/em/text()'
            posttime = selector.xpath(regx).get()
            posttime = datetime.strptime(posttime.split('发表于')[1].strip(), '%Y-%m-%d %H:%M:%S')

            regx = '//div[@id="postlist"]/div[starts-with(@id, "post_")][1]//td[@class="pls"]/div/div/div/a[@class="xw1"]/@href'
            uid = selector.xpath(regx).get()
            uid = uid.split("-")[-1].split('.')[0]

            regx = '//div[@id="postlist"]/div[starts-with(@id, "post_")][1]//td[@class="pls"]/div/div/div/a[@class="xw1"]/text()'
            uname = selector.xpath(regx).get()

            regx = '//*[contains(@id,"postmessage")]'
            posttext = selector.xpath(regx)
            if posttext:
                regx = ('./descendant-or-self::*[not(@class="xg1 y") '
                        'and not(@class="xg1") '
                        'and not(@class="jammer")'
                        'and not(@class="mag_viewthread")'
                        'and not(@style="display:none")'
                        'and not(contains(text(), "下载附件"))'
                        'and not(starts-with(text(), "image"))'
                        'and not(@type="text/javascript") '
                        ']/text()')
                posttext = ''.join([i.get().strip() for i in posttext[0].xpath(regx)])

            regx = '//div[@id="postlist"]/div[starts-with(@id, "post_")][1]/@id'
            rid = selector.xpath(regx).get().split('_')[1]

            user_parse(uid)

            posts.append({
                'rid': rid,
                'pid': pid,
                'title': title,
                'uid': uid,
                'uname': uname,
                'replytime': posttime.strftime('%Y-%m-%d %H:%M:%S'),
                'text': posttext,
                'floor': 1
            })

        # 回帖
        reply_div = selector.xpath('//div[@id="postlist"]/div[starts-with(@id, "post_")]')
        for i, r in enumerate(reply_div):
            if page == 1 and i == 0:
                continue

            regx = './/td[@class="plc"]/div[@class="pi"]/strong/a/text()'
            floor = r.xpath(regx).get()
            if not floor:
                continue

            regx = './/@id'
            rid = r.xpath(regx).get()
            if rid:
                rid = rid.split('_')[1]

            regx = './/td[@class="pls"]/div/div/div/a[@class="xw1"]/@href'
            uid = r.xpath(regx).get()
            if uid:
                uid = uid.split("-")[-1].split('.')[0]
            else:
                continue

            regx = './/td[@class="pls"]/div/div/div/a[@class="xw1"]/text()'
            uname = r.xpath(regx).get()

            regx = './/*[contains(@id, "authorposton")]/text()'
            replytime = r.xpath(regx).get()
            if replytime != '发表于 ':
                replytime = datetime.strptime(replytime.split('发表于 ')[1], '%Y-%m-%d %H:%M:%S')
            else:
                regx = './/*[contains(@id, "authorposton")]/span/@title'
                replytime = datetime.strptime(r.xpath(regx).get(), '%Y-%m-%d %H:%M:%S')

            regx = './/*[contains(@id, "postmessage")]/text()'
            text = ''.join([i.get().strip() for i in r.xpath(regx)])

            user_parse(uid)

            posts.append({
                'rid': rid,
                'pid': pid,
                'uid': uid,
                'uname': uname,
                'replytime': replytime.strftime('%Y-%m-%d %H:%M:%S'),
                'text': text,
                'floor': floor
            })

    return posts

