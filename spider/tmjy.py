# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 15:17
# @Author  : Lumen
# @File    : tmjy.py
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Optional, List, Dict
from urllib.parse import parse_qs, urlparse

import requests
from PIL import Image
from lxml import etree
from urlextract import URLExtract

from spider.untils import headers
from spider.untils.requester import get
from spider.untils.selector import Selector


def login(username: str, password: str) -> requests.Session:

    def _get_seccode(login_selector: Selector):
        img = login_selector.xpath('//img[@class="seccodeimg"]/@src').get()
        img_params = parse_qs(img.split('?')[1].replace("&amp;", "&"))
        update = img_params.get('update', [None])[0]
        idhash = img_params.get('idhash', [None])[0]
        response = get(f"https://bbs.tnbz.com/misc.php?mod=seccode&update={update}&idhash={idhash}&mobile=2",
                       headers={**phone_headers,
                                **{'referer': 'https://bbs.tnbz.com/member.php?mod=logging&action=login&mobile=2'}},
                       session=session)
        image = Image.open(BytesIO(response.content))
        image.show()

        return response.content

    # 手机headers
    phone_headers = {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Mobile Safari/537.36',
    }

    session = requests.Session()

    response = get('https://bbs.tnbz.com/member.php?mod=logging&action=login&mobile=2', headers=phone_headers, session=session)
    root = etree.HTML(response.text.encode('utf-8'))
    selector = Selector(root=root)
    seccodehash = selector.xpath('//input[@name="seccodehash"]/@value').get()
    formhash = selector.xpath('//input[@name="formhash"]/@value').get()
    cookietime = selector.xpath('//input[@name="cookietime"]/@value').get()
    loginhash = selector.xpath('//form[@id="loginform"]/@action').get()
    params = parse_qs(loginhash.split('?')[1])
    loginhash = params.get('loginhash', [None])[0]

    img_content = _get_seccode(selector)
    seccodeverify = input("请输入验证码：")
    data = {
        'formhash': formhash,
        'fastloginfield': 'username',
        'cookietime': cookietime,
        'username': username,
        'password': password,
        'questionid': 0,
        'answer': '',
        'seccodehash': seccodehash,
        'seccodeverify': seccodeverify
    }

    post_url = f'https://bbs.tnbz.com/member.php?mod=logging&action=login&loginsubmit=yes&loginhash={loginhash}&mobile=2&handlekey=loginform&inajax=1'
    response = session.post(
        url=post_url,
        data=data,
        headers=phone_headers
    )
    if "欢迎您回来" in response.text:
        print("登录成功！")
        if not os.path.exists('seccode/'):
            os.makedirs('seccode/')
        with open(f"seccode/{seccodeverify}.jpg", "wb") as img_file:
            img_file.write(img_content)
        return session
    else:
        raise ConnectionError('登陆失败!')


def user_parse(uid: str, session: requests.Session, delay: Optional[List[int]] = None) -> Dict:

    url = 'https://bbs.tnbz.com/home.php?mod=space&uid={}&do=profile'

    response = get(url=url.format(uid), headers=headers, session=session, time_delay=delay)
    html = etree.HTML(response.text)
    selector = Selector(root=html)

    regx = '//title/text()'
    uname = selector.xpath(regx).get()
    if uname:
        uname = uname.replace("【新提醒】", "").split('的个人资料')[0]

    regx = '//div[@class="pbm mbm bbda cl"][last()]/ul/li/em[starts-with(text(),"用户组")]/following-sibling::span/a//text()'
    group = selector.xpath(regx).get()

    regx = '//ul[@id="pbbs"]/li/em[starts-with(text(), "注册时间")]/parent::li/text()'
    registertime = selector.xpath(regx).get()

    regx = '//ul[@class="cl bbda pbm mbm"]/li/a[starts-with(text(), "主题数")]/text()'
    posts = selector.xpath(regx).get()
    if posts:
        posts = posts.split()
        if len(posts) > 1:
            posts = posts[1]
        else:
            posts = 0

    regx = '//ul[@class="cl bbda pbm mbm"]/li/a[starts-with(text(), "回帖数")]/text()'
    replys = selector.xpath(regx).get()
    if replys:
        replys = replys.split()
        if len(replys) > 1:
            replys = replys[1]
        else:
            replys = 0

    regx = '//ul[@class="pf_l cl"]/li/em[starts-with(text(), "糖尿病类型")]/following-sibling::text()'
    disease = selector.xpath(regx).get()
    if not replys:
        disease = '未知'

    user_info = {
        'uid': uid,
        'uname': uname,
        'group': group,
        'registertime': registertime,
        'posts': posts,
        'replys': replys,
        'disease': disease
    }

    return user_info


def post_parse(pid: int, login_session: requests. Session, delay: Optional[List[int]] = None) -> List[Dict]:
    if delay is None:
        delay = [1, 3]

    url = 'https://bbs.tnbz.com/thread-{}-{}-1.html'
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

            regx = '//div[@id="pt"]/div[@class="z"]/a[starts-with(@href, "https://bbs.tnbz.com/forum.php?gid=")]/@href'
            gid = selector.xpath(regx).get()
            if gid:
                gid = re.findall(re.compile(r'gid=(\d+)'), gid)[0]

            regx = '//div[@id="pt"]/div[@class="z"]/a[starts-with(@href, "https://bbs.tnbz.com/forum.php?gid=")]/text()'
            gname = selector.xpath(regx).get()

            regx = '//div[@id="pt"]/div[@class="z"]/a[starts-with(@href, "https://bbs.tnbz.com/forum.php?mod=forumdisplay&fid=")]/@href'
            fid = selector.xpath(regx).get()
            if fid:
                fid = re.findall(re.compile(r'fid=(\d+)'), fid)[0]
            else:
                regx = '//div[@id="pt"]/div[@class="z"]/a[starts-with(@href, "https://bbs.tnbz.com/forum-")]/@href'
                fid = selector.xpath(regx).get()
                if fid:
                    fid = fid.split('-')[1]

            regx = '//div[@id="pt"]/div[@class="z"]/a[starts-with(@href, "https://bbs.tnbz.com/forum.php?mod=forumdisplay&fid=")]/text()'
            fname = selector.xpath(regx).get()
            if not fname:
                regx = '//div[@id="pt"]/div[@class="z"]/a[starts-with(@href, "https://bbs.tnbz.com/forum-")]/text()'
                fname = selector.xpath(regx).get()

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
                posttext = ''.join([i.get().strip().replace('\t', ' ') for i in posttext[0].xpath(regx)])
            else:
                posttext = ''

            regx = '//div[@class="hm ptn"]/span[@class="xi1"][2]/text()'
            replies = selector.xpath(regx).get()

            regx = '//div[@id="postlist"]/div[starts-with(@id, "post_")][1]/@id'
            rid = selector.xpath(regx).get().split('_')[1]

            regx = '//div[@id="postlist"]/div[starts-with(@id, "post_")][1]//div[@class="pct"]//img[@id]'
            imgs = len(selector.xpath(regx))

            # netloc_original = urlparse(response.url).netloc
            # regx = '//*[contains(@id,"postmessage")]//a[starts-with(@href, "https") or starts-with(@href, "http")]/@href'
            # hrefs = selector.xpath(regx).get_all()
            # external_links_count = 0
            # for href in hrefs:
            #     netloc = urlparse(href).netloc
            #     if netloc and netloc != netloc_original:  # 检查域名是否存在且与原始域名不同
            #         external_links_count += 1
            extractor = URLExtract()
            external_links_count = len(extractor.find_urls(posttext))

            user_info = user_parse(uid, login_session, delay)
            posts.append({
                'rid': rid,
                'pid': pid,
                'title': title,
                'uid': uid,
                'uname': uname,
                'replytime': posttime.strftime('%Y-%m-%d %H:%M:%S'),
                'text': posttext,
                'floor': 1,
                'replies': replies,
                'imgs': imgs,
                'external_links_count': external_links_count,
                **user_info
            })

        # 回帖
        reply_div = selector.xpath('//div[@id="postlist"]/div[starts-with(@id, "post_")]')
        for i, r in enumerate(reply_div):
            if page == 1 and i == 0:
                continue

            regx = './/td[@class="plc"]/div[@class="pi"]/strong/a/em/text()'
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
            text = ''.join([i.get().strip().replace('\t', ' ') for i in r.xpath(regx)])

            user_info = user_parse(uid, login_session, delay)

            posts.append({
                'rid': rid,
                'pid': pid,
                'uid': uid,
                'uname': uname,
                'replytime': replytime.strftime('%Y-%m-%d %H:%M:%S'),
                'text': text,
                'floor': floor,
                **user_info
            })

    return posts
