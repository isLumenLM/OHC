# -*- coding: utf-8 -*-
# @Time    : 2024/3/26 17:14
# @Author  : LiuMing
# @Email   : liuming04073@zulong.com
# @File    : post2json.py
import json
import time

from spider.house086 import post_parse

pid = 292368


def read_json(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    timestamp = int(time.time() * 1000)

    if data['created_at'] == 0:
        data['created_at'] = timestamp
        data['data']['root']['data']['created'] = timestamp

    data['updated_at'] = timestamp

    return data


def wrap_text(text: str, limit: int):
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


init_json = read_json('init.json')
posts = post_parse(pid)
children = []

for idx, post in enumerate(posts):
    if idx == 0:
        init_json['data']['root']['data']['text'] = (
            f"【时间】：{post.get('replytime')}\n【作者】：{post.get('uname')}\n【文本内容】：{post.get('text')[0:20]}")
    else:
        children.append({
            'data': {
                "id": idx + 1,
                "created": int(time.time() * 1000),
                "text": f"【时间】：{post.get('replytime')}\n【作者】：{post.get('uname')}\n【文本内容】：{wrap_text(post.get('text'), 30)}",
                "resource": ["主意", "论证", "疑问", "信息", "支持", "反对", "补充", "疑问"]
            },
            "children": []
        })

init_json['data']['root']['children'] = children
init_json = json.dumps(init_json, ensure_ascii=False)
with open(f'{pid}.json', 'w', encoding='utf-8') as f:
    f.write(init_json)
