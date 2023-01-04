import threading
from graia.ariadne.app import Ariadne
from graia.ariadne.event.message import GroupMessage
from graia.ariadne.message.element import Plain, Image
from graia.saya import Saya, Channel
from graia.saya.builtins.broadcast.schema import ListenerSchema
from graia.ariadne.message.chain import MessageChain
from graia.ariadne.model import Group
from useful_tools import *

__name__ = "Basement"

saya = Saya.current()
channel = Channel.current()
lock = threading.Lock()

channel.name(__name__)

@channel.use(ListenerSchema(listening_events=[GroupMessage]))
async def func(app: Ariadne, message: MessageChain, group: Group):
    s_message = message.as_persistent_string()
    if s_message == "wby写文":
        await app.send_message(group, MessageChain([Plain("你先别急，让lmy先急")]))
    elif s_message.startswith("骰子 "):
        l = []
        temp = message.as_persistent_string().split(' ')[1].split('d')
        x, y = int(temp[0]), int(temp[1])
        for i in range(x):
            l.append(random.randint(1, y))
        await app.send_message(group, MessageChain([Plain(str(l))]))
    elif s_message.startswith("接龙 "):
        s = message.as_persistent_string().split(' ')[1]
        if s == "3p":
            await app.send_message(group, MessageChain([Plain(str(random_num(3)))]))
        elif s == "4p":
            await app.send_message(group, MessageChain([Plain(str(random_num(4)))]))
        elif s == "题材":
            await app.send_message(group, MessageChain([Plain(read_file("resources/接龙/题材.txt"))]))
        elif s == "池1":
            await app.send_message(group, MessageChain([Plain(read_file("resources/接龙/池1.txt"))]))
        elif s == "池2":
            await app.send_message(group, MessageChain([Plain(read_file("resources/接龙/池2.txt"))]))
    elif s_message.startswith("Gloomhaven "):
        await app.send_message(group, MessageChain([Plain("本功能正在开发中")]))
    elif s_message == "数据库使用说明":
        d_message = MessageChain([Plain("数据库使用说明：\n" +
                                        "可能涉及到的消息发送格式：\n" +
                                        "页码：页码范围内的数字\n" +
                                        "查询关键词：key 关键词\n" +
                                        "修改：update id或id起始值-id终止值 字段名1 关键词1...\n" +
                                        "删除：del id或id起始值-id终止值\n" +
                                        "增加：add 数据1 数据2...\n" +
                                        "自定义：define 表名 字段名1 字段名2...\n")])
        await app.send_message(group, d_message)