import threading
import re
from graia.ariadne.app import Ariadne
from graia.ariadne.exception import AccountMuted
from graia.ariadne.event.message import GroupMessage
from graia.saya import Saya, Channel
from graia.saya.builtins.broadcast.schema import ListenerSchema
from graia.ariadne.message.chain import MessageChain
from graia.ariadne.model import Group

# 插件信息
__name__ = "Repeater"
__description__ = "复读🐓（x"
__author__ = "SAGIRI-kawaii"
__usage__ = "两个相同message即可触发复读"

saya = Saya.current()
channel = Channel.current()
group_repeat = {}
lock = threading.Lock()

channel.name(__name__)
channel.description(f"{__description__}\n使用方法：{__usage__}")
channel.author(__author__)

@channel.use(ListenerSchema(listening_events=[GroupMessage]))
async def repeater(app: Ariadne, message: MessageChain, group: Group):
    group_id = group.id
    message_serialization = message.as_persistent_string()
    # message_serialization = message_serialization.replace(
    #     "[mirai:source:" + re.findall(r'\[mirai:source:(.*?)]', message_serialization, re.S)[0] + "]",
    #     ""
    # )

    # lock.acquire()
    if group_id in group_repeat.keys():
        group_repeat[group.id]["lastMsg"] = group_repeat[group.id]["thisMsg"]
        group_repeat[group.id]["thisMsg"] = message_serialization
        if group_repeat[group.id]["lastMsg"] != group_repeat[group.id]["thisMsg"]:
            group_repeat[group.id]["stopMsg"] = ""
        else:
            if group_repeat[group.id]["thisMsg"] != group_repeat[group.id]["stopMsg"]:
                group_repeat[group.id]["stopMsg"] = group_repeat[group.id]["thisMsg"]
                try:
                    await app.send_group_message(group, message.as_sendable())
                except AccountMuted:
                    pass
    else:
        group_repeat[group_id] = {"lastMsg": "", "thisMsg": "", "stopMsg": ""}
        print(group_repeat)
    # lock.release()
