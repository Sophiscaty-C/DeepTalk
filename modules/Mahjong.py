import requests
import threading
from graia.ariadne.app import Ariadne
from graia.ariadne.exception import AccountMuted
from graia.ariadne.event.message import GroupMessage
from graia.ariadne.message.element import Plain, Image
from graia.ariadne.message.parser.base import ContainKeyword
from graia.saya import Saya, Channel
from graia.saya.builtins.broadcast.schema import ListenerSchema
from graia.ariadne.message.chain import MessageChain
from graia.ariadne.model import Group
import random
import os

__name__ = "Mahjong"

saya = Saya.current()
channel = Channel.current()
temp = {}
lock = threading.Lock()

channel.name(__name__)

@channel.use(ListenerSchema(listening_events=[GroupMessage]))
async def func(app: Ariadne, message: MessageChain, group: Group):
    mes=message.as_persistent_string()
    name = "resources/何切300/Question/"
    name2 = "resources/何切300/Answer/"
    if mes=="何切！":
        while(True):
            r = random.randint(1, 300)
            f = str(r) + "_0.png"
            if os.path.exists(name+f):
                os.rename(name+f, name + str(r) + "_1.png")
                os.rename(name2 + f, name2 + str(r) + "_1.png")
                f=str(r) + "_1.png"
                break
        temp["何切！"] = (r, f)
        await app.send_message(group, MessageChain([Image(path=name+f)]))
    if mes=="答案" and "何切！" in temp:
        f=open("resources/何切300/Text.txt", "r", encoding='utf-8')
        l=f.readlines()[temp["何切！"][0]-1]
        await app.send_message(group, MessageChain([Image(path=name2+temp["何切！"][1]), Plain(l)]))
        temp.clear()