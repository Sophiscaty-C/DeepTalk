import threading
from graia.ariadne.app import Ariadne
from graia.ariadne.event.message import GroupMessage
from graia.ariadne.message.element import Plain, Image
from graia.saya import Saya, Channel
from graia.saya.builtins.broadcast.schema import ListenerSchema
from graia.ariadne.message.chain import MessageChain
from graia.ariadne.model import Group
import json
import random
from useful_tools import get_name

__name__ = "ToukenBrowser"

saya = Saya.current()
channel = Channel.current()
group_repeat = {}
lock = threading.Lock()

channel.name(__name__)

@channel.use(ListenerSchema(listening_events=[GroupMessage]))
async def func(app: Ariadne, message: MessageChain, group: Group):
    if message.as_persistent_string().startswith("锻刀 "):
        materials=' '.join(message.as_persistent_string().split()).split(' ')
        if len(materials)!=5:
            s0="这些材料不是锻刀材料......"
        else:
            s=""
            for i in range(1, 5):
                s+=materials[i]
                if i!=4:
                    s+=" "
            with open('D:/Study/Projects/MyBot/resources/刀男/test_data.json', 'r', encoding='utf8') as json_file:
                json_data = json.load(json_file)
            if s not in json_data:
                s0="似乎没有这样的公式......"
            else:
                t=json_data[s]
                r=round(random.uniform(0, 1), 3)
                sum=0.000
                name=0
                for k in range(len(t)):
                    sum+=t[k][1]
                for k in range(len(t)):
                    t[k][1]=round(t[k][1]/sum, 3)
                    if t[k][1]==0.0:
                        t[k][1]=0.001
                    print(t[k])
                ss=0.000
                for k in range(len(t)):
                    ss += t[k][1]
                    if r<=ss:
                        name=t[k][0]
                        break
                if name==0:
                    s0="Some Errors Occured......"
                else:
                    s0=get_name(str(name))
        await app.send_message(group, MessageChain([Plain(s0)]))