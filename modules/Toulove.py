import os
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
import xlrd

__name__ = "Toulove"

saya = Saya.current()
channel = Channel.current()
group_repeat = {}
lock = threading.Lock()

channel.name(__name__)

@channel.use(ListenerSchema(listening_events=[GroupMessage]))
async def func(app: Ariadne, message: MessageChain, group: Group):
    if message.as_persistent_string().startswith("刀男 "):
        name=message.as_persistent_string().split(' ')[1]
        data = xlrd.open_workbook("resources/刀男/touken.xlsx")
        table = data.sheets()[0]
        l=[]
        for i in range(table.nrows):
            if table.cell_value(i, 2)==name:
                l=table.row(i)
                break
        s0=name+" 刀剑番号："+str(int(l[0].value))+"\n"
        s=""
        s += "\n稀有度：" + str(int(l[1].value)) + " "
        s += "刀剑种类：" + l[3].value + "\n"
        s += "生存：" + str(int(l[4].value)) + " "
        s += "打击：" + str(int(l[5].value)) + "\n"
        s += "防御：" + str(int(l[6].value)) + " "
        s += "机动：" + str(int(l[7].value)) + "\n"
        s += "冲力：" + str(int(l[8].value)) + " "
        s += "必杀：" + str(int(l[9].value)) + "\n"
        s += "侦察：" + str(int(l[10].value)) + " "
        s += "隐蔽：" + str(int(l[11].value))
        f=""
        for filename in os.listdir("resources/刀男/images"):
            if "极" in name:
                if "极" in filename and name in filename:
                    f=filename
                    break
            else:
                if "极" not in filename and name in filename:
                    f=filename
                    break
        await app.send_message(group, MessageChain([Plain(s0), Image(path="resources/刀男/images/"+f), Plain(s)]))