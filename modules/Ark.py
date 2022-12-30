import requests
import threading
from graia.ariadne.app import Ariadne
from graia.ariadne.exception import AccountMuted
from graia.ariadne.event.message import GroupMessage
from graia.ariadne.message.element import Plain
from graia.ariadne.message.parser.base import ContainKeyword
from graia.saya import Saya, Channel
from graia.saya.builtins.broadcast.schema import ListenerSchema
from graia.ariadne.message.chain import MessageChain
from graia.ariadne.model import Group

__name__ = "Ark"

saya = Saya.current()
channel = Channel.current()
group_repeat = {}
lock = threading.Lock()

channel.name(__name__)

@channel.use(ListenerSchema(listening_events=[GroupMessage]))
async def func(app: Ariadne, message: MessageChain, group: Group):
    if message.as_persistent_string().startswith("舟游抽卡 "):
        t=message.as_persistent_string().split(' ')
        password, username=t[2], t[1]
        url_get_token = 'https://as.hypergryph.com/user/auth/v1/token_by_phone_password'
        json = {
            "password": password,
            "phone": username
        }
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        re = requests.post(headers=headers, url=url_get_token, json=json, timeout=2)
        # print(re.json()["msg"])
        jfile={"list":[]}
        if re.json()["msg"] == "OK":
            flag=0
            token = re.json()["data"]["token"]
            # print(token)
            for page in range(1, 11):
                url_get_msg = 'https://ak.hypergryph.com/user/api/inquiry/gacha?page=%d&token=%s' % (page, token)
                re_msg = requests.get(url=url_get_msg)
                if "data" in re_msg.json():
                    flag=1
                    list_chars = re_msg.json()["data"]["list"]
                    jfile["list"].append(list_chars)
            if flag==1:
                dict = {}
                new = []
                for k in jfile["list"]:
                    for l in k:
                        if str(l) == "":
                            continue
                        pool = l["pool"]
                        temp = {}
                        if pool not in dict:
                            temp[3] = []
                            temp[4] = []
                            temp[5] = []
                            temp[6] = []
                            dict[pool] = temp
                        for i in l["chars"]:
                            rarity = i["rarity"] + 1
                            dict[pool][rarity].append(i["name"])
                            if i["isNew"] == True and i["name"] not in new:
                                new.append(i["name"])
                text = "【以下为最近30天内的抽卡总览】\n"
                for d in dict:
                    six = len(dict[d][6])
                    five = len(dict[d][5])
                    four = len(dict[d][4])
                    three = len(dict[d][3])
                    sum = six + five + four + three
                    text += "【" + d + "】"
                    text += "\n6星：" + str(six) + " 出货率：" + str(round(float(six) / float(sum), 2))
                    text += "\n5星：" + str(five) + " 出货率：" + str(round(float(five) / float(sum), 2))
                    text += "\n4星：" + str(four) + " 出货率：" + str(round(float(four) / float(sum), 2))
                    text += "\n3星：" + str(three) + " 出货率：" + str(round(float(three) / float(sum), 2))
                    for i in dict[d]:
                        text += "\n" + str(i) + "星："
                        for j in dict[d][i]:
                            text += j + " "
                    text += "\n"
                text += "获得新干员：\n"
                for s in new:
                    text += s + " "
                await app.send_message(group, MessageChain([Plain(text)]))
            else:
                await app.send_message(group, MessageChain([Plain("No Data......")]))
        else:
            await app.send_message(group, MessageChain([Plain("Some Errors Occured......")]))