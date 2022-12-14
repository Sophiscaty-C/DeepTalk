import sqlite3
from creart import create
from utils import *
from database_utils import *
from useful_tools import *

from graia.saya import Saya
from graia.broadcast import Broadcast
from graia.ariadne.app import Ariadne
from graia.ariadne.connection.config import config, WebsocketClientConfig
from graia.ariadne.event.message import *
from graia.ariadne.message.element import Plain, Image, At, Source, App, File, Element, Voice
from graia.ariadne.message.chain import MessageChain
from graia.ariadne.model import Group

global ai
ai=False
flag={}
is_reply={}
data_tag={}

def create_database_default(group, member):
    if os.path.isdir('./database/'+str(group.id)) is False:
        os.mkdir('./database/'+str(group.id))
        Type_dict={0: "DEFAULTTABLE", 1: "TIMETABLE", 2: "ADDRESS", 3: "FILES"}
        set_dict("./database/" + str(group.id) + "/Type_dict.json", Type_dict)
    if (group.id, member.id) not in flag:
        s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
        connect = sqlite3.connect(s)
        cursor = connect.cursor()
        create_sql='''CREATE TABLE IF NOT EXISTS DEFAULTTABLE  
                        (ID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        CONTENT TEXT,
                        TIME DATETIME,
                        TAG TEXT);'''
        cursor.execute(create_sql)
        connect.commit()
        create_sql = '''CREATE TABLE IF NOT EXISTS TIMETABLE 
                        (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                        CONTENT TEXT,
                        TIME DATETIME,
                        TAG TEXT);'''
        cursor.execute(create_sql)
        connect.commit()
        create_sql = '''CREATE TABLE IF NOT EXISTS ADDRESS 
                        (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                        TITLE TEXT,
                        TIME DATETIME,
                        TAG TEXT);'''
        cursor.execute(create_sql)
        connect.commit()
        create_sql = '''CREATE TABLE IF NOT EXISTS FILES 
                        (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                        TITLE TEXT,
                        TIME DATETIME,
                        FILE TEXT,
                        TAG TEXT);'''
        cursor.execute(create_sql)
        connect.commit()
        connect.close()

def operate_datatag(group, member, x):
    data_tag[(group.id, member.id)].clear()
    v, data_type, word_list, search_date_list = x
    if v == 0:
        data_tag[(group.id, member.id)][0] = 0
    elif v==1 or v==3:
        l=[]
        for i in word_list:
            l.append(i)
        for i in search_date_list:
            l.append(i)
        data_tag[(group.id, member.id)][v] = (data_type, l)
    elif v==2:
        data_tag[(group.id, member.id)][v] = data_type

create(Broadcast)
saya = create(Saya)
app = Ariadne(config(2890921034, "1234567890", WebsocketClientConfig("http://localhost:8080")))
ignore = ["__init__.py", "__pycache__"]

with saya.module_context():
    for module in os.listdir("modules"):
        if module in ignore:
            continue
        try:
            if os.path.isdir(module):
                saya.require(f"modules.{module}")
            else:
                saya.require(f"modules.{module.split('.')[0]}")
        except ModuleNotFoundError:
            pass

@app.broadcast.receiver("GroupMessage")
async def group_message_listener(app: Ariadne, group: Group, message: MessageChain, member: Member):
    global ai
    time=get_time(message)
    create_database_default(group, member)
    Type_dict = get_dict("./database/" + str(group.id) + "/Type_dict.json")
    t=(group.id, member.id)
    if t not in flag:
        flag[t]=""
        is_reply[t]=False
        data_tag[t]={}
    if is_reply[t] is False:
        operate_datatag(group, member, get_model(message.as_persistent_string(), time))
        if 0 in data_tag[t] and data_tag[t][0]==0:
            s_message=message.as_persistent_string()
            if ai is False:
                if s_message=="??????AI":
                    ai=True
                    await app.send_message(group, MessageChain([Plain("?????????")]))
            elif ai is True:
                if s_message=="??????AI":
                    ai=False
                else:
                    s=get_reply(s_message)
                    await app.send_message(group, MessageChain([Plain(s)]))
                    # reply=get_replys(message.as_persistent_string())
                    # msg = MessageChain([Plain(reply)])
                    # await app.send_message(group, msg)
        elif 1 in data_tag[t] and data_tag[t][1] is not None:
            s, p=select_database(group, member, data_tag[t][1], Type_dict)
            is_reply[t]=True
            await app.send_message(group, MessageChain([Plain(s+"\n????????????"+str(p)+"???\n???????????????????????????????????????????????????????????????")]))#number key update del
        elif 2 in data_tag[t] and data_tag[t][2] is not None:
            is_reply[t]=True
            await app.send_message(group, MessageChain([Plain("?????????????????????????????????????????????????????????")]))#add define
        elif 3 in data_tag[t] and data_tag[t][3] is not None:
            s, p = select_database(group, member, data_tag[t][3], Type_dict)
            is_reply[t]=True
            await app.send_message(group, MessageChain([Plain(s+"\n????????????"+str(p)+"???\n?????????????????????(Y/N)")]))
    else:
        if 1 in data_tag[t] and data_tag[t][1] is not None:
            if message.as_persistent_string().isdigit():
                s, p=select_database(group, member, data_tag[t][1], Type_dict, message.as_persistent_string(), int(message.as_persistent_string()))
                await app.send_message(group, MessageChain([Plain(s)]))
            elif message.as_persistent_string().startswith("key"):
                s, p=select_database(group, member, data_tag[t][1], Type_dict, message.as_persistent_string())
                await app.send_message(group, MessageChain([Plain(s)]))
            elif message.as_persistent_string().startswith("update"):
                update_database(group, member, data_tag[t][1], message.as_persistent_string(), Type_dict)
                await app.send_message(group, MessageChain([Plain("??????????????????")]))
            elif message.as_persistent_string().startswith("del"):
                delete_database(group, member, data_tag[t][1], Type_dict, message.as_persistent_string())
                await app.send_message(group, MessageChain([Plain("??????????????????")]))
            else:
                is_reply[t] = False
        elif 2 in data_tag[t] and data_tag[t][2] is not None:
            if message.as_persistent_string().startswith("add"):
                insert_database(group, member, data_tag[t][2], message.as_persistent_string(), Type_dict)
                await app.send_message(group, MessageChain([Plain("??????????????????")]))
            elif message.as_persistent_string().startswith("define"):
                r=create_database(group, member, message.as_persistent_string(), Type_dict)  # ??????????????????
                storagedata_type_update(r)
                await app.send_message(group, MessageChain([Plain("??????????????????")]))
            else:
                is_reply[t]=False
        elif 3 in data_tag[t] and data_tag[t][3] is not None:
            if message.as_persistent_string()=="Y":
                delete_database(group, member, data_tag[t][3], Type_dict)
                await app.send_message(group, MessageChain([Plain("??????????????????")]))
            else:
                is_reply[t] = False
    if str(message.get(Element)[1])=="@2890921034":
        s_message=MessageChain([Plain("Bot????????????\n?????????/?????????AI" +
                                    "\n?????????????????????" +
                                    "\nwby??????" +
                                    "\n?????? ????????????d????????????" +
                                    "\n?????? ???3p/4p/??????/???1/???2???" +
                                    "\n????????? ????????????\n???????????? ????????? ?????????" +
                                    "\n?????? ??????????????????" +
                                    "\n?????? ????????? ?????? ????????? ??????(50-999)???" +
                                    "\nGloomhaven ???????????????/???????????????" +
                                    "\n?????????????????????" +
                                    "\n??????????????????????????????" +
                                    "\n?????????AI???????????????????????????"), Image(path="C:/Users/SophiscatyC/Pictures/Fire.png")])
        await app.send_message(group, s_message)

app.launch_blocking()