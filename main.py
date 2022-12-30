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
                if s_message=="开启AI":
                    ai=True
                    await app.send_message(group, MessageChain([Plain("爷活辣")]))
                elif s_message=="wby写文":
                    await app.send_message(group, MessageChain([Plain("你先别急，让lmy先急")]))
                elif s_message.startswith("骰子 "):
                    l=[]
                    temp=message.as_persistent_string().split(' ')[1].split('d')
                    x, y=int(temp[0]), int(temp[1])
                    for i in range(x):
                        l.append(random.randint(1, y))
                    await app.send_message(group, MessageChain([Plain(str(l))]))
                elif s_message.startswith("接龙 "):
                    s = message.as_persistent_string().split(' ')[1]
                    if s=="3p":
                        await app.send_message(group, MessageChain([Plain(str(random_num(3)))]))
                    elif s=="4p":
                        await app.send_message(group, MessageChain([Plain(str(random_num(4)))]))
                    elif s=="题材":
                        await app.send_message(group, MessageChain([Plain(read_file("resources/接龙/题材.txt"))]))
                    elif s == "池1":
                        await app.send_message(group, MessageChain([Plain(read_file("resources/接龙/池1.txt"))]))
                    elif s == "池2":
                        await app.send_message(group, MessageChain([Plain(read_file("resources/接龙/池2.txt"))]))
                # elif message.as_persistent_string() == "何切！":
                #     已放入modules
                # elif message.as_persistent_string().startswith("舟游数据 "):
                #     已放入modules
                # elif message.as_persistent_string().startswith("刀男 "):
                #     已放入modules
                # elif message.as_persistent_string().startswith("锻刀 "):
                #     已放入modules
                elif s_message.startswith("Gloomhaven "):
                    await app.send_message(group, MessageChain([Plain("本功能正在开发中")]))
                elif s_message=="数据库使用说明":
                    d_message = MessageChain([Plain("数据库使用说明：\n" +
                                                    "可能涉及到的消息发送格式：\n" +
                                                    "页码：页码范围内的数字\n" +
                                                    "查询关键词：key 关键词\n" +
                                                    "修改：update id或id起始值-id终止值 字段名1 关键词1...\n" +
                                                    "删除：del id或id起始值-id终止值\n" +
                                                    "增加：add 数据1 数据2...\n" +
                                                    "自定义：define 表名 字段名1 字段名2...\n")])
                    await app.send_message(group, d_message)
            elif ai is True:
                if s_message=="关闭AI":
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
            await app.send_message(group, MessageChain([Plain(s+"\n数据共有"+str(p)+"页\n请回复页码、关键词或者需要修改和删除的数据")]))#number key update del
        elif 2 in data_tag[t] and data_tag[t][2] is not None:
            is_reply[t]=True
            await app.send_message(group, MessageChain([Plain("请回复需要增加的数据或者需要自定义的表")]))#add define
        elif 3 in data_tag[t] and data_tag[t][3] is not None:
            s, p = select_database(group, member, data_tag[t][3], Type_dict)
            is_reply[t]=True
            await app.send_message(group, MessageChain([Plain(s+"\n数据共有"+str(p)+"页\n是否确认删除？(Y/N)")]))
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
                await app.send_message(group, MessageChain([Plain("数据修改完成")]))
            elif message.as_persistent_string().startswith("del"):
                delete_database(group, member, data_tag[t][1], Type_dict, message.as_persistent_string())
                await app.send_message(group, MessageChain([Plain("数据删除完成")]))
            else:
                is_reply[t] = False
        elif 2 in data_tag[t] and data_tag[t][2] is not None:
            if message.as_persistent_string().startswith("add"):
                insert_database(group, member, data_tag[t][2], message.as_persistent_string(), Type_dict)
                await app.send_message(group, MessageChain([Plain("数据增加完成")]))
            elif message.as_persistent_string().startswith("define"):
                r=create_database(group, member, message.as_persistent_string(), Type_dict)  # 需要返回表名
                storagedata_type_update(r)
                await app.send_message(group, MessageChain([Plain("数据增加完成")]))
            else:
                is_reply[t]=False
        elif 3 in data_tag[t] and data_tag[t][3] is not None:
            if message.as_persistent_string()=="Y":
                delete_database(group, member, data_tag[t][3], Type_dict)
                await app.send_message(group, MessageChain([Plain("数据删除完成")]))
            else:
                is_reply[t] = False
    if message.has(At(2890921034)):
        s_message=MessageChain([Plain("Bot关键词：\n【开启/关闭】AI" +
                                    "\n数据库使用说明" +
                                    "\nwby写文" +
                                    "\n骰子 【数字】d【数字】" +
                                    "\n接龙 【3p/4p/题材/池1/池2】" +
                                    "\n何切！ 【答案】\n舟游抽卡 【账号 密码】" +
                                    "\n刀男 【角色名称】" +
                                    "\n锻刀 【木炭 玉钢 冷却材 砥石(50-999)】" +
                                    "\nGloomhaven 【角色名称/卡牌名称】" +
                                    "\n触发隐藏功能：" +
                                    "\n发送两条以上相同消息" +
                                    "\n在开启AI状态下发送查询好感"), Image(path="C:/Users/SophiscatyC/Pictures/Fire.png")])
        await app.send_message(group, s_message)

app.launch_blocking()