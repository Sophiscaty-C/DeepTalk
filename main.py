import os
import pytz
import sqlite3
from datetime import datetime, timedelta, timezone
from creart import create
from smart_storage import main, storagedata_type_update, operation_clf_model
from execute import main_predict, Lang
import json
from utils import get_replys

from graia.saya import Saya
from graia.broadcast import Broadcast
from graia.ariadne.app import Ariadne
from graia.ariadne.connection.config import config, WebsocketClientConfig
from graia.ariadne.event.message import *
from graia.ariadne.message.element import Plain, Image, At, Source, App, File, Element, Voice
from graia.ariadne.message.chain import MessageChain
from graia.ariadne.model import Group

flag={}
is_reply={}
data_tag={}

def in_range(n, start, end = 0):
  return start <= n <= end if end >= start else end <= n <= start

def get_dict(s):
    with open(s, 'r', encoding='UTF-8') as f:
        Type_dict = json.load(f)
    return Type_dict

def set_dict(s, Type_dict):
    json_str = json.dumps(Type_dict)
    with open(s, 'w', encoding='UTF-8') as f:
        f.write(json_str)

def get_time(message):
    e = str(message.get(Element)).split(', ')
    time_list = []
    time_list.append(int(e[1].split('(')[1]))
    for i in range(2, 7):
        if i==6:
            if e[i].isdigit():
                time_list.append(int(e[i]))
            else:
                time_list.append(int(1))
        else:
            time_list.append(int(e[i]))
    time = datetime(time_list[0], time_list[1], time_list[2], time_list[3], time_list[4], time_list[5]).replace(tzinfo=pytz.utc)
    time = time.astimezone(timezone(timedelta(hours=8)))
    return time

def get_col_name(cursor, t):
    cursor.execute('pragma table_info({})'.format(t))
    col_name = cursor.fetchall()
    col_name = [x[1] for x in col_name]
    return col_name

def get_model(s, time):
    v, data_type, word_list, search_date_list=main(s, time)
    return v, data_type, word_list, search_date_list

def get_reply(s):
    return main_predict(s)

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

def create_database(group, member, message, Type_dict):
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    create_sql="CREATE TABLE IF NOT EXISTS "
    temp=message.split(' ')
    for i in range(1, len(temp)):
        if i==1:
            create_sql+=temp[i]+" (ID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
            Type_dict[len(Type_dict)]=temp[i]
        elif i<len(temp)-1:
            create_sql+=temp[i]+" TEXT, "
        else:
            create_sql += temp[i] + " TEXT);"
    set_dict("./database/" + str(group.id) + "/Type_dict.json", Type_dict)
    cursor.execute(create_sql)
    connect.commit()
    connect.close()
    return temp[1]

def insert_database(group, member, t, message, Type_dict):
    t = str(t)
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    col_name=get_col_name(cursor, Type_dict[t])
    temp = message.split(' ')
    insert_sql="INSERT INTO "+Type_dict[t]+" ("
    for i in range(1, len(temp)):
        if i<len(temp)-1:
            insert_sql+=col_name[i]+", "
        else:
            insert_sql += col_name[i] + ") VALUES ("
    for i in range(1, len(temp)):
        if i<len(temp)-1:
            insert_sql+="'"+temp[i]+"', "
        else:
            insert_sql +="'"+temp[i]+"')"
    cursor.execute(insert_sql)
    connect.commit()
    connect.close()

def select_database(group, member, t, Type_dict, message="", page=1):
    p, l = t
    p = str(p)
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    col_name=get_col_name(cursor, Type_dict[p])
    st=""
    for i in col_name:
        st+=i+" "
    st+="\n"
    select_sql=""
    if len(l)==0:
        select_sql+="SELECT * FROM "+Type_dict[p]
    elif len(l)!=0 and message=="":
        select_sql += "SELECT * FROM " + Type_dict[p] + " WHERE "
        for i in range(len(col_name)):
            for j in range(len(l)):
                if i<len(col_name)-1 and j<len(l)-1:
                    select_sql += col_name[i] + " LIKE '%" + l[j] + "%' OR "
                else:
                    select_sql += col_name[i] + " LIKE '%" + l[j] + "%'"
    elif message!="":
        select_sql+="SELECT * FROM "+Type_dict[p]+" WHERE "
        for i in range(len(col_name)):
            if i < len(col_name) - 1:
                select_sql+=col_name[i]+" LIKE '%"+message+"%' OR "
            else:
                select_sql += col_name[i] + " LIKE '%" + message + "%'"
    c=cursor.execute(select_sql)
    count=0
    for row in c:
        count+=1
        if (page - 1)*10 < count <= page*10:
            for col in range(len(col_name)):
                st+=str(row[col])+" "
            st+='\n'
        else:
            continue
    connect.close()
    if count%10==0:
        pagecount=int(count/10)
    else:
        pagecount=int(count/10)+1
    return st, pagecount

def update_database(group, member, t, message, Type_dict):
    p, l = t
    p = str(p)
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    temp = message.split(' ')
    update_sql=""
    if temp[1].isdigit():
        id = temp[1]
        for i in range(2, len(temp), 2):
            update_sql += "UPDATE " + Type_dict[p] + " SET "+temp[i]+"='"+temp[i+1]+"' WHERE ID=" + id
    else:
        id1 = temp[1].split('-')[0]
        id2 = temp[1].split('-')[1]
        for i in range(2, len(temp), 2):
            update_sql += "UPDATE " + Type_dict[p] + " SET "+temp[i]+"='"+temp[i+1]+"' WHERE ID>=" + id1 + " AND ID<=" + id2
    cursor.execute(update_sql)
    connect.commit()
    connect.close()
def delete_database(group, member, t, Type_dict, message=""):
    p, l = t
    p = str(p)
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    col_name = get_col_name(cursor, Type_dict[p])
    if message=="":
        select_sql=""
        if len(l) == 0:
            select_sql += "SELECT ID FROM " + Type_dict[p]
        elif len(l) != 0 and message == "":
            select_sql += "SELECT ID FROM " + Type_dict[p] + " WHERE "
            for i in range(len(col_name)):
                for j in range(len(l)):
                    if i < len(col_name) - 1 and j < len(l) - 1:
                        select_sql += col_name[i] + " LIKE '%" + l[j] + "%' OR "
                    else:
                        select_sql += col_name[i] + " LIKE '%" + l[j] + "%'"
        delete_sql="DELETE FROM "+Type_dict[p]+" WHERE ID IN (SELECT ID FROM(" +select_sql+") AS TMP)"
    else:
        temp=message.split(' ')[1]
        if temp.isdigit():
            id=temp
            delete_sql="DELETE FROM "+Type_dict[p]+" WHERE ID="+id
        else:
            id1=temp.split('-')[0]
            id2=temp.split('-')[1]
            delete_sql = "DELETE FROM " + Type_dict[p] + " WHERE ID>=" + id1+" AND ID<="+id2
    cursor.execute(delete_sql)
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
            # s=get_reply(message.as_persistent_string())
            # await app.send_message(group, MessageChain([Plain(s)]))
            reply=get_replys(message.as_persistent_string())
            msg = MessageChain([Plain(reply)])
            await app.send_message(group, msg)
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
        s_message=MessageChain([Plain("DeepTalk闲聊机器人使用说明：\n"+
                                      "自由聊天\n"+
                                      "自由增删改查内容\n"+
                                      "可能涉及到的消息发送格式：\n"+
                                      "页码：页码范围内的数字\n"+
                                      "查询关键词：key 关键词\n"+
                                      "修改：update id或id起始值-id终止值 字段名1 关键词1...\n"+
                                      "删除：del id或id起始值-id终止值\n"+
                                      "增加：add 数据1 数据2...\n"+
                                      "自定义：define 表名 字段名1 字段名2...\n"+
                                      "结束操作：x")])
        await app.send_message(group, s_message)

app.launch_blocking()