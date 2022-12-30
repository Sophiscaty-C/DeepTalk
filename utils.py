from graia.ariadne.connection.config import config, WebsocketClientConfig
from graia.ariadne.event.message import *
from graia.ariadne.message.element import Plain, Image, At, Source, App, File, Element, Voice
import json
import requests
import pytz
from datetime import *
from smart_storage import main, storagedata_type_update, operation_clf_model
from implementation import *

'''
Choose Bot IDE
'''
def load_config(config_file: str = "./config.json") -> dict:
    with open(config_file, 'r', encoding='utf-8') as f:  # 从json读配置
        config = json.loads(f.read())
    for key in config.keys():
        config[key] = config[key].strip() if isinstance(config[key], str) else config[key]
    return config

def get_replys(msg):
    config = load_config()
    print(config)
    bot = config['bot']
    if bot == 'qingyunke':
        return get_qingyunke_reply(msg)
    elif bot == 'ruyi':
        return get_ruyi_reply(msg)
    elif bot == 'tuling':
        return get_tuling_reply(msg)


# 青云客机器人，https://api.qingyunke.com
def get_qingyunke_reply(msg):
    reply = ''
    img_path = None
    voice_path = None

    key = config['qingyunke']['key']
    url= f'http://api.qingyunke.com/api.php?key={ key }&appid=0&msg={ msg }'
    response=requests.get(url)
    return response.json()["content"]


# 如意机器人，https://ruyi.ai/
def get_ruyi_reply(msg):
    reply = ''
    img_path = None
    voice_path = None

    # appKey = config['ruyi']['appKey']
    # userID = config['ruyi']['userID']
    # url = f"http://api.ruyi.ai/v1/message?q={ msg }&app_key={ appKey }&user_id={ userID }"
    # replys = []
    # async with ClientSession() as session:
    #     async with session.get(url) as response:
    #         content = await response.json()
    #         if content['code'] == 0:
    #             replys = content['result']['intents'][0]['outputs']
    # for r in replys:
    #     if r['type'] == 'dialog':
    #         reply = r['property']['text']
    
    return ' ' + reply, img_path, voice_path


# 图灵机器人，http://www.tuling123.com/
def get_tuling_reply(msg):
    reply = ' '
    # apiKey = config['tuling']['apiKey']
    # userID = config['tuling']['userID']
    # url = 'http://openapi.tuling123.com/openapi/api/v2'
    # data = {
    #     "reqType":0,
    #     "perception": {
    #         "inputText": {
    #             "text": msg
    #         }
    #     },
    #     "userInfo": {
    #         "apiKey": apiKey,
    #         "userId": userID
    #     }
    # }
    # async with ClientSession() as session:
    #     async with session.post(url, json = data) as response:
    #         content = await response.read()
    #         print(content)
    #         results = json.loads(content.decode('utf-8'))
    # if results['intent']['code'] == 4003:
    #     reply += '我今天不能和你聊天啦~'
    # else:
    #     for res in results['results']:
    #         reply += res['values']['text']
    return reply, None, None

'''
Tool Functions
'''
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
    time = datetime.datetime(time_list[0], time_list[1], time_list[2], time_list[3], time_list[4], time_list[5]).replace(tzinfo=pytz.utc)
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
    return predict(s)