import random
import requests
import json
import urllib
from urllib import request
from bs4 import BeautifulSoup
import xlrd

def random_num(num):
    l=[]
    temp=[]
    for i in range(num):
        l.append(0)
    c = 0
    while c != num:
        b1 = random.randint(0, num-1)
        if l[b1] == 0:
            temp.append(b1+1)
            c += 1
            l[b1] = 1
        else:
            continue
    return temp
def read_file(path):
    f = open(path, encoding = "utf-8")
    l = f.readlines()
    num = random.randint(0, len(l) - 1)
    return l[num].rstrip('\n')

def get_names():
    l=[]
    data=xlrd.open_workbook("resources/刀男/touken.xlsx")
    table=data.sheets()[0]
    col=table.col(2)
    for i in col:
        if i.value!='' and '极' not in i.value:
            l.append(i.value)
    return l
def crawl_image(l):
    for i in l:
        url_get_token = "https://zh.moegirl.org.cn/刀剑乱舞:"+i
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(headers=headers, url=url_get_token, timeout=2)
        bs=BeautifulSoup(response.text, "html.parser")
        for i in bs.find_all('img'):
            if i.has_attr('alt') and "图鉴" in i['alt']:
                print(i['alt'])
                req = urllib.request.Request(url=i['src'], headers=headers)
                response = urllib.request.urlopen(req)
                filename = i['alt']
                with open("resources/刀男/images/"+filename, "wb") as f:
                    content = response.read()  # 获得图片
                    f.write(content)  # 保存图片
                    response.close()

def get_name(x):
    data=xlrd.open_workbook("resources/刀男/touken.xlsx")
    table=data.sheets()[0]
    for i in range(1, table.nrows):
        if str(int(table.cell_value(i, 0))) == x:
            print(table.cell_value(i, 2))
            return table.cell_value(i, 2)

def crawl_data(i, d):
        for j in range(1, 31):
            url_get_token = "http://touken.7moe.com/index.php?type="+str(i)+"&page="+str(j)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(headers=headers, url=url_get_token, timeout=2)
            bs=BeautifulSoup(response.text, "lxml")
            for j in bs.find_all('table'):
                if j.has_attr('class') and 'table-striped' in j['class']:
                    count=0
                    t=""
                    for k in j.find_all('td'):
                        if count%6==1:
                            t=k.contents[0]
                            if t not in d:
                                d[t]=[]
                        elif count%6==4:
                            x=round(float(k.contents[0].split('%')[0])*0.01, 3)
                            d[t].append((i, x))
                        count+=1
        return d

if __name__=='__main':
    for i in range(1, 213):
        print(i)
        with open('resources/刀男/test_data.json', 'r', encoding='utf8') as json_file:
            json_data = json.load(json_file)
        d=crawl_data(i, json_data)
        json_str = json.dumps(d)
        print(json_str)
        with open('resources/刀男/test_data.json', 'w', encoding='utf8') as json_file:
            json_file.write(json_str)