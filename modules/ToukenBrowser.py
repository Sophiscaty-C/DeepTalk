import requests
import json
import bs4
import urllib
from urllib import request
from bs4 import BeautifulSoup
import xlrd
import pickle


# def get_name(x):
#     data=xlrd.open_workbook("resources/刀男/touken.xlsx")
#     table=data.sheets()[0]
#     for i in range(table.nrows):
#         if str(table.cell_value(i, 0)) == x:
#             return table.cell_value(i, 2)


def crawl_data():
    dict = {}
    for i in range(1, 213):
        for j in range(1, 31):
            url_get_token = "http://touken.7moe.com/index.php?type=" + str(i) + "&page=" + str(j)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                     'Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(headers=headers, url=url_get_token, timeout=2)
            # bs = BeautifulSoup(response.text, "lxml")
            bs = BeautifulSoup(response.text, 'html.parser')
            for j in bs.find_all('table'):
                if j.has_attr('class') and 'table-striped' in j['class']:
                    count = 0
                    t = ""
                    for k in j.find_all('td'):
                        if count % 6 == 1:
                            t = k.contents[0]
                            if t not in dict:
                                dict[t] = []
                        elif count % 6 == 4:
                            dict[t].append((i, float(k.contents[0].split('%')[0]) * 0.01))
                        count += 1
            print(dict)
    json_str = json.dumps(dict)
    with open('test_data.json', 'w') as json_file:
        json_file.write(json_str)


crawl_data()
