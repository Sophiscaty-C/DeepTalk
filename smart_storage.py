#! /usr/bin/env python   
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import json
from typing import Union, List, Dict, Tuple
import re
import pickle
import datetime
from datetime import datetime as dt
import jieba
import jieba.posseg as pseg
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from config import getConfig

gConfig = {}
gConfig = getConfig.get_config()

# 加载数据库数据类型字典
storagedata_type_path = gConfig['storagedata_type_path']
if os.path.isfile(storagedata_type_path):
    with open(storagedata_type_path, 'r', encoding='utf-8') as f:
        Storagedata_type = json.load(f)
else:
    with open(storagedata_type_path, 'w', encoding='utf-8') as f:
        Storagedata_type = {0: ['默认'], 
                            1: ['日程', '计划', '待办', '行程', '课表', '排班', '课程表'],
                            2: ['链接', '网址', '网站'],
                            3: ['文件', '图片', '视频', '音频', '文档', 'pdf', 'word']}
        json.dump(Storagedata_type, f, ensure_ascii=False, indent=2)

# 加载停用词词典
stopwords_path = gConfig['stopwords_path']
stopwords = {}
with open(stopwords_path, 'r', encoding='utf-8') as fr:
    for word in fr:
        stopwords[word.strip()] = 0


# 定义“操作分类模型”类
class operation_clf_model:
    """
    该类将所有模型训练、预测、数据预处理、意图识别的函数包括其中
    """
    # 初始化模块
    def __init__(self):
        self.model = ""  # 成员变量，用于存储模型
        self.vectorizer = ""  # 成员变量，用于存储tfdif统计值

    # 训练模块
    def train(self):
        # 从excel文件读取训练样本
        d_train = pd.read_excel(gConfig['operation_clf_model_data_path'])
        # 对训练数据进行预处理
        train_data = []
        for s in d_train.sentence.values:
            pro_s = self.preprocess([w.word for w in pseg.cut(s)])
            train_data.append(pro_s)
        print("训练样本 = %d" % len(d_train))
        # 利用sklearn中的函数进行tfidf训练
        self.vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
        features = self.vectorizer.fit_transform(train_data)
        print("训练样本特征表长度为 " + str(features.shape))
        # 使用逻辑回归进行训练和预测
        self.model = LogisticRegression(C=10)
        self.model.fit(features, d_train.label)

    # 预测模块（使用模型预测）
    def predict_model(self, cooked_sentence):
        """
        :param sentence: 用户输入语句预处理后的结果
        :return: 意图类别，分数
        """
        sent_features = self.vectorizer.transform([cooked_sentence])
        pre_test = self.model.predict_proba(sent_features).tolist()[0]
        clf_result = pre_test.index(max(pre_test))
        score = max(pre_test)
        # print(pre_test)
        return clf_result, score

    # 预测模块（使用规则）
    def predict_rule(self, cooked_sentence, words_v_n):
        """
        :param sentence: 用户输入语句预处理后的结果
        :return: 意图类别，分数
        """
        cooked_sentence = cooked_sentence.replace(' ', '')
        save_score = self.save_rule(cooked_sentence, words_v_n)
        search_score = self.search_rule(cooked_sentence, words_v_n)
        # print(save_score, search_score)
        if save_score > search_score:
            return 2, save_score
        elif search_score > save_score:
            return 1, search_score
        else:
            if search_score == 0:
                return 0, 0.8
            else:
                return

    # "搜索"意图判断规则
    def search_rule(self, cooked_sentence, words_v_n):
        condition1 = 1 if re.findall(r'查|看|找|搜索', cooked_sentence) else 0
        condition2 = 1 if re.findall(r'行程|日程|计划|数据|待办|课表', cooked_sentence) else 0
        condition3 = 0
        for item in ["查", "找", "看看","查找", "查看", "查询", "搜索", "搜寻", "寻找"]:
            if item in words_v_n:
                condition3 = 1
                break
        if condition1 and condition2 and condition3:
            return 0.8
        elif not condition1 and not condition3:
            return 0
        else:
            return 0.6

    # "存储"意图判断规则
    def save_rule(self, cooked_sentence, words_v_n):
        condition1 = 1 if re.findall(r'存|备份|保留|收藏', cooked_sentence) else 0
        condition2 = 1 if re.findall(r'行程|日程|计划|数据|待办|课表', cooked_sentence) else 0
        condition3 = 0
        for item in ["保存", "存", "存储", "储存", "收藏"]:
            if item in words_v_n:
                condition3 = 1
                break
        if condition1 and condition2 and condition3:
            return 0.8
        elif not condition1 and not condition3:
            return 0
        else:
            return 0.6

    # 预处理函数
    def preprocess(self, raw_sentence_words: List):
        """
        :param sentence: 用户输入语句
        :return: 预处理结果
        """
        # 分词，并去除停用词
        word_lst = [w for w in raw_sentence_words if w not in stopwords]
        output_str = ' '.join(word_lst)
        output_str = re.sub(r'\s+', ' ', output_str)
        return output_str.strip()

    # 分类主函数
    def clf_main(self, raw_sentence, words_v_n):
        """
        :param sentence: 用户输入语句
        :return: 意图类别，分数
        """
        # 对用户输入进行预处理
        cooked_sentence = self.preprocess(raw_sentence)
        # 得到意图分类结果（0为“其他”类别, 1为“查询”类别, 2为“存储”类别, 3为“改动”类别, 4为“删除”类别 
        if re.findall(r'找|查|看|搜索|有没有|存|备份|收藏|保留|删', raw_sentence):
            clf_result, score = self.predict_model(cooked_sentence)  # 使用训练的模型进行操作意图预测
        else:
            clf_result, score = 0, 0.8
        # clf_result, score = self.predict_rule(cooked_sentence, words_v_n)  # 使用规则进行操作意图预测
        return clf_result, score


# 定义“待搜索数据类别分类模型”类
# class type_clf_model:
#     def __init__(self):
#         self.model = ""  # 成员变量，用于存储模型
#         self.vectorizer = ""  # 成员变量，用于存储tfdif统计值

#     # 训练模块
#     def train(self):
#         # 从excel文件读取训练样本
#         d_train = pd.read_excel('./chatbox_data/smart_storage/type_data_train.xlsx')
#         # 对训练数据进行预处理
#         train_data = []
#         for s in d_train.sentence.values:
#             pro_s = self.preprocess([w.word for w in pseg.cut(s)])
#             train_data.append(pro_s)
#         print("训练样本 = %d" % len(d_train))
#         # 利用sklearn中的函数进行tfidf训练
#         self.vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
#         features = self.vectorizer.fit_transform(train_data)
#         print("训练样本特征表长度为 " + str(features.shape))
#         # 使用逻辑回归进行训练和预测
#         self.model = LogisticRegression(C=10)
#         self.model.fit(features, d_train.label)

#     # 预测模块（使用模型预测）
#     def predict_model(self, sentence):
#         """
#         :param sentence: 用户输入语句预处理后的结果
#         :return: 意图类别，分数
#         """
#         sent_features = self.vectorizer.transform([sentence])
#         pre_test = self.model.predict_proba(sent_features).tolist()[0]
#         clf_result = pre_test.index(max(pre_test))
#         score = max(pre_test)
#         return clf_result, score


# 存储/搜索数据类型确定
def type_identify(raw_sentence: str, words_n: List) -> int:
        """
        :param sentence: 用户原输入语句
        :return: 
        """
        type_prob = [0] * len(Storagedata_type)
        for j, (type, values) in enumerate(Storagedata_type.items()):
            if type == 0:
                continue
            for i, sub_type in enumerate(values):
                if re.findall(sub_type, raw_sentence) or sub_type in words_n:
                    type_prob[j] = 1
        non_zero = np.count_nonzero(type_prob)
        if non_zero == 1:
            for i, prob in enumerate(type_prob):
                if prob:
                    return i
        else:
            return 0
        

# 待搜索数据的可能存储时间/最后一次修改时间的确定
def search_past_time(raw_sentence: str, now_date: dt) ->  List:
        """
        :param sentence: 用户原输入语句
        :return: 用户想要搜索的文件的可能存储或者最后修改时间的列表
        """
        week_list = ["一", "二", "三", "四", "五", "六", "日", "周天"]
        special_date1 = ["昨天", "前天", "大前天"]
        special_date2 = ["上周末", "上个周末"]

        if not now_date:
            now_date = datetime.datetime.now()   # 这里可以换成系统获取到的用户此时发消息时当地的时间，相应的后面获取日子的方式也要改
        now_date_list = now_date.strftime('%Y-%m-%d').split('-')
        now_weekday_idx = datetime.date(int(now_date_list[0]), int(now_date_list[1]), int(now_date_list[2])).weekday()

        search_undetermined_date = list(set(re.findall(r'\d{,2}月\d{,2}号|\d{,4}月\d{,2}日', raw_sentence)))
        search_date = []
        for date in search_undetermined_date:
            num = re.findall("\d+", date)
            if int(now_date_list[1]) > int(num[0]) or (int(now_date_list[1]) == int(num[0]) and int(now_date_list[2]) > int(num[1])):
                year = now_date_list[0]
                month = '0' + num[0] if len(num[0]) == 1 else num[0]
                day = '0' + num[1] if len(num[1]) == 1 else num[1]
                search_date.append(year + '-' + month + '-' + day)

        search_common_date = re.findall(r'周一|周二|周三|周四|周五|周六|周日|周天'
                                        r'星期一|星期二|星期三|星期四|星期五|星期六|星期日|星期天', raw_sentence)
        search_undetermined_date = re.findall(r'本周一|本周二|本周三|本周四|本周五|本周六|本周日|本周天'
                                                r'这周一|这周二|这周三|这周四|这周五|这周六|这周日|这周天'
                                                r'这星期一|这星期二|这星期三|这星期四|这星期五|这星期六|这星期日|这星期天'
                                                r'这个周一|这个周二|这个周三|这个周四|这个周五|这个周六|这个周日|这个周天'
                                                r'这个星期一|这个星期二|这个星期三|这个星期四|这个星期五|这个星期六|这个星期日|这个星期天', raw_sentence)
        search_determined_special_date = re.findall(r'大前天|前天|昨天', raw_sentence)
        search_determined_common_date = re.findall(r'上周一|上周二|上周三|上周四|上周五|上周六|上周日|上周天|上周末'
                                                    r'上星期一|上星期二|上星期三|上星期四|上星期五|上星期六|上星期日|上星期天'
                                                    r'上个周一|上个周二|上个周三|上个周四|上个周五|上个周六|上个周日|上个周天|上个周末'
                                                    r'上个星期一|上个星期二|上个星期三|上个星期四|上个星期五|上个星期六|上个星期日|上个星期天', raw_sentence)

        common_date_dict = {}
        for date in search_common_date:
            if date in common_date_dict.keys():
                common_date_dict[date] += 1
            else:
                common_date_dict[date] = 1
        common_date_list = sorted(common_date_dict.items(), key = lambda x:x[1], reverse = True)
        diff = len(search_common_date) - len(search_undetermined_date + search_determined_common_date)
        assert len(common_date_dict) >= diff
        search_common_date = [item[0] for item in common_date_list[:diff]]

        last_date = list(set(search_determined_special_date)) + list(set(search_determined_common_date))
        this_date = list(set(search_undetermined_date)) + list(set(search_common_date))
        search_candidate_date = last_date + this_date
        last_tag_list = [1] * len(last_date) + [0] * len(this_date)

        # print(search_candidate_date)
        for j, date in enumerate(search_candidate_date):
            search_weekday_idx = -1
            last_tag = last_tag_list[j]
            process_tag = 0
            double_tag = 0
            for i, prefix in enumerate(week_list):
                if prefix in date:
                    search_weekday_idx = i
                    break
            
            if search_weekday_idx == -1:
                for i, item in enumerate(special_date1):
                    if item == date:
                        search_date.append((now_date + datetime.timedelta(days=-(i+1))).strftime("%Y-%m-%d"))
                        process_tag = 1
                        break
                for i, item in enumerate(special_date2):
                    if item == date:
                        search_weekday_idx = 6
                        double_tag = 1

            if process_tag:
                continue

            if search_weekday_idx == 7:
                search_weekday_idx -= 1

            if search_weekday_idx < now_weekday_idx or last_tag:
                search_date.append((now_date + datetime.timedelta(days=search_weekday_idx - now_weekday_idx - last_tag * 7)).strftime("%Y-%m-%d"))
                if double_tag:
                    search_date.append((now_date + datetime.timedelta(days=search_weekday_idx - now_weekday_idx - last_tag * 7 - 1)).strftime("%Y-%m-%d"))

        return search_date


# 词性分析并过滤
def filter_sentence(raw_sentence: str):
    total_cut_words = [(w.word, w.flag) for w in pseg.cut(raw_sentence)]

    words_n = [item[0] if item[1] == 'n' else None for item in total_cut_words]
    words_n = list(filter(lambda x : x is not None, words_n))

    words_v_n = [item[0] if item[1] == 'v' or 'n' else None for item in total_cut_words]
    words_v_n = list(filter(lambda x : x is not None, words_v_n))

    words_nx = [item[0] if item[1].startswith('n') else None for item in total_cut_words]
    words_nx = list(filter(lambda x : x is not None, words_nx))

    return total_cut_words, words_n, words_v_n, words_nx


# 基于规则的操作类型再分类
def verify(raw_sentence: str, words: List[Tuple]) -> int:
    search_indicator = re.findall(r'查|找|看', raw_sentence)
    save_indicator = re.findall(r'存|备份|保留|收藏', raw_sentence)
    delete_indicator = re.findall(r'删', raw_sentence)
    # if search_indicator and not save_indicator and not delete_indicator:
    #     return 1
    # elif not search_indicator and save_indicator and not delete_indicator:
    #     return 2
    # elif not search_indicator and not save_indicator and delete_indicator:
    #     return 3
    # else:
    search_tag = 0
    save_tag = 0
    delete_tag = 0
    windows = 2
    for i, w in enumerate(words):
        if re.findall(r'查|找|看', w[0]) and w[1]  == 'v':
            tag = 0
            for j in range(1, windows+1):
                if i + j < len(words):
                    tag = 1 if words[i+j][1].startswith('u') else 0  
                    if tag:
                        break
            if not tag:
                search_tag += 1
            
        elif re.findall(r'存|备份|保留|收藏', w[0]) and w[1]  == 'v':
            tag = 0
            for j in range(1, windows+1):
                if i + j < len(words):
                    tag = 1 if words[i+j][1].startswith('u') else 0
                    if tag:
                        break
            if not tag:
                save_tag += 1

        elif re.findall(r'删', w[0]) and w[1]  == 'v':
            tag = 0
            for j in range(1, windows+1):
                if i + j < len(words):
                    tag = 1 if words[i+j][1].startswith('u') else 0
                    if tag:
                        break
            if not tag:
                delete_tag += 1
        
    tag_list = [search_tag, save_tag, delete_tag]
    max_tag = max(tag_list)
    max_tag_index = []
    for i, v in enumerate(tag_list):
        if max_tag == v:
            max_tag_index.append(i + 1)
    if len(max_tag_index) == 1:
        return max_tag_index[0]
    else:
        return 0


# 存储/搜索数据类型文件更新
def storagedata_type_update(type_name: str):
    new_idx = len(Storagedata_type)
    Storagedata_type[new_idx] = [type_name]
    with open(storagedata_type_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(Storagedata_type))


# 搜索数据的关键词提取
def keywords_extract(raw_sentence: str, data_type: int, words_nx: List, Storagedata_type: Dict) -> List:

    keywords = []
    analyzer = TextRank4Keyword()
    analyzer.analyze(text=raw_sentence, lower=True, window=3)
    for item in analyzer.get_keywords(10, word_min_len=1):
        if item.word in words_nx and item.word not in list(Storagedata_type.values())[data_type]:
            keywords.append(item.word)
    
    return keywords


def train_save(model: Union[operation_clf_model] , clf_model_path) -> Union[operation_clf_model]:
    
    if os.path.exists(clf_model_path):
        with open(clf_model_path, 'rb') as file:
            model = pickle.loads(file.read())
    else:
        model.train()
        save_model = open(clf_model_path, 'wb')
        save_model.write(pickle.dumps(model))
        save_model.close()
    return model


def main(input_sentence, now_time: dt):
    """
    :param input_sentence: 用户输入语句
    :return: 执行操作: 0 其他, 1 需要“查询已存数据”, 2: 需要“存储数据”, 3: 需要“删除数据”
             数据类型: 0 默认, 1 日程表, 2 链接, 3 文件, 4 用户自定义的数据类型1, ...
             时间列表(只有在搜索时用到)
             关键词列表(只有在搜索时用到)
    """
    operation_clf_model_path = gConfig['operation_clf_model_path']
    operation_clf_obj = operation_clf_model()
    operation_clf_obj = train_save(operation_clf_obj, operation_clf_model_path)
    
    passing_threshold = 0.4
    verify_threshold = 0.55
    verify_tag = True

    # 词性分析并过滤
    total_words, words_n, words_v_n, words_nx = filter_sentence(input_sentence)

    clf_result, score = operation_clf_obj.clf_main(input_sentence, words_v_n)
    if score < passing_threshold or clf_result == 0:
        v = 0
    elif passing_threshold <= score < verify_threshold:
        if verify_tag:
            v = verify(input_sentence, total_words)
        else:
            v = clf_result
    else:
        v = clf_result

    print("最终预测结果: {} 分类结果: {} 分类分数: {}".format(v, clf_result, score))
    
    # 判断储存或者搜索的数据类型
    data_type = 0
    if v != 0:
        data_type = type_identify(input_sentence, words_n)

    # 获取搜索数据的 可能创建时间或者最后一次修改的时间
    search_date_list = []
    if v == 1:
        search_date_list = search_past_time(input_sentence, now_time)

    # 获取 其他关键词
    keywords = []
    if v == 1 or v == 3:
        keywords = keywords_extract(input_sentence, data_type, words_nx, Storagedata_type)

    print(input_sentence)
    print(['其他', '搜索', '存储', '删除'][v], [item[0] for item in Storagedata_type.values()][data_type], search_date_list, keywords)
    return v, data_type, search_date_list, keywords


# s1 = "帮我找个文件"
# s2 = "帮我删掉昨天存的文件"
# s3 = "帮我把后天上午的日程改成开会"
# s4 = "你这么做是存心的吧"
# s5 = "存个明天的日程"
# main(s1, None)
# main(s2, None)
# main(s3, None)
# main(s4, None)
# main(s5, None)
