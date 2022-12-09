import pandas as pd
import jieba.posseg as pseg
import jieba
import fool
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 加载停用词词典
stopwords = {}
with open('./data/stopword.txt', 'r', encoding='utf-8') as fr:
    for word in fr:
        stopwords[word.strip()] = 0


# 定义类
class clf_model:
    """
    该类将所有模型训练、预测、数据预处理、意图识别的函数包括其中
    """

    # 初始化模块
    def __init__(self):
        self.model = ""  # 成员变量，用于存储模型
        self.vectorizer = ""  # 成员变量，用于存储tfidf统计值

    # 训练模块
    def train(self):
        # 从excel文件读取训练样本
        d_train = pd.read_excel('./data/smart_storage_data_train.xlsx')
        # 对训练数据进行预处理
        d_train.sentence_train = d_train.sentence_train.apply(self.preprocess)
        train_data = d_train[]
        print("训练样本 = %d" % len(d_train))
        # 利用sklearn中的函数进行tfidf训练
        self.vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
        features = self.vectorizer.fit_transform(d_train.sentence_train)
        print("训练样本特征表长度为 " + str(features.shape))
        # 使用逻辑回归进行训练和预测
        self.model = LogisticRegression(C=10)
        self.model.fit(features, d_train.label)

    # 预测模块（使用模型预测）
    def predict_model(self, sentence):
        """
        :param sentence: 用户输入语句预处理后的结果
        :return: 意图类别，分数
        """
        sent_features = self.vectorizer.transform([sentence])
        pre_test = self.model.predict_proba(sent_features).tolist()[0]
        clf_result = pre_test.index(max(pre_test))
        score = max(pre_test)
        return clf_result, score

    # 预测模块（使用规则）
    def predict_rule(self, sentence):
        """
        :param sentence: 用户输入语句预处理后的结果
        :return: 意图类别，分数
        """
        sentence = sentence.replace(' ', '')
        words = [w.word if w.flag == 'v' else None for w in pseg.cut(sentence)]
        # words = [item[0] if item[1] == 'v' else None for item in fool.pos_cut(sentence)[0]]
        words = list(filter(lambda x : x is not None, [i[0] if i[1] == 'v' else None for i in words]))
        save_score = self.save_rule(sentence, words)
        search_score = self.search_rule(sentence, words)
        if not save_score:
            return 2, save_score
        elif not search_score:
            return 1, search_score
        else:
            return 0, 0.8

    # "搜索"意图判断规则
    def search_rule(self, sentence, words):
        condition1 = 1 if re.findall(r'查|看|找|', sentence) else 0
        condition2 = 1 if re.findall(r'行程|日程|计划|数据', sentence) else 0
        condition3 = 0
        for item in ["查", "找", "查找", "查看"]:
            if item in words:
                condition3 = 1
                break
        if condition1 and condition2 and condition3:
            return 0.8
        elif not condition1 and not condition3:
            return 0
        else:
            return 0.6

    # "存储"意图判断规则
    def save_rule(self, sentence, words):
        condition1 = 1 if re.findall(r'存|备份|保留|', sentence) else 0
        condition2 = 1 if re.findall(r'行程|日程|计划|数据', sentence) else 0
        condition3 = 0
        for item in ["保存", "存", "存储", "储存"]:
            if item in words:
                condition3 = 1
                break
        if condition1 and condition2 and condition3:
            return 0.8
        elif not condition1 and not condition3:
            return 0
        else:
            return 0.6

    # 预处理函数
    def preprocess(self, sentence):
        """
        :param sentence: 用户输入语句
        :return: 预处理结果
        """
        # 分词，并去除停用词
        words = [w.word for w in pseg.cut(sentence)]
        # words = [item[0] for item in fool.pos_cut(sentence)[0]]
        word_lst = [w for w in words if w not in stopwords]
        output_str = ' '.join(word_lst)
        output_str = re.sub(r'\s+', ' ', output_str)
        return output_str.strip()

    # 分类主函数
    def fun_clf(self, sentence):
        """
        :param sentence: 用户输入语句
        :return: 意图类别，分数
        """
        # 对用户输入进行预处理
        sentence = self.preprocess(sentence)
        # 得到意图分类结果（0为“其他”类别，1为“查询”类别，2为“存储”类别）
        clf_result, score = self.predict_model(sentence)  # 使用训练的模型进行意图预测
        # clf_result, score = self.predict_rule(sentence)  # 使用规则进行意图预测（可与用模型进行意图识别的方法二选一）
        return clf_result, score


def verify():
    confirm_s = input("请问是要小p帮你存数据或者查找以前存的数据吗\n")
    if re.findall(r'没有|不|NO|no|No', confirm_s):
        return 0
    else:
        return 1


def main(input_sentence):
    """
    :param input_sentence: 用户输入语句
    :return: 0 非”存储数据“或”查询已存数据“, 1 需要”查询已存数据“, 2: 需要”存储数据“;
    """
    clf_obj = clf_model()
    clf_obj.train()
    passing_threshold = 0.55  # 用户定义阈值（当分类器分类的分数大于阈值才采纳本次意图分类结果，目的是排除分数过低的意图分类结果）
    verify_threshold = 0.75
    verify_tag = True

    clf_result, score = clf_obj.fun_clf(input_sentence)
    # 2: 存储
    # 1: 查询
    # 0: 其他
    # 用户输入未达到“查询已存数据”、“存储新数据”类别的阈值 OR 被分类为“终止服务”
    if score < passing_threshold or clf_result == 0:
        return 0
    elif passing_threshold <= score < verify_threshold:
        if verify_tag:
            v = verify()
        else:
            v = 1
    else:
        v = 1
    # 用户输入分类为“查询已存数据” OR “存储新数据”
    if v:
        return clf_result
    else:
        return 0
