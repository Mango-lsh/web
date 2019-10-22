import numpy as np
import re
import pandas as pd
import nltk
from collections import Counter
import wordcloud  # 词云展示库
from PIL import Image  # 图像处理库
import matplotlib.pyplot as plt  # 图像展示库



global ENG_PUNCTUATIONS
ENG_PUNCTUATIONS = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&',
                    '!', '*', '@', '#', '$', '%', '’', '‘', '-', '\n',
                    '\r', '""', '``', "''", '”', '“', '...', "'"]


# 词语类型
def wordTypes(x):
    b = nltk.word_tokenize(x)
    a = [word for word in b if not word in ENG_PUNCTUATIONS]
    return len(set(a))


# 词汇密度
def getLD(content):
    count = 0
    li = []
    for i in nltk.pos_tag(content):
        li.append(i[1])
    L = ['ADJ', ' ADV', 'N', 'NP', 'NN', 'V', 'VD', 'VG', 'VN']
    for j in li:
        if j in L:
            count += 1
    return count


class NaLiPao:

    def __init__(self, df):
        self.df = df
        self.total_doc = 0
        self.total_word = 0
        self.total_type = 0
        self.total_counter = Counter('')
        self.total_dict = {}
        self.total_term = 0

    def process(self):
        self.total_doc = len(self.df['fileName'])
        self.total_word = sum(self.df['length'])
        self.unique_word = wordTypes(''.join(self.df['fileContent']))
        self.total_counter = self.df['counter'][0]
        for i in range(1, self.total_doc):
            self.total_counter += self.df['counter'][i]
        self.total_dict = dict(self.total_counter.most_common(len(self.total_counter)))
        self.total_term = self.getTerms()
        self.df['ratio'] = round(self.df['wordType'] / self.df['length'], 2)

    # 词频前n的词语
    def countWords(self, which=0, n=10):
        print(self.df['counter'][which].most_common(n))




    # terms相对频数
    def getTerms(self):
        term_dict = {}
        term_dict['ID'] = list(range(1, len(self.total_counter) + 1))
        term_dict['Term'] = [key for key in self.total_dict]
        term_dict['Count'] = [self.total_dict[key] for key in self.total_dict]
        for i in range(self.total_doc):
            term_dict[self.df['fileName'][i]] = []
            for j in term_dict['Term']:
                term_dict[self.df['fileName'][i]].append(self.df['counter'][i][j] / self.df['length'][i])
        term_df = pd.DataFrame.from_dict(term_dict)
        return term_df



    # 文章总体总结
    def summary(self):
        s = []
        for i in range(13):
            s.append('')

        s[0] += ('This corpus has %d documents with %d total words and %d unique word forms.' % (
            self.total_doc, self.total_word, self.total_type))

        s[1] += ('Document Length:\n')
        s[2] += ('·Longest :  ')
        s[2] += (self.sortType('length', False))
        s[3] += ('·Shortest :  ')
        s[3] += (self.sortType('length', True))

        s[4] += ('Lexical Density:\n')
        s[5] += ('·Highest :  ')
        s[5] += (self.sortType('LD', False))
        s[6] += ('·Lowest :  ')
        s[6] += (self.sortType('LD', True))

        s[7] += ('Average Words Per Sentence:\n')
        s[8] += ('·Longest :  ')
        s[8] += (self.sortType('perLength', False))
        s[9] += ('·Shortest :  ')
        s[9] += (self.sortType('perLength', True))

        s[10] += ('Most frequent words in the corpus:\n')
        tc = dict(self.total_counter.most_common(5))
        for i in tc:
            s[11] += ('%s' % i)
            s[11] += ('(%d);' % tc[i])

        s[12] += ('Distinctive words (compared to the rest of the corpus):')

        dw = []
        dw += (self.getDW())
        return s ,dw

    # 按所给属性排序
    def sortType(self, Type, ascend):
        df = self.df.sort_values(Type, ascending=ascend)
        count = 0
        s = ''
        for i in range(len(df[Type])):
            s += ('%s' % (df.iloc[i, :]['fileName']))
            if Type == 'length':
                s += ('(%d); ' % (df.iloc[i, :][Type]))
            elif Type == 'LD':
                s += ('(%.3f); ' % (df.iloc[i, :][Type]))
            elif Type == 'perLength':
                s += ('(%.1f); ' % (df.iloc[i, :][Type]))
            else:
                s += ('(%d); ' % (df.iloc[i, :][Type]))
            count += 1
            if count == 5:
                break
        s += '\n'
        return s

    # 寻找区别词
    # 基本方法为除去在总计数其中前n的词语后，各文档所用的前五个词（默认去除前100）
    def getDW(self):
        ss = []
        d = []
        dd = dict(self.total_counter.most_common(100))
        for i in dd:
            d.append(i)
        for k in range(self.total_doc):
            s = ''
            name = self.df['fileName'][k]
            dw = {}
            c = dict(self.df['counter'][k].most_common(10000))
            # print (c)
            count = 0
            for l in c:
                if l not in d:
                    count += 1
                    dw[l] = c[l]
                    if count == 5:
                        break
            s += '%d.' % (k + 1)
            s += ('%s' % name + ':')
            first = 1
            for m in dw:
                if first:
                    s += ('%s' % m + '(%d)' % dw[m])
                    first = 0
                else:
                    s += (',' + '%s' % m + '(%d)' % dw[m])
            s += '.'
            ss.append(s)

        return ss



#根据文档返回不同
    # 每个文档的基本情况
    def getDocuments(self):
        df_doc = self.df[['fileName', 'length', 'wordType', 'ratio', 'perLength']]
        df_doc =  df_doc.rename(
            columns={'fileName': 'Title', 'length': 'Words', 'wordType': 'Types', 'perLength': 'Words/Sentence'})
        return df_doc


    # 词云
    def getCloud(self, which=0,word_num=200):
        # mask = np.array(Image.open('wordcloud.jpg')) # 定义词频背景
        wc = wordcloud.WordCloud(
            font_path='/System/Library/Fonts/STHeiti Medium.ttc',  # 设置字体格式
            width=500,
            height=300,
            background_color='white',  # 设置背景图
            max_words= word_num,  # 最多显示词数
            max_font_size=100  # 字体最大值
        )

        wc.generate_from_frequencies(self.df['counter'][which])  # 从字典生成词云
        # image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
        # wc.recolor(color_func=image_colors) # 将词云颜色设置为背景图方案
        plt.imshow(wc)  # 显示词云
        plt.axis('off')  # 关闭坐标轴
        wc.to_file("/Users/mangguoshu/Documents/pycharm/web2/static/wordcloud.jpg")

    # 柱状图
    def getHistogram(self, which, n=50):
        c = dict(self.df['counter'][which])
        x = {i: c[i] for i in c if c[i] > n}
        fig, ax = plt.subplots()
        positions = [i for i in x]
        heights = [x[i] for i in x]
        ax.bar(positions, heights, 0.5)
        ax.set_xticklabels(positions,  rotation=45)
        fig.set_facecolor('none')
        plt.savefig('/Users/mangguoshu/Documents/pycharm/web2/static/histogram.jpg',bbox_inches = 'tight', transparent = True)




#根据词语展示不同

    # 找到关键词所在的句子
    def context(term, self):
        sentences = []
        filenames = []
        terms = []
        for i in range(self.total_doc):
            sentence = re.findall(r'[^,.;:\?\!\n\r\"]*?{}[^,.;:\?\!\n\r\"]*?[,.;:\!\?\n\r\"]'.format(term),
                                  self.df['fileContent'][i], re.IGNORECASE)
            filenames += [str(i + 1) + ')' + self.df['fileName'][i]] * len(sentence)
            sentences += sentence
        terms = [term] * len(sentences)
        data = {'Documents': filenames, 'Term': terms, 'Sentence': sentences}
        df_c = pd.DataFrame(data)
        return df_c


    # 根据所选单词返回对应相对频数折线图(最好不超过五个单词)
    def lineChart(self, terms):
        names = list(self.df['fileName'])
        names_pr = []
        for n in names:
            x1 = n.index(' ')
            x2 = n.index(' ', x1 + 1)
            names_pr.append(n[:x2] + '...')

        lineTypes = ['bD-', 'r^-', 'go-', 'y*-', 'cs-']
        x = range(len(names_pr))
        rf = {}
        types = 0
        for i in terms:
            try:
                rf[i] = list(self.total_term.loc[self.total_term['Term'] == i].iloc[0][names])
            except:
                print('输入单词%s有误' % i)
                return
            plt.plot(x, rf[i], lineTypes[types], label=i)
            if types < 5:
                types += 1
        plt.legend()  # 让图例生效
        plt.xticks(x, names_pr, rotation=45)

        # plt.margins(0)
        # plt.subplots_adjust(bottom=0.10)
        plt.xlabel('Name')  # X轴标签
        plt.ylabel("Relative Frequencies")  # Y轴标签
        plt.locator_params('y', nbins=15)
        plt.show()