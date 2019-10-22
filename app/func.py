import jieba
import os
import re
import os.path
import codecs
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import chardet
from collections import Counter
import pyocr
import importlib
import sys
#import docx
from app import nlp
from django.shortcuts import HttpResponse,render,redirect
importlib.reload(sys)


#pdf处理
'''
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
'''


global  ENG_PUNCTUATIONS
ENG_PUNCTUATIONS = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&',
                            '!', '*', '@', '#', '$', '%', '’', '‘', '-','\n',
                            '\r','""','``',"''",'”','“','...',"'"]

# 读取文件
def loadFile():
    filePaths = []
    fileNames = []
    fileContents = []

    for root, dirs, files in os.walk("/Users/mangguoshu/Documents/pycharm/web2/filedata"):
        for name in files:
            if name[0] == '.':
                continue
            filePath = os.path.join(root, name)
            filePaths.append(filePath)
            fileName = name[0:name.rfind('.', 1)]
            fileNames.append(fileName)

            if filePath.endswith('.txt'):
                f = open(filePath, 'rb')
                r = f.read()
                # 获取文本的编码方式
                f_charInfo = chardet.detect(r)
                fileContent = r.decode(f_charInfo['encoding'])  # 通过取得的文本格式读取txt
                f.close()
                fileContents.append(fileContent)



    corpus = pd.DataFrame({
        'fileName': fileNames,
        'filePath': filePaths,
        'fileContent': fileContents
    })
    return corpus


# 处理dataframe
def processDF(input_file):


    # 句子数
    def countSentence(sentence):
        alpha = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        ALPHA = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        A_S = alpha + ALPHA

        count = 0
        sentList = sentence.split('.')
        for s in sentList:
            for i in range(len(s)):
                if s[i] in A_S:
                    count += 1
                    break
        return count

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

    # 词语类型
    def wordTypes(x):
        b = nltk.word_tokenize(x)
        a = [word for word in b if not word in ENG_PUNCTUATIONS]
        return len(set(a))

    # 预处理文档
    input_file['pr_fileContent'] = input_file['fileContent'].apply(lambda x: x.lower())
    input_file['pr_fileContent'] = input_file['pr_fileContent'].apply(lambda x: nltk.word_tokenize(x))
    input_file['pr_fileContent'] = [[word for word in document if not word in ENG_PUNCTUATIONS] for document in
                                    input_file['pr_fileContent']]
    input_file['pr_fileContent'] = [[word for word in document if not word in stopwords.words('english')] for document in
                                    input_file['pr_fileContent']]

    # 计数器
    input_file['counter'] = input_file['pr_fileContent'].apply(lambda x: Counter(x))

    # 文章单词树
    input_file['length'] = input_file['fileContent'].apply(lambda x: len(list(filter(None, re.split(r"[\n|\s|,|!|.|'|?]", x)))))

    # 文章句子数
    input_file['sentence'] = input_file['fileContent'].apply(lambda x: countSentence(x))

    # 平均句长
    input_file['perLength'] = input_file['length'] /input_file['sentence']

    # 词汇种类
    input_file['wordType'] = input_file['fileContent'].apply(lambda x: wordTypes(x))

    # 词汇密度
    input_file['LD'] = input_file['pr_fileContent'].apply(lambda x: getLD(x))
    input_file['LD'] = input_file['LD'] / input_file['length']
    return input_file


def main():
    df = processDF(loadFile())
    global qjbl
    qjbl = nlp.NaLiPao(df)
    qjbl.process()
    return qjbl






#登录界面
def upload(request):
    if request.method=='GET':
        return render(request,'upload.html')
    else:
        return redirect('/uploadFile/')


#上传文件功能 保存在filedata文件夹下
def uploadFile(request):
    global summa,dword

    if request.method =="GET":
        return render(request, 'uploadFile.html',
                      {
                          'result':summa,
                          'dw':dword,

                          'document':'none',

                          'wordcloud':'none',
                          'histogram':'none',

                          'wc_sild': 'none',
                          'his_sild':'none',

                      })

    else:
        global myfile
        myfile = request.FILES.getlist('myfile')


    if not myfile:
        return HttpResponse("no files for upload!")
    for f in myfile:
        destination = open(os.path.join("filedata", f.name), 'wb+')
        for chunk in f.chunks():
            destination.write(chunk)
            destination.close()

    main()

    summa,dword = qjbl.summary()
    return render(request, 'uploadFile.html',
                  {
                      'result':summa,
                      'dw':dword,

                      'document':'none',

                      'wordcloud': 'none',
                      'histogram': 'none',

                      'wc_sild':'none',
                      'his_sild':'none',

                   })
    '''return render(request, 'uploadFile.html')'''


def document(request):
    if request.method =="GET":
        n = qjbl.total_doc
        df = qjbl.getDocuments()

        global docu
        docu = []
        for i in range(n):
            d = {}
            d['id'] = i
            d['Title'] = df['Title'][i]
            d['Words'] = df['Words'][i]
            d['Types'] = df['Types'][i]
            d['Ratio'] = df['ratio'][i]
            d['WordsSentence'] = df['Words/Sentence'][i]
            docu.append(d)


        return render(request, 'uploadFile.html',
                      {
                          'summary':'none',
                          'docu':docu,

                          'wordcloud': 'none',
                          'histogram': 'none',

                          'wc_sild': 'none',
                          'his_sild': 'none',


                      })


#文档默认显示
def document_single (request):
    global nid
    nid = int(request.GET.get('nid'))

    qjbl.getCloud(nid)
    qjbl.getHistogram(nid)
    global src_wc,src_his
    src_wc = "/static/wordcloud.jpg"
    src_his = "/static/histogram.jpg"

    return render(request, 'uploadFile.html',
                  {
                      'summary':'none',
                      'docu':docu,

                      'src_wc':src_wc,
                      'src_his':src_his,



                    })


def wordcloud_single(request):
    wc_num = int(request.POST['num'])
    qjbl.getCloud(nid, wc_num)
    return render(request, 'uploadFile.html',
                  {

                      'summary': 'none',
                      'docu': docu,

                      'src_wc':src_wc,
                      'src_his':src_his,
                      'def_num_wc': wc_num

                  })


def histogram_single(request):
    his_num = int(request.POST['num'])
    qjbl.getHistogram(nid,his_num)
    return render(request, 'uploadFile.html',
                  {
                      'summary': 'none',
                      'docu': docu,
                      'src_wc': src_wc,
                      'src_his': src_his,
                      'def_num_his': his_num

                  })