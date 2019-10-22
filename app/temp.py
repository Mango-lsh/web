import os
import re
import os.path
import codecs
import numpy as np
import pandas as pd
#import docx
import nltk
from nltk.corpus import stopwords
import chardet
from collections import Counter
import wordcloud  # 词云展示库
from PIL import Image  # 图像处理库
import matplotlib.pyplot as plt  # 图像展示库
import pyocr
import importlib
import sys
import time
import os.path
import jieba
from wordcloud import WordCloud
from django.shortcuts import HttpResponse,render,redirect
importlib.reload(sys)
from  app import func
#pdf处理
'''
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
'''

'''# 词云功能
def wordcloud(request):
    a = pd.read_excel('filedata/' + i, header=None)
    a['word1'] = a[0].apply(lambda x: jieba.lcut(str(x)))
    new_text = ' '
    for i in a['word1']:
        for x in i:
            new_text = new_text + x + ' '

        wordcloud = WordCloud(font_path='/System/Library/Fonts/STHeiti Medium.ttc',
                                  background_color='white').generate(new_text)
        plt.imshow(wordcloud)
        plt.axis('off')
        wordcloud.to_file("static/" + view.myfile[0]+ ".jpg")
        return render(request, 'wc.html', )'''
