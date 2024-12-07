import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import string
from nltk import ngrams
import wordcloud as wc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn import metrics
import pickle
import streamlit as st


# 1. Data preprocessing
review_df = pd.read_csv("./Cung_cap_HV/data/Danh_gia.csv")
review_df = review_df[review_df['noi_dung_binh_luan'].isnull()==False]
review_df['thai_do'] = review_df["so_sao"].apply(lambda x: 'negative' if x<=3  else 'positive')
#LOAD EMOJICON
file = open('./Cung_cap_HV/files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('./Cung_cap_HV/files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
#################
# LOAD ADJECTIVE LIST
file = open('./Cung_cap_HV/files/tinhtu.txt', 'r', encoding="utf8")
adj_lst = file.read().split('\n')
file.close()
#################
# LOAD VERB LIST
file = open('./Cung_cap_HV/files/dongtu.txt', 'r', encoding="utf8")
verb_lst = file.read().split('\n')
file.close()
#################
# LOAD NOUN LIST
file = open('./Cung_cap_HV/files/danhtu.txt', 'r', encoding="utf8")
noun_lst = file.read().split('\n')
file.close()
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def process_text(text, emoji_dict, teen_dict):
    document = str(text).lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):

        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))

        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))

        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())

        new_sentence = new_sentence+ sentence + '. '
        
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document

tu_phu_dinh = ['không', 'chẳng', 'chả', "chưa", "khỏi"]

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        twograms = ngrams(sentence.split(), 2)
        for grams in twograms:
            word = grams[0]+"_"+grams[1]
            if (word.lower() in adj_lst) or (word.lower() in tu_phu_dinh):
                new_document += " " + word.lower()
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

import re
# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "lònggggg" thành "lòng", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)

# Áp dụng hàm chuẩn hóa cho văn bản

review_df['noi_dung_binh_luan'] = review_df["noi_dung_binh_luan"].apply(lambda x: covert_unicode(x))
review_df['noi_dung_binh_luan'] = review_df["noi_dung_binh_luan"].apply(lambda x: process_text(x,emoji_dict, teen_dict))
review_df['noi_dung_binh_luan'] = review_df["noi_dung_binh_luan"].apply(lambda x: normalize_repeated_characters(x))
review_df['noi_dung_binh_luan'] = review_df["noi_dung_binh_luan"].apply(lambda x: process_postag_thesea(x))

review_df.to_csv('processed_data.csv', index=False)

# 2. Build model

df = pd.read_csv('./processed_data.csv')
df = df[['noi_dung_binh_luan','thai_do']]

df.fillna('trống', inplace=True)

thai_do_dict = {'positive':1, 'negative':0}
df['thai_do'] = df['thai_do'].map(thai_do_dict)

X_train, X_test, y_train, y_test = train_test_split(df['noi_dung_binh_luan'], df['thai_do'], test_size=0.2, random_state=42, stratify=df['thai_do'])

from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

vectorizer = TfidfVectorizer(max_features=500) 
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

svm_model = LinearSVC(class_weight=class_weight_dict)
svm_model.fit(X_train_vectorized, y_train)

y_pred = svm_model.predict(X_test_vectorized)

score_train = svm_model.score(X_train,y_train)
score_test = svm_model.score(X_test,y_test)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cr = classification_report(y_test, y_pred)

y_prob = svm_model.predict_proba(X_test)
roc = roc_auc_score(y_test, y_prob[:, 1])

#5. Save models
# luu model classication
pkl_filename = "svm_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(svm_model, file)
  
# luu model Vectorizer 
pkl_vectorizer = "vectorizer_model.pkl"  
with open(pkl_vectorizer, 'wb') as file:  
    pickle.dump(vectorizer, file)

#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    svm_model = pickle.load(file)
# doc model count len
with open(pkl_vectorizer, 'rb') as file:  
    vectorizer_model = pickle.load(file)
    
#--------------
# GUI
st.title("Data Science Project")
st.write("## Sentimental Analysis")

menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Bùi Văn Bình
                 Lê Thị Thanh Trúc""")
st.sidebar.write("""#### Giảng viên hướng dẫn: Phương Khuất Thùy""")
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Classifying spam and ham messages is one of the most common natural language processing tasks for emails and chat engines. With the advancements in machine learning and natural language processing techniques, it is now possible to separate spam messages from ham messages with a high degree of accuracy.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for ham and spam message classification.""")
    st.image("ham_spam.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(review_df[['noi_dung_binh_luan','thai_do']].head(3))
    st.dataframe(df[['noi_dung_binh_luan','thai_do']].tail(3))  
    st.write("##### 2. Visualize positive & negative")
    fig1 = sns.countplot(data=df[['thai_do']], x='thai_do')    
    st.pyplot(fig1.figure)

    st.write("##### 3. Build model...")
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("Accuracy:"+str(round(acc,2)))
    st.write("###### Confusion matrix:")
    st.code(cm)
    st.write("###### Classification report:")
    st.code(cr)
    st.code("Roc AUC score:" + str(round(roc,2)))

    # calculate roc curve
    st.write("###### ROC curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    fig, ax = plt.subplots()       
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(fpr, tpr, marker='.')
    st.pyplot(fig)

    st.write("##### 5. Summary: This model is good enough for negative & positive classification.")

elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0]     
            flag = True                          
    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            lines = np.array([content])
            flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)        
            x_new = vectorizer_model.transform(lines)        
            y_pred_new = svm_model.predict(x_new)       
            st.code("New predictions (0: Ham, 1: Spam): " + str(y_pred_new)) 
