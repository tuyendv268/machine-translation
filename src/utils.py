from ntpath import join
from underthesea import word_tokenize

import configparser
import json
import os

def load(path, lang):
    datas = []
    for file in os.listdir(path):
        if not file.endswith(lang):
            continue
        abs_path = os.path.join(path, file)
        tmp_datas = load_data(abs_path)
        datas += tmp_datas
    return datas
        

def load_data(path):
    with open(path, "r", encoding="utf-8") as tmp:
        datas = tmp.readlines()
        datas = [sent.replace("\n", "") for sent in datas]
    
    return datas

def build_vi_dict(path):
    print("-----building vietnamese word2index-----")
    config = configparser.ConfigParser()
    config.read('src/conf/config.cfg')
    
    datas = load(path, lang="vi")
    datas = [word_tokenize(sent) for sent in datas]
    
    START_TOKEN, START_TOKEN_ID = config["dict"]["START_TOKEN"], config["dict"]["START_TOKEN_ID"]
    PAD_TOKEN, PAD_TOKEN_ID = config["dict"]["PAD_TOKEN"], config["dict"]["PAD_TOKEN_ID"]
    END_TOKEN, END_TOKEN_ID = config["dict"]["END_TOKEN"], config["dict"]["END_TOKEN_ID"]
    UNK_TOKEN, UNK_TOKEN_ID = config["dict"]["UNK_TOKEN"], config["dict"]["UNK_TOKEN_ID"]
    
    word2index = {}
    word2index[START_TOKEN] = START_TOKEN_ID
    word2index[PAD_TOKEN] = PAD_TOKEN_ID
    word2index[END_TOKEN] = END_TOKEN_ID
    word2index[UNK_TOKEN] = UNK_TOKEN_ID
    
    for sent in datas:
        for word in sent:
            if word in word2index:
                continue
            else:
                word2index[word]=len(word2index.keys())
    word2index_path = config["path"]["word2index_vi"]
    with open(word2index_path, "w", encoding="utf-8") as tmp:
        tmp.write(json.dumps(word2index ,indent=4, ensure_ascii=False))
        print("saved: ", word2index_path)
    
    print("-----building vietnamese index2word-----")
    index2word = {}
    
    for key, value in word2index.items():
        index2word[value] = key
        
    index2word_path = config["path"]["index2word_vi"]
    with open(index2word_path, "w", encoding="utf-8") as tmp:
        tmp.write(json.dumps(index2word ,indent=4, ensure_ascii=False))
        print("saved: ", index2word_path)
    print("done!")
        
def build_en_dict(path):
    print("-----building english word2index-----")
    config = configparser.ConfigParser()
    config.read('src/conf/config.cfg')
    
    datas = load(path, lang="en")
    datas = [sent.split() for sent in datas]
    
    START_TOKEN, START_TOKEN_ID = config["dict"]["START_TOKEN"], config["dict"]["START_TOKEN_ID"]
    PAD_TOKEN, PAD_TOKEN_ID = config["dict"]["PAD_TOKEN"], config["dict"]["PAD_TOKEN_ID"]
    END_TOKEN, END_TOKEN_ID = config["dict"]["END_TOKEN"], config["dict"]["END_TOKEN_ID"]
    UNK_TOKEN, UNK_TOKEN_ID = config["dict"]["UNK_TOKEN"], config["dict"]["UNK_TOKEN_ID"]
    
    word2index = {}
    word2index[START_TOKEN] = START_TOKEN_ID
    word2index[PAD_TOKEN] = PAD_TOKEN_ID
    word2index[END_TOKEN] = END_TOKEN_ID
    word2index[UNK_TOKEN] = UNK_TOKEN_ID
    
    for sent in datas:
        for word in sent:
            if word in word2index:
                continue
            else:
                word2index[word]=len(word2index.keys())
    
    word2index_path = config["path"]["word2index_en"]
    with open(word2index_path, "w", encoding="utf-8") as tmp:
        tmp.write(json.dumps(word2index ,indent=4, ensure_ascii=False))
        print("saved: ", word2index_path)
        
    print("-----building english index2word-----")
    index2word = {}
    
    for key, value in word2index.items():
        index2word[value] = key
        
    index2word_path = config["path"]["index2word_en"]
    with open(index2word_path, "w", encoding="utf-8") as tmp:
        tmp.write(json.dumps(index2word ,indent=4, ensure_ascii=False))
        print("saved: ", index2word_path)
    print("done!")
        
    