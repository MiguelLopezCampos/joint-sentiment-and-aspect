import os
import numpy as np
from nltk import word_tokenize
from sklearn.metrics import *
import argparse
import nltk
from nltk.corpus import stopwords


def get_emb(vec_file):
    print(vec_file)
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    idx = 0
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        word_emb[word] = np.array(vec)
        vocabulary[word] = idx
        vocabulary_inv[idx] = word
        idx += 1
    print("# of vocabulary "+str(len(vocabulary)))
    return word_emb, vocabulary, vocabulary_inv

def get_temb(vec_file):
    print(vec_file)
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    t_emb = []
    senti_topic = []
    aspect_topic = []
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        senti = word.split('(')[1].split(',')[0]
        aspect = word.split(')')[0].split(',')[1]
        if senti not in senti_topic:
            senti_topic.append(senti)
        if aspect not in aspect_topic:
            aspect_topic.append(aspect)
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        t_emb.append(np.array(vec))
    return np.array(t_emb), senti_topic, aspect_topic

def get_temb_from_w(word_emb,topic2id):
    t_emb = []
    for t in topic2id:
        t_emb.append(np.array(word_emb[t]))
    return np.array(t_emb)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='datasets/yelp')
    parser.add_argument('--topic_file', default='mix')
    parser.add_argument('--test_file', default='yelp_test.txt')
    parser.add_argument('--gt',default=1)
    parser.add_argument('--hyper')

    args = parser.parse_args()
    test_file = args.test_file
    dataset = args.dataset
    hyper = args.hyper
    gt = args.gt


    text = []
    pred1 = []
    pred2 = []
    gt1 = []
    gt2 = []
    stop_words = set(stopwords.words('english'))

    if gt == 1:  
        with open(os.path.join(dataset, test_file)) as f:
            for line in f:
                tmp = line.split('\t')
                if len(tmp) != 4:
                    continue
                gt1.append(int(tmp[1]))
                gt2.append(int(tmp[2]))
                s = ' '.join([w.lower() for w in word_tokenize(tmp[3].strip()) if w.lower() not in stop_words])
                text.append(s)
    else:
        with open(os.path.join(dataset, test_file)) as f:
            for line in f:
                s = ' '.join([w.lower() for w in word_tokenize(line.strip()) if w.lower() not in stop_words])
                text.append(s)


    topic2id = {}
    id2topic = {}
    

    aspect = args.topic_file
    w_emb_file = 'emb1_'+aspect + hyper + '_w.txt'
    t_emb_file = 'emb1_'+aspect + hyper + '_t.txt'
    
    idx=0
    with open(os.path.join(args.dataset, t_emb_file)) as f:
        for line in f:
            if len(line.strip().split(' ')) < 100:
                continue
            id2topic[str(idx)] = line.strip().split(' ')[0]
            topic2id[line.strip().split(' ')[0]] = str(idx)
            idx += 1

    print(id2topic)

    word_emb, vocabulary, vocabulary_inv = get_emb(os.path.join(args.dataset, w_emb_file))
    topic_emb, senti_topic, aspect_topic = get_temb(vec_file=os.path.join(args.dataset, t_emb_file))
    

    not_in_vocab = 0
    zero_sen = 0
    logit_list = np.zeros((len(text), len(topic2id)))
    with open('pred_'+aspect+'.txt','w') as fout:
        for i,s in enumerate(text):
        # for s in range(len(doc_emb)):
            tmp = np.sum([1 if w in vocabulary else 0 for w in s.split(' ')])
            not_in_vocab += len(s.split(' ')) - tmp
            if tmp == 0:
                zero_sen += 1
                print(s)
            s_rep = np.sum([word_emb[w] if w in vocabulary else np.zeros((100)) for w in s.split(' ')], axis=0)/len(text)


            label = np.argmax(np.dot(s_rep, np.transpose(topic_emb))/np.linalg.norm(topic_emb, axis=1))
            if gt == 1:
                label2 = int(label/len(aspect_topic))
                label1 = label%len(aspect_topic)
                pred1.append(label1)
                pred2.append(1-label2)
                fout.write(aspect_topic[gt1[i]]+' '+aspect_topic[label1]+' '+senti_topic[1-gt2[i]]+' '+senti_topic[label2]+'\n')
            else:
                # print(logit_list[i])
                # print(np.dot(s_rep, np.transpose(topic_emb))/np.linalg.norm(topic_emb, axis=1)/np.linalg.norm(s_rep))
                if np.linalg.norm(s_rep) == 0:
                    logit_list[i] = np.zeros((1,len(id2topic)))
                else:
                    logit_list[i][label] = np.max(np.dot(s_rep, np.transpose(topic_emb))/np.linalg.norm(topic_emb, axis=1)/np.linalg.norm(s_rep))

    if gt != 1:
        print(logit_list[:10])

    if gt == 1:
        acc = accuracy_score(gt1, pred1)
        p = precision_score(gt1, pred1, average='macro')
        r = recall_score(gt1, pred1, average='macro')
        f1_mac = f1_score(gt1, pred1, average='macro')
        
        print(f"Aspect Accuracy: {acc} Precision: {p} Recall: {r} mac-F1: {f1_mac}")
        print(confusion_matrix(gt1, pred1))
        print(f"not in vocab: {not_in_vocab}  zero sentence: {zero_sen}")

        with open(os.path.join(args.dataset, 'pred_res_'+aspect+'.txt'), 'a') as fout:
            fout.write(w_emb_file+' '+t_emb_file+'\n')
            fout.write(f"Accuracy: {acc} Precision: {p} Recall: {r} mac-F1: {f1_mac}\n")


        acc = accuracy_score(gt2, pred2)
        p = precision_score(gt2, pred2, average='macro')
        r = recall_score(gt2, pred2, average='macro')
        f1_mac = f1_score(gt2, pred2, average='macro')
        
        print(f"Sentiment Accuracy: {acc} Precision: {p} Recall: {r} mac-F1: {f1_mac}")
        print(confusion_matrix(gt2, pred2))
        print(f"not in vocab: {not_in_vocab}  zero sentence: {zero_sen}")

        with open(os.path.join(args.dataset, 'pred_res_'+aspect+'.txt'), 'a') as fout:
            # fout.write(w_emb_file+' '+t_emb_file+'\n')
            fout.write(f"Accuracy: {acc} Precision: {p} Recall: {r} mac-F1: {f1_mac}\n\n")
    else:
        with open('pred_'+aspect+'.txt','w') as fout:
            for i,idx in enumerate(id2topic):
                sort_id = np.argsort(-logit_list[:,i])
                fout.write(id2topic[str(i)]+'\n')
                fout.write('\t'.join([str(x) for x in sort_id[:10]])+'\n')
                print([logit_list[x,i] for x in sort_id[:10]])
                print('\n')




    