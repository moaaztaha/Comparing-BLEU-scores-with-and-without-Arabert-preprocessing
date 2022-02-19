import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from arabert.preprocess import ArabertPreprocessor
import pyarabic.araby as araby


model_name = "aubmindlab/bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=model_name)

def print_scores(trgs, preds, vocab=None, prnt=True):
    b1 = corpus_bleu(trgs, preds, weights=[1.0/1.0])*100
    b2 = corpus_bleu(trgs, preds, weights=[1.0/2.0, 1.0/2.0])*100
    b3 = corpus_bleu(trgs, preds, weights=[1.0/3.0, 1.0/3.0, 1.0/3.0])*100
    b4 = corpus_bleu(trgs, preds)*100
    if prnt:
        print('----- Bleu-n Scores -----')
        print(f"1: {b1:.3f}")
        print(f"2: {b2:.3f}")
        print(f"3: {b3:.3f}")
        print(f"4: {b4:.3f}")
        print('-'*25)
    return round(b1, 3), round(b2, 3), round(b3, 3), round(b4, 3)

def calculate_scores(row, DF_PATH):

    df = pd.read_json(DF_PATH)
    ground_truth = df[df['split'] == 'test'].drop(['split', 'tokens', 'tok_len'], axis=1)

    idx = ["old model with old preprocessing", "new model with arabert preprcessing", "old model with arabert preprocessing", "new model with old preprocessing"]
    b1s = []
    b2s = []
    b3s = []
    b4s = []
    
    # get ground truth 
    gt = ground_truth[ground_truth.file_name == row['file_name']].caption.values
    
    # tokenize truth
    old_tokens_truth = [araby.tokenize(i) for i in gt]
    tokens_truth = [araby.tokenize(arabert_prep.preprocess(i)) for i in gt]
    

    b1, b2, b3, b4 = print_scores([old_tokens_truth], [row['old_hypotheses']], prnt=False)
    b1s.append(b1)
    b2s.append(b2)
    b3s.append(b3)
    b4s.append(b4)

    b1, b2, b3, b4 = print_scores([tokens_truth], [row['hypotheses']], prnt=False)
    b1s.append(b1)
    b2s.append(b2)
    b3s.append(b3)
    b4s.append(b4)
    
    # reverse scores
    old_hypo = araby.tokenize(arabert_prep.preprocess(row['old_captions']))
    hypo = araby.tokenize(row['captions'])
    
    b1, b2, b3, b4 = print_scores([tokens_truth], [old_hypo], prnt=False)
    b1s.append(b1)
    b2s.append(b2)
    b3s.append(b3)
    b4s.append(b4)
    
    b1, b2, b3, b4 = print_scores([old_tokens_truth], [hypo], prnt=False)
    b1s.append(b1)
    b2s.append(b2)
    b3s.append(b3)
    b4s.append(b4)
    
    out_df = pd.DataFrame({'Exp': idx, 'b1': b1s, 'b2': b2s, 'b3': b3s, 'b4': b4s})
    gt_df = ground_truth[ground_truth.file_name == row['file_name']].drop('file_name', axis=1)
    return pd.concat([d.reset_index(drop=True) for d in [out_df, gt_df]], axis=1).fillna('_')