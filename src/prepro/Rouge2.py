from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('/Users/shota/Documents/ginza/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers', do_lower_case=True)

def Rouge1(out,ref):
    Fm = 0
    out_list = tokenizer.tokenize(out)
    ref_list = set(tokenizer.tokenize(ref))
    mach = 0
    for word in ref_list:
        if word in out_list:
            mach += 1
    R = mach/len(ref_list)
    if len(out_list) != 0:
        P = mach/(len(out_list))
    else:
        P = 0
    if R+P != 0: 
        Fm = 2*R*P/(R+P)
    return(Fm)

def Rouge2(out,ref):
    Fm = 0
    out_list = tokenizer.tokenize(out)
    ref_list = tokenizer.tokenize(ref)
    mach = 0
    for i, word in enumerate(ref_list[:-2]):
        for j,word2 in enumerate(out_list[:-2]):
            if word == word2 and ref_list[i+1]==out_list[j+1]:
                mach += 1
                
    R = mach/(len(ref_list)-1)
    if len(out_list) != 0:
        P = mach/(len(out_list))
    else:
        P = 0
    if R+P != 0: 
        Fm = 2*R*P/(R+P)
    return(Fm)

def RougeL(out,ref):
    Fm = 0
    out_list = tokenizer.tokenize(out)
    ref_list = tokenizer.tokenize(ref)
    mach = 0
    max_mach = 0
    for i, word in enumerate(ref_list):
        for j,word2 in enumerate(out_list[:-2]):
            if word == word2:
                l = 1
                while j+l < len(out_list) and i+l < len(ref_list):
                    if ref_list[i+l] == out_list[j+l]:
                        l += 1
                    else:
                        break
                if l > max_mach:
                    max_mach = l  
  
                
    R = max_mach/(len(ref_list))
    if len(out_list) != 0:
        P = max_mach/(len(out_list))
    else:
        P = 0
    if R+P != 0: 
        Fm = 2*R*P/(R+P)
    return(Fm)
    