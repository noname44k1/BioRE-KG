import json  
import re  
  
state_dict = {"p": 0, "c": 0, "g": 0}  
state_dict_h = {"p": 0, "c": 0, "g": 0}  
state_dict_t = {"p": 0, "c": 0, "g": 0}  
state_dict_r = {"p": 0, "c": 0, "g": 0}  
  
def calclulate_f1(statics_dict, prefix=""):  
    """  
    Calculate the prec, recall and f1-score for the given state dict.  
    The state dict contains predict_num, golden_num, correct_num.  
    Reutrn a dict in the form as "prefx-recall": 0.99.  
    """  
    #{'p': 93, 'c': 34, 'g': 320}  
    prec, recall, f1 = 0, 0, 0  
    if statics_dict["c"] != 0:  
        prec = float(statics_dict["c"] / statics_dict["p"])  
        recall = float(statics_dict["c"] / statics_dict["g"])  
        f1 = float(prec * recall) / float(prec + recall) * 2  
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}  
  
  
i=-1  
with open("chuck_og_triplet_api_8000.json", "r", encoding="utf-8") as fr:  
    for line in fr.readlines():  
        line=line.strip()  
        line=json.loads(line)  
        i=i+1  
        P=set()  
        sentence=line["sentence"].lower()  
        gold_triples=set()  
        ground_truth=line["ground_truth"]  
         
        gold_t=ground_truth.split("|")  
        gh=gold_t[0].lower().replace("       "," ")  
        gr=gold_t[1].lower()  
        gt=gold_t[2].lower().replace("       "," ")  
        
        # Chỉ thêm complete triple vào gold_triples  
        gold_triples.add((gh,gr,gt))  
              
        predictions=line["predicted"].split("\n\n")[3].split("\n")  
  
        if len(predictions)==2:  
            predicted_=predictions[1]  
            # Cắt bỏ phần rác model hallucinate sau triplet (vd: " Context: ...", " ### ...")  
            predicted_=re.split(r'\s+Context:|\s+###|\s+\|Context', predicted_)[0].strip()  
            predicte_t=predicted_.split("|")  
            if len(predicte_t)==3:  
                ph=predicte_t[0].lower().replace("       "," ").strip()  
                pt=predicte_t[2].lower().replace("       "," ").strip()  
                pr=predicte_t[1].lower()  
               
                # Chỉ thêm complete triple vào P  
                P.add((ph,pr,pt))  
          
        # Cập nhật state_dict cho complete triples (giống logic gốc)  
        state_dict["p"] += len(P)  
        state_dict["g"] += len(gold_triples)  
        state_dict["c"] += len(P & gold_triples)  
          
        # Theo dõi individual components riêng biệt  
        gold_h_set = set([gh])  
        gold_t_set = set([gt])  
        gold_r_set = set([gr])  
          
        pred_h_set = set()  
        pred_t_set = set()  
        pred_r_set = set()  
          
        if len(predictions)==2 and len(predicte_t)==3:  
            pred_h_set.add(ph)  
            pred_t_set.add(pt)  
            pred_r_set.add(pr)  
          
        # Cập nhật state_dict cho từng thành phần  
        state_dict_h["p"] += len(pred_h_set)  
        state_dict_h["g"] += len(gold_h_set)  
        state_dict_h["c"] += len(pred_h_set & gold_h_set)  
          
        state_dict_t["p"] += len(pred_t_set)  
        state_dict_t["g"] += len(gold_t_set)  
        state_dict_t["c"] += len(pred_t_set & gold_t_set)  
          
        state_dict_r["p"] += len(pred_r_set)  
        state_dict_r["g"] += len(gold_r_set)  
        state_dict_r["c"] += len(pred_r_set & gold_r_set)  
  
# Tính và in F1 cho tất cả các thành phần  
triple_metirc_results = calclulate_f1(state_dict, 'triple')  
h_metirc_results = calclulate_f1(state_dict_h, 'h')  
t_metirc_results = calclulate_f1(state_dict_t, 't')  
r_metirc_results = calclulate_f1(state_dict_r, 'r')  
  
print(triple_metirc_results)  
print(h_metirc_results)  
print(t_metirc_results)  
print(r_metirc_results)