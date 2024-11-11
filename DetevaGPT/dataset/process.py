#with open('./data/en_gpt3_features1.jsonl','r',encoding='utf-8')as f:
    lines=f.readlines()
#lines1=[line.strip() for line in lines]

with open('./data/en_gpt3_features.jsonl','r',encoding='utf-8')as f:
        lines=f.readlines()
        lines=[line.strip() for line in lines]
import json
with open('./data_new/en_gpt3_features.jsonl','w',encoding='utf-8')as f:
    for i,line  in enumerate(lines):
        #print(line)
        js_line=eval(line)
        js_line1=eval(lines1[i])
        #print(js_line['losses'])
        losses=js_line['losses']
        losses1=js_line1['losses'][0]
        losses.append(losses1)
        js_line['losses']=losses
        
        begin_idx_list=js_line['begin_idx_list']
        begin_idx_list1=js_line1['begin_idx_list'][0]
        begin_idx_list.append(begin_idx_list1)
        js_line['begin_idx_list']=begin_idx_list
        ll_tokens_list=js_line['ll_tokens_list']
        ll_tokens_list1=js_line1['ll_tokens_list'][0]
        ll_tokens_list.append(ll_tokens_list1)
        js_line['ll_tokens_list']=ll_tokens_list
        # 使用 json.dumps() 序列化 JSON 对象
        json_str = json.dumps(js_line, ensure_ascii=False)
        # 写入文件，并在末尾添加换行符
        f.write(json_str + '\n')
