
import json
import transformers
from dataclasses import dataclass, field
import requests
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

API_KEY = "TG885GFJUskqteqVlsInIWO4"
SECRET_KEY = "EeLxf5eXyWbbfq04yY4l1kow2YRly19x"
# API_KEY = None
# SECRET_KEY = None

@dataclass()
class DataArguments():
    prompt_path: str = field(
        default=None, metadata={"help": "Path to the prompt file to get data from llm."}
    )


    user_role: str = field(default="用户", metadata={"help": "role of user in prompt" })
    llm_role: str = field(default=None, metadata={"help": "role of LLM in prompt" })

    write_data_path : str = field(default="./data.json", metadata={"help": "the path of data for writing" })
    run_num: int =field(default=3, metadata={"help": "the run num of one  prompt" })
    use_local_model : bool=field(default=False, metadata={"help": "use the local model" })
    api_key: str = field(default=None, metadata={"help": "the api key of baidu" })
    secret_key: str = field(default=None, metadata={"help": "the secret key of baidu" })

@dataclass()
class ModelArguments():
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to the model."}
    )

print("start geting data from llm")

def get_access_token(api_key,secret_key):
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key}
    return str(requests.post(url, params=params).json().get("access_token"))


def get_data():
    parser = transformers.HfArgumentParser(
        (DataArguments,ModelArguments)
    )
    data_args,model_args = parser.parse_args_into_dataclasses()
    print(data_args.prompt_path)
    print(data_args.use_local_model)

    datas=[]
    model=None
    if data_args.use_local_model:
    	tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    	model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    device_map="auto",
    trust_remote_code=True
).eval()



    
    
    with open(data_args.prompt_path) as fcc_file:
    	fcc_data = json.load(fcc_file)
    	for prompt in fcc_data["prompts"]:
    		for i in range(data_args.run_num):
    			if data_args.use_local_model:
    				result, history = model.chat(tokenizer, prompt, history=None)
    			else:
    				# API_KEY = data_args.api_key
    				# SECRET_KEY = data_args.secret_key

    				
		    		url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token="+get_access_token(data_args.api_key,data_args.secret_key)
		    		payload = json.dumps({
					    "messages": [
					        {
					            "role": "user",
					            "content": prompt
					        },
					       
					    ],
					    "disable_search": False,
					    "enable_citation": False
					})
			    	headers = {
					    'Content-Type': 'application/json'
					}
			    	response = requests.request("POST", url, headers=headers, data=payload)
			    	data = json.loads(response.text)
			    	print(data)
			    	result=data['result']
		    	print("result")
		    	print(result)
		    	pattern_user_turn = re.compile(r'{user_role}：(.*?)\n{llm_role}：'.format(user_role=data_args.user_role,llm_role=data_args.llm_role), re.DOTALL)
		    	pattern_llm_turn = re.compile(r'{llm_role}：(.*?)\n{user_role}：'.format(llm_role=data_args.llm_role,user_role=data_args.user_role), re.DOTALL)  
		    	llm_turns= pattern_llm_turn.findall(result)
		    	user_turns = pattern_user_turn.findall(result)
		    	turn={"cov":[]}

		    	for user_turn,llm_turn in zip(user_turns,llm_turns):
				
				    turn['cov'].append({"value":user_turn.strip(),"from":"user"})
				    turn['cov'].append({"value":llm_turn.strip(),"from":"llm"})
				    print(user_turn.strip())
				    print(llm_turn.strip())
				
				
		    	datas.append(turn)
		    	print(datas)
    with open(data_args.write_data_path,"a") as f:
    	json.dump(datas,f, indent=4,ensure_ascii=False)
    	print("ending...")
		
		



		    
			# pattern_llm_turn = re.compile(r'卡卡罗特：(.*?)\n用户：', re.DOTALL)  
			# user_turns = pattern_user_turn.findall(text) 
			# llm_turns= pattern_llm_turn.findall(text)  




get_data()





