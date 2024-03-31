from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Model names: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("./output/longzhu", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained(
    "./output/longzhu",
    device_map="auto",
    trust_remote_code=True
).eval()

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
#model.generation_config = GenerationConfig.from_pretrained("./output/Deita-7B-Scorers", trust_remote_code=True)

i=0
while True:
    print("我：", end="")
    text = input()
    if i==0:
        history=None
    response, history = model.chat(tokenizer, text, history=history)
    print("AI：", end="")
    print(response)



# 1st dialogue turn
# response, history = model.chat(tokenizer, "你好！卡卡罗特", history=None,top_p=0.8)
# print("用户:"+)
# print(response)
# #print(history)
# response, history = model.chat(tokenizer, "你和贝吉塔之间的关系！", history=history)
# print(response)
# print(history)
# response, history = model.chat(tokenizer, "你和贝吉塔之间有什么故事吗", history=history)
# print(response)
# print(history)
# 你好！很高兴为你提供帮助。

# # 2nd dialogue turn
# response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
# print(response)
# # 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# # 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# # 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# # 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# # 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# # 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# # 3rd dialogue turn
# response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
# print(response)
# # 《奋斗创业：一个年轻人的成功之路》
