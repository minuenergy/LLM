import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

model_id = "TinyPixel/Llama-2-7B-bf16-sharded"
peft_model_id = "LLM-Finetuning/guanaco-llama2-finetune/checkpoint-801/"
#peft_model_id = "weights/Llama-2-7B-bf16-sharded/"


config = PeftConfig.from_pretrained(peft_model_id)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    load_in_4bit=False,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.eval()

prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: %s ### Response: "

def gen(x):
    print('Q:', x)
    q = prompt % (x,)
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
    )
    return tokenizer.decode(gened[0]).replace(q, "")


#print(gen("Please could you give me a summary of the tv show Monkey, produced in the 1970's and 80's and based on the Chinese novel Journey to the West."))
#print(gen("What is the best camera lens for general purpose photography?"))
#print(gen("Can you simplify this python condition? elif a[1] == str(3) or a[1] == str(4)"))
#print(gen("Can you make it also be able to deal with the numbers as integers?"))

print('A:',gen("In which US state can you get a license to hunt a mythical creature with a horn? Why did they pass this law?"))
print('A:',gen("Whats the legislation say on Ohio gambling laws for opening up a cassino?"))
print('A:',gen("Is it legal to download YouTube videos in the US?"))

