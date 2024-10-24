from transformers import AutoTokenizer

# calm2-7b-chat用chat template
CALM2_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n'}}{% elif message['role'] == 'system' %}{{ '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + eos_token  + '\\n'}}{% endif %}{% if loop.last and add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}{% endfor %}"

tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")
if tokenizer.chat_template == None:
    tokenizer.chat_template = CALM2_CHAT_TEMPLATE

tokenizer.push_to_hub("calm2-tokenizer")
