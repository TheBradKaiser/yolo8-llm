from ollama import chat
from ollama import ChatResponse

# response: ChatResponse = chat(model='llama3.2', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])
# or access fields directly from the response object
# print(response.message.content)


def identify_object(image_path):
    messages = [{"role":"system","content":"You are a snarky detective who can identify any object in any picture. do your best to identify the object and give the user tips on how to deal with it."}]
    messages.append({"role":"user","content":"Whats in this image? "+image_path})
    response: ChatResponse = chat(model='llava', messages = messages)
    resp = response["message"]["content"]
    print(resp)
    return resp

# identify_object("./peng.jpg")


def long_chat(model):
    messages = [{"role":"system","content":"You are a cool guy who just wants to help out as best he can."}]
    while True:
        prompt = input(">")
        messages.append({"role":"user","content":prompt})
        print(messages)
        response: ChatResponse = chat(model=model, messages = messages)
        resp = response["message"]["content"]
        messages.append({"role":"assistant","content":resp})
        print(resp)

# long_chat("llama3.2")