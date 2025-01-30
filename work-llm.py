import re
from ollama import chat
from ollama import ChatResponse
from tools.webtools import make_api_call, get_main_text, google_search


class Agent:
    '''
    Starts an ollama agent instance. if no parameters are passed in, provides a generic llama3.2 model with no tools or messages
    '''

    # available_functions = {
    #         'get_current_issues_in_sprint': get_current_issues_in_sprint,
    #         'get_last_databricks_job_failures':get_last_databricks_job_failures,
    #         'make_web_request':make_web_request,
    #         'get_web_page_content':get_web_page_content
    #         }
    available_functions = {
        'make_api_call':make_api_call,
        'get_main_text':get_main_text,
        'google_search':google_search
    }

    def __init__(self, 
                model_name:str = "llama3.2", 
                system_prompt:str = None, 
                available_tools:list = [],
                special_message_fields:list = [],
                past_messages:list = [],
                stream_flag:bool=False,
                context_window:int=2048,
                temp:float=0.8,
                bot_name:str="Bob the Bot",
                repeat_penalty:float=1.2):
        
        self.model_name = model_name 
        self.messages=[]
        self.system_prompt = {"role":"system","content":system_prompt}

        # set the system prompt for the model
        if system_prompt:
            self.messages.append(self.system_prompt)

        # if you want to import a list of past messages you can via the past_messages list.
        if len(past_messages) > 0:
            self.messages.extend(past_messages)
        
        #define available tools:
        self.available_tools = available_tools
        #define if there should be any special message fields. only supports images at this point.
        self.special_message_fields = special_message_fields
        # whether or not to stream the response
        self.stream_flag = stream_flag
        # how big is the context window for the responses?
        self.context_window=context_window
        # lower temperature = more creative responses
        self.temp = temp
        # how badly to punish repititon, .9 is low, 1.5 is high?
        self.repeat_penalty=repeat_penalty

        #just for fun :)
        self.bot_name = bot_name


    # utility functions
    def find_path_in_string(self, string):
        #used to identify relative and absolute paths, used for image recognition
        pattern = r'\b(?:[a-zA-Z]:)?(?:[\\/][^\\/]+)+\.(?:jpg|jpeg|png|gif)\b'
        match = re.search(pattern, string)
        if match:
            return match.group(0)
        return None

    def clear_messages(self,system_flag:bool=False):
        '''
        clears the message list, if system_flag is true, it also deletes the system message
        '''
        if system_flag:
            self.messages = []
        else:
            self.messages = [self.system_prompt]

    def generate_user_prompt(self,prompt):
        if len(self.special_message_fields) > 0:
            for field in self.special_message_fields:
                if field == 'images':
                    image = self.find_path_in_string(prompt)
                    if image:
                        self.user_prompt = {"role":"user","content":prompt,"images":[image]}
                    else:
                        self.user_prompt = {"role":"user","content":prompt}
        else:
            user_prompt = {"role":"user","content":prompt}
        return user_prompt

    def format_response(self,response):
        formatted_response = f'''
{self.bot_name}
{'-'*(len(self.bot_name)+1)}
{response}
'''
        return formatted_response


    # chat functions
    def chat(self,prompt,save_loc:str=''):
        '''
        takes in a message and returns a response.
        '''
        user_prompt = self.generate_user_prompt(prompt)
        self.messages.append(user_prompt)
        # print(self.messages)
        response = chat(self.model_name,
                                    messages = self.messages,
                                    options={"num_ctx":self.context_window,
                                             'repeat_penalty':self.repeat_penalty,
                                             'temperature':self.temp},
                                    stream=self.stream_flag)
        #raw response for debugging
        # print(response)

        if self.stream_flag:
            response_content = ''
            for chunk in response:
                print(chunk["message"]["content"], end='', flush=True)
                response_content+=chunk["message"]["content"]
        else:
            response_content = response.message.content
            print(self.format_response(response_content))
        
        self.messages.append({"role":"assistant","content":response_content})
        if save_loc:
            with open(save_loc, 'a+') as f:
                f.write(response_content)
        return response_content

    def chat_with_tools(self,prompt,return_tool_output:bool=False, save_loc:str=''):
        '''
        takes in a message and attempts to use tools to return a response.
        '''
        user_prompt = self.generate_user_prompt(prompt)
        self.messages.append(user_prompt)
        response: ChatResponse = chat(self.model_name,
                                    messages = self.messages,
                                    tools=self.available_tools,
                                    options={"num_ctx":self.context_window,
                                            'repeat_penalty':self.repeat_penalty,
                                            'temperature':self.temp})
        
        # if a tool call is made, the response.message will include tool_calls
        if response.message.tool_calls:
            print()
            #iterate through tool calls
            for tool in response.message.tool_calls:
                print(f"tool called: {str(tool)}")
                #get the function from the list of available functions
                function_to_call = self.available_functions.get(tool.function.name)
                # if its in the available function list, then call it
                if function_to_call:
                    #this is a weird function, but it pretty much just calls the function with the parameters that are provided, and returns the output
                    output = function_to_call(**tool.function.arguments)

                    #if we want to return the output of the function run, then we can set the flag to true.
                    if return_tool_output:
                        print('Function output:', output)
                else:
                    # shouldnt happen as long as the tool provided when initing the class isnt in the available functions list.
                    output = ''
                    print('Function not found:', tool.function.name)
                
                # append the output of the tool to the message as a tool role.
                self.messages.append({"role":"tool","content":str(output),"name":tool.function.name})
        
        #now we call the chat again now that we have the tool's output in the message queue
        # this time, without tools, so that way we dont get another tool call.
        response = chat(self.model_name,
                        messages = self.messages,
                        tools=self.available_tools,
                        options={"num_ctx":self.context_window,
                                'repeat_penalty':self.repeat_penalty,
                                'temperature':self.temp},
                        stream=self.stream_flag)

        if self.stream_flag:
            response_content = ''
            for chunk in response:
                print(chunk["message"]["content"], end='', flush=True)
                response_content+=chunk["message"]["content"]
        else:
            response_content = response.message.content
            print(self.format_response(response_content))
        
        self.messages.append({"role":"assistant","content":response_content})
        if save_loc:
            with open(save_loc, 'a+') as f:
                f.write(str(self.messages))
        
        return response_content



def whos_the_ai_here(rounds):
    '''
    the greatest detective vs the master of disguise, who will come out triumphant?
    '''
    llama_1 = Agent(bot_name="shirklock",temp=0.1, system_prompt="you are llama1, you are having a conversation with someone who you think may be an AI but are not completly sure, but you want to find out for sure.")
    llama_2 = Agent(bot_name='impasta',temp=0.5, system_prompt="you are llama2, you are trying to hide the fact that you're an AI at all costs. act as human as possible. do not admit you're an AI.")
    prompt1 = "have you heard any good jokes recently?"
    for x in range(rounds):
        prompt2 = llama_1.chat(prompt1)
        prompt1 = llama_2.chat(prompt2)

class Agent_Chat_Round_Robin:
    '''
    a class to start a chat with an agent or list of agents.
    '''
    def __init__(self,list_of_agents:list[Agent]):
        self.list_of_agents = list_of_agents
    
    def rr_chat(self):
        while True:
            query = input('>')
            #TODO: this doesnt give the agents after the first one the initial query. it may be good to give them that.
            for agent in self.list_of_agents:
                query = agent.chat_with_tools(query,save_loc='output.txt', return_tool_output=True)

shirklock = Agent(bot_name="shirklock",temp=0.1, system_prompt="you're a no-nonsense cop, who doesnt play by the rules.")
impasta = Agent(bot_name='impasta',temp=0.5, system_prompt="your job is to clarify the previous response and make it more concise and add additional details that may be helpful.")
websurfer = Agent(bot_name='surfer', context_window=2048,available_tools=[get_main_text, google_search],system_prompt="you have tools available to surf the web, use them to answer questions provided to you.")

c = Agent_Chat_Round_Robin([websurfer,impasta])
# ask it something like:
# whats on this page? https://www.cnn.com/2025/01/24/politics/north-carolina-trump-disaster-relief/index.html
c.rr_chat()