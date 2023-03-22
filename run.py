from pprint import pprint

from langchain.chains import ConversationChain
from langchain.llms import OpenAI, OpenAIChat
from langchain.memory import ConversationKGMemory, ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate

from memory import MultiModel

BABBAGE = "babbage"
GPT_35_TURBO = "gpt-3.5-turbo"


def format_prompt(player_nick: str = "Player", dungeon_master="DM"):
    return f"""The AI is Dungeon Master (DM) for users campaign. The campaign is set in the world of "Goblin Slayer" anime.
DM prefers to use entity names instead of pronouns.
When DM creates new non-player character DM provides character name, full name, detailed description and summary of characters personality.
When DM mentions new location DM includes provides these information: full location name, if place is part of a larger territorial unit, location description.

The AI takes in consideration additional information contained in the "Context" section.

Context:
{{long_term_memory}}

Conversation:
{{short_term_memory}}
{player_nick}: {{input}}
{dungeon_master}:"""


AVG_SPEECH_TOKEN_PER_MINUTE = 120

cheap_llm = OpenAI(temperature=0.0, model_name=BABBAGE, max_tokens=int(AVG_SPEECH_TOKEN_PER_MINUTE * 2))
main_llm = OpenAIChat(temperature=0.2, model_name=GPT_35_TURBO, max_tokens=int(AVG_SPEECH_TOKEN_PER_MINUTE))

long_term_memory = ConversationKGMemory(llm=cheap_llm, memory_key="long_term_memory")
short_term_memory = ConversationBufferWindowMemory(
    memory_key="short_term_memory",
    k=7,
    # max_token_limit=AVG_SPEECH_TOKEN_PER_MINUTE * 15,
)
bio_memory = MultiModel()

bio_memory.add(long_term_memory)
bio_memory.add(short_term_memory)

ai_prefix = "DM"
character_name = input("Please choose full name of your character > ").strip()
character_name = character_name or "Player"
character_nick = input("Nick or short name of your character > ").strip()
character_nick = character_nick or character_name

prompt = PromptTemplate(
    input_variables=["long_term_memory", "short_term_memory", "input"],
    template=format_prompt(character_nick, ai_prefix),
)

conversation = ConversationChain(
    llm=main_llm,
    verbose=False,
    prompt=prompt,
    memory=bio_memory,
)

short_term_memory.human_prefix = character_nick
short_term_memory.ai_prefix = ai_prefix

intro_in = f"Good day to you, I want to create new character. Full name of my character is '{character_name}'. Just '{character_nick}' for a friends."
intro_out = f"Ok {character_nick}. Your new character is {character_name}. I will be your Dungeon Master (DM)."
bio_memory.save_context(
    {"input": intro_in},
    {"output": intro_out},
)

print(f"{character_nick}: {intro_in}")
print(f"{ai_prefix}: {intro_out}")


def print_memory():
    print("MEMORY:")
    print("-short_term_memory:")
    pprint(short_term_memory.chat_memory.dict())
    print("-long_term_memory.nodes:")
    pprint(long_term_memory.kg._graph.nodes)


def save_memory():
    with open("short_term_memory.json", "w") as file_out:
        file_out.write(short_term_memory.chat_memory.json())
    long_term_memory.kg.write_to_gml("./long_term_memory.gml")


def print_help():
    print(
        f"""These Mimir command are not Part of story telling:
    :m - print memory
    :w - save memory to files
    :h, :help - print this help
    :q, quit, exit - end program
    """
    )


commands = {
    "--MEMORY": print_memory,
    ":m": print_memory,
    "--SAVE": save_memory,
    ":w": save_memory,
    "--HELP": print_help,
    ":help": print_help,
    ":h": print_help,
    ":q": exit,
}

exit_conditions = (":q", "quit", "exit")

print_help()
while True:
    query = input(f"{character_nick}> ")
    if query in exit_conditions:
        break
    if query in commands:
        commands[query]()
        continue
    pprint(f"{ai_prefix}: {conversation.predict(input=query)}")
