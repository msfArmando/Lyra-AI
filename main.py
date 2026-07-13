from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_postgres import PGVectorStore, PGEngine
from langchain.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
#from langchain.agents import create_agent
import chainlit as cl
from typing import Any
import sys
import asyncio
#from langgraph.checkpoint.postgres import PostgresSaver
#import json
from pdf2image import convert_from_path
import pytesseract
from embedding import Embedding
from dotenv import load_dotenv
import edge_tts
import os
import re

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
TABLE_NAME = os.getenv("TABLE_NAME")
SCHEMA_NAME = os.getenv("SCHEMA_NAME")

CONNECTION_STRING = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
    f":{POSTGRES_PORT}/{POSTGRES_DB}"
)
MEMORY_CONNECTION_STRING = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
    f":{POSTGRES_PORT}/{POSTGRES_DB}"
)

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

llm: ChatOllama = ChatOllama(model="gpt-oss:20b", base_url="http://192.168.1.128:11434/")
embeddings: OllamaEmbeddings = OllamaEmbeddings(model="bge-m3", base_url="http://192.168.1.128:11434/")
VOICE = "pt-BR-ThalitaNeural"

connection_string: str = CONNECTION_STRING
memory_connection_string: str = MEMORY_CONNECTION_STRING

collection_name: str = "vectorstore"
vectorstore: PGVectorStore

def remove_special_caracteres(texto_markdown):
    regex_markdown = r'[\*#_\~`\[\]\(\)\-\+\>\!]+'
    texto_limpo = re.sub(regex_markdown, '', texto_markdown)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo)
    texto_limpo = texto_limpo.replace("<br>", " ")
    return texto_limpo.strip()

async def init_vectorstore():
    global vectorstore
    engine = PGEngine.from_connection_string(url=connection_string)
    cl.user_session.set("vectorstore_engine", engine)

    vectorstore = await PGVectorStore.create(
        engine=engine, embedding_service=embeddings, table_name=collection_name
    )

async def call_model(state: MessagesState):
    prompt = """
    You are Lyra, an personal assistant, equipped with analytical intelligence.
    You always communicate in portuguese pt-BR.
    You was development by Armando Monsão, a AI Engineer from Recife-PE, Brazil.
    Your temperament is calm, direct, and professional, yet you communicate in a friendly, encouraging, sympathetic, and occasionally informal manner.
    Never invent data. Be concise in your responses, focusing on solutions and suggestions for improvement.
    Critical Rules:

    * CONTEXT ONLY RESPONSES: Once the tool returns data, answer the user strictly and exclusively using facts found within that data. If the tool does not return any data, try to answer based on your own knowledge. 
    * NO HALLUCINATIONS: Do not invent or extrapolate any information that does not exist.

    Your workflow: Receive user input -> Evaluate if a tool call is necessary -> If yes: Call the required tool; If no: Answer with your own knowledge -> Respond based on the data returned from the tool. If no relevant content is found, answer using your own knowledge.
    """
    messages = [{"role": "system", "content": prompt}] + state["messages"]

    llm_with_tools = llm.bind_tools([search_database])
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

async def sould_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

async def init_agent(checkpointer):
    await init_vectorstore()  
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode([search_database]))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", sould_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    agent = workflow.compile(checkpointer=checkpointer)
    return agent

@cl.step(type="tool")
async def search_database(query: str) -> str:
    """
        General search tool in the vector database. 
        This tool should be used to look for information in the vector database whenever it's not possible to answer with your own knowledge.
        Args:
        Query: The exact, raw text input or question provided by the user.
    """
    print("Chamando ferramenta... 1")

    results = await vectorstore.asimilarity_search(query, k=4)
    if results:
        return results[0].page_content
    return "Nenhum resultado encontrado."

@cl.on_chat_start
async def on_chat_start():
    checkpointer_context = AsyncPostgresSaver.from_conn_string(memory_connection_string)
    checkpointer = await checkpointer_context.__aenter__()

    await checkpointer.setup()
    
    cl.user_session.set("checkpointer_context", checkpointer_context)
    cl.user_session.set("checkpointer", checkpointer)

    chatagent = await init_agent(checkpointer)
    cl.user_session.set("chatagent", chatagent)


@cl.on_message
async def on_message(message: cl.Message):

    complete_text: str = ""
    format_elementname: str = ""
    
    if message.elements:
        async with cl.Step(name="registro de informações na base de dados"):
            for element in message.elements:
                
                filepath = element.path
                images = convert_from_path(filepath, poppler_path="poppler-26.02.0/library/bin")

                for i, image in enumerate(images):
                    text_page = pytesseract.image_to_string(image, lang="por")
                    if text_page:
                        complete_text = f"--- Página {i+1} ---\n{text_page}"

                        elementname = element.name
                        format_elementname, _ = os.path.splitext(elementname)

                        await Embedding(complete_text, format_elementname)

    chatagent: Any = cl.user_session.get("chatagent")
    thread_config = {"configurable": {"thread_id": cl.user_session.get("id")}}

    if not message.content and complete_text:
        print("Não tem mensagem, mas tem arquivo.")
        result = await chatagent.ainvoke(
            {"messages": [{"role": "user", "content": f"{message.content}"}]},
            thread_config
        )
    elif message.content and complete_text:
        print("Tem mensagem e tem arquivo")
        result = await chatagent.ainvoke(
            {"messages": [{"role": "user", "content": f"{message.content}. Use the search database tool."}]},
            thread_config
        )
    elif message.content and not complete_text:
        print("Só tem mensagem")
        result = await chatagent.ainvoke(
            {"messages": [{"role": "user", "content": f"{message.content}"}]},
            thread_config
        )

    ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]

    if ai_messages: 
        response = ai_messages[-1].content
        print(response)
    else:
        response = "Desculpe, ocorreu um erro ao processar a resposta."

    msg = await cl.Message(
        content=response,
        author='Aurora'
    ).send()

    audiopath = f"audiofiles/response_{message.id}.mp3"

    formated_response = remove_special_caracteres(response)

    communicate = edge_tts.Communicate(formated_response, VOICE)
    await communicate.save(audiopath)

    audioelement = cl.Audio(
        name="Aurora voice",
        path=audiopath,
        display="inline"
    )
    
    msg.elements = [audioelement]
    await msg.update()

@cl.on_chat_end
async def end():
    path = "audiofiles"

    for nome_arquivo in os.listdir(path):
        caminho_completo = os.path.join(path, nome_arquivo)
        if os.path.isfile(caminho_completo):
            os.remove(caminho_completo)
            print(f'Arquivo {nome_arquivo} excluído com sucesso.')


    checkpointer_context = cl.user_session.get("checkpointer_context")
    if checkpointer_context:
        try:
            await checkpointer_context.__aexit__(None, None, None)
        except Exception as e:
            print(f"Erro ao fechar o checkpointer: {e}")

    engine = cl.user_session.get("vectorstore_engine")
    if engine:
        try:
            if hasattr(engine, "aclose"):
                await engine.aclose()
            else:
                await engine.close()
        except Exception as e:
            print(f"Erro ao fechar o PGEngine: {e}")
