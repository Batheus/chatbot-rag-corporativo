import streamlit as st
import os
import logging
from dotenv import load_dotenv

st.set_page_config(page_title="Batheus.dev - Chatbot Corporativo", page_icon="🤖")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from prompts import RAG_PROMPT_TEMPLATE, CONTEXTUALIZE_Q_SYSTEM_PROMPT

load_dotenv()

st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .stButton button {background-color: #4CAF50; color: white;}
</style>
""", unsafe_allow_html=True)

# Título e barra lateral 
st.title("🤖 Chatbot de RH - Batheus.dev")
st.markdown("Tire suas dúvidas sobre o *Manual do Colaborador* usando IA.")

with st.sidebar:
    st.header("Sobre o Projeto")
    st.markdown("""
    Este chatbot utiliza **RAG (Retrieval-Augmented Generation)** para ler o PDF oficial da empresa Batheus.dev e responder perguntas com precisão.
    
    **Tecnologias Utilizadas:**
    - Python 3.13
    - LangChain (LCEL)
    - Google Gemini 2.5 Flash
    - FAISS (Vector Store)
    """)

#  Cache pra não recarregar o modelo a cada clique
@st.cache_resource
def get_vector_store():
    if not os.environ.get("GOOGLE_API_KEY"):
        return None
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_directory = "./faiss_db_index"
    
    if os.path.exists(persist_directory):
        return FAISS.load_local(
            folder_path=persist_directory, 
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    return None

def get_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = context_prompt | llm | StrOutputParser()
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    def contextualized_question(input_dict):
        if input_dict.get("chat_history"):
            return history_aware_retriever
        else:
            return input_dict["input"]
            
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            question=lambda x: x["input"]
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Qual o valor do reembolso de jantar?"):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Por favor, configure a GOOGLE_API_KEY no arquivo .env ou na barra lateral.")
        st.stop()

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    vectorstore = get_vector_store()
    if vectorstore:
        chain = get_rag_chain(vectorstore)
        
        with st.chat_message("assistant"):
            with st.spinner("Consultando o manual..."):
                try:
                    # Invoca a chain passando o histórico salvo na sessão do Streamlit
                    response = chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    st.markdown(response)
                    
                    # Atualiza histórico
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    from langchain_core.messages import HumanMessage, AIMessage
                    st.session_state.chat_history.extend([
                        HumanMessage(content=prompt),
                        AIMessage(content=response)
                    ])
                    
                except Exception as e:
                    st.error(f"Erro: {e}")
    else:
        st.error("Banco vetorial não encontrado. Verifique se a pasta 'faiss_db_index' está no repositório.")