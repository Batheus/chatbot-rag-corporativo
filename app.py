import streamlit as st
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Batheus.dev - Chatbot Corporativo", page_icon="🤖")
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .stButton button {background-color: #4CAF50; color: white;}
    
    /* Ajuste GLOBAL para links em títulos */
    h1 a {
        color: inherit !important;
        text-decoration: none !important;
        border-bottom: 1px dashed #666;
    }
    
    h1 a:hover {
        color: inherit !important;
        text-decoration: none !important;
        border-bottom: 2px solid #4CAF50;
        opacity: 0.8;
    }

    p a {
        color: inherit !important;
        text-decoration: none !important;
        border-bottom: 1px dashed #666;
    }
    
    p a:hover {
        color: inherit !important;
        text-decoration: none !important;
        border-bottom: 2px solid #4CAF50;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from prompts import RAG_PROMPT_TEMPLATE, CONTEXTUALIZE_Q_SYSTEM_PROMPT

load_dotenv()

with st.sidebar:
    st.title("🤖 Chatbot RH")
    st.markdown("Tire suas dúvidas sobre o *Manual do Colaborador* usando IA.")
    st.markdown("---")
    st.header("📄 Documentação")
    st.markdown("Sinta-se à vontade para abrir o Manual do Colaborador para mais informações e poder fazer perguntas sobre o conteúdo.")
    pdf_url = "https://github.com/Batheus/chatbot-rag-corporativo/blob/main/Manual_Colaborador_BatheusDev.pdf"
    st.link_button("📥 Abrir Manual (PDF)", pdf_url)
    st.markdown("---")
    st.header("Sobre o Projeto")
    st.markdown("""
    Este chatbot utiliza **RAG (Retrieval-Augmented Generation)** para responder perguntas com precisão.
    
    **Tecnologias Utilizadas:**
    - Python 3.13
    - LangChain (LCEL)
    - Google Gemini 2.5 Flash
    - FAISS (Vector Store)
    """)
    st.markdown("---")
    st.caption("Desenvolvido por [batheus.dev](https://batheus.dev)")

@st.cache_resource
def get_vector_store():
    if not os.environ.get("GOOGLE_API_KEY"): return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("./faiss_db_index"):
        return FAISS.load_local("./faiss_db_index", embeddings, allow_dangerous_deserialization=True)
    return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.4, "k": 3}
    )
    
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
            docs=contextualized_question | retriever,
            question=lambda x: x["input"]
        )
        .assign(context=lambda x: format_docs(x["docs"]))
        .assign(answer=qa_prompt | llm | StrOutputParser())
    )
    return rag_chain

st.markdown("# 💬 Assistente Corporativo | [batheus.dev](https://batheus.dev)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua dúvida..."):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("A API Key do Google não foi configurada nos Secrets do Streamlit.")
        st.stop()

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    vectorstore = get_vector_store()
    if vectorstore:
        chain = get_rag_chain(vectorstore)
        
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    response = chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer_text = response['answer']
                    
                    st.markdown(answer_text)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer_text
                    })
                    st.session_state.chat_history.extend([
                        HumanMessage(content=prompt),
                        AIMessage(content=answer_text)
                    ])
                    
                except Exception as e:
                    st.error(f"Erro: {e}")
    else:
        st.error("Banco de dados vetorial não encontrado.")