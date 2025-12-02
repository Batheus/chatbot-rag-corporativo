# 🤖 Chatbot Corporativo com RAG (Retrieval-Augmented Generation)

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain_(LCEL)-1C3C3C?logo=langchain)
![Gemini](https://img.shields.io/badge/AI-Google_Gemini_2.5-8E75B2?logo=google)

Uma aplicação Full Stack de Inteligência Artificial projetada para democratizar o acesso à informação corporativa. Este chatbot transforma manuais estáticos (PDFs) em uma interface conversacional inteligente, capaz de responder dúvidas de colaboradores com precisão, citando fontes e mantendo o contexto da conversa.

🔗 **[Acesse a Demo Online](https://chatbot-rag-corporativo-batheusdev.streamlit.app/)**

---

## 🎯 O Problema vs. A Solução

**O Problema:** Manuais de RH e normas técnicas costumam ser documentos longos e densos. Encontrar uma informação específica (como "regras de reembolso" ou "configuração de VPN") exige tempo e gera atrito operacional.

**A Solução:** Um assistente virtual que utiliza **RAG (Retrieval-Augmented Generation)**. O sistema "lê" o documento oficial, busca os trechos relevantes para a pergunta do usuário e gera uma resposta baseada estritamente nesses dados, eliminando alucinações comuns em LLMs genéricos.

---

## 🛠️ Stack Tecnológica

O projeto foi desenvolvido focando em **modernidade** e **eficiência**:

* **Linguagem:** Python 3.13
* **Frontend:** [Streamlit](https://streamlit.io/) (Interface web interativa).
* **Orquestração de IA:** [LangChain](https://www.langchain.com/) utilizando **Pure LCEL (LangChain Expression Language)** para maior controle e modularidade.
* **LLM (Cérebro):** Google Gemini 2.5 Flash (Otimizado para baixa latência e raciocínio lógico).
* **Banco Vetorial:** FAISS (Facebook AI Similarity Search) para busca semântica local de alta performance.
* **Infraestrutura:** Deploy via Streamlit Community Cloud.

---

## ✨ Funcionalidades Chave

### 1. Memória Conversacional Inteligente
Diferente de sistemas de busca simples, este bot entende o contexto.
* **Usuário:** "Qual o notebook para desenvolvedores?"
* **Bot:** "É o MacBook Pro M3..."
* **Usuário:** "E para o RH?" (O bot entende que "E para..." se refere aos notebooks).

### 2. Anti-Alucinação (Grounding)
A engenharia de prompt restringe o modelo a responder **apenas** com base no contexto recuperado. Se a informação não estiver no PDF, o bot informa que não sabe, em vez de inventar.

### 3. Persistência de Dados
O índice vetorial (FAISS) é gerado e persistido em disco. Isso evita que o PDF precise ser reprocessado toda vez que a aplicação reinicia, garantindo um boot instantâneo.

---

## 🚀 Como Rodar Localmente

Siga os passos abaixo para executar o projeto na sua máquina:

### Pré-requisitos
* Python 3.10 ou superior.
* Uma API Key do Google AI Studio (Gemini).

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Batheus/chatbot-rag-corporativo.git](https://github.com/Batheus/chatbot-rag-corporativo.git)
    cd chatbot-rag-corporativo
    ```

2.  **Crie um ambiente virtual (Recomendado):**
    ```bash
    python -m venv .venv
    # Windows:
    .\.venv\Scripts\Activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as Variáveis de Ambiente:**
    Crie um arquivo `.env` na raiz do projeto e adicione sua chave:
    ```env
    GOOGLE_API_KEY="sua-chave-aqui"
    ```

5.  **Execute a aplicação:**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Estrutura do Projeto

```text
chatbot-rag-corporativo/
├── app.py                   # Frontend (Streamlit) e Lógica RAG
├── prompts.py               # Templates de Prompts (System Instructions)
├── faiss_db_index/          # Banco vetorial persistido (Embeddings)
├── Manual_Colaborador...pdf # Documento fonte
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação