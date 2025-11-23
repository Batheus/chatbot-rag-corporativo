# Chatbot Corporativo RAG

Chatbot corporativo RAG (Retrieval-Augmented Generation) em Python para
responder perguntas sobre manuais internos (PDF), citando fontes e
realizando cálculos simples quando permitido. Implementação modular que
usa LangChain + Google Generative API (Gemini) para geração e
GoogleGenerativeAIEmbeddings para embeddings; armazenamento em memória
via SKLearnVectorStore.

## Funcionalidades principais

-   Carregamento e divisão de PDF em chunks.
-   Geração de embeddings com GoogleGenerativeAIEmbeddings.
-   Recuperação de trechos relevantes e respostas com Gemini.
-   Citações de fontes obrigatórias.
-   Permite cálculos simples quando o manual contém valores base.

## Instalação

``` bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate.bat  # Windows
pip install -r requirements.txt
```

## Configuração

Defina a variável de ambiente da API Key:

``` bash
export GOOGLE_API_KEY="sua_chave"
```

## Uso

``` bash
python rag_chatbot.py
```

## Estrutura técnica

-   Loader: PyPDFLoader
-   Splitter: RecursiveCharacterTextSplitter
-   Embeddings: GoogleGenerativeAIEmbeddings
-   Vector Store: SKLearnVectorStore
-   LLM: ChatGoogleGenerativeAI (gemini-2.5-flash)

## Prompt

O template impõe: 
- Uso exclusivo do contexto do manual;
- Citações obrigatórias;
- Cálculos permitidos apenas quando baseados no PDF.

## Licença

MIT
