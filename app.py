import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Importações das ferramentas do ecossistema LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA (STREAMLIT)
# ==============================================================================
st.set_page_config(
    page_title="Voz em Foco - Assistente Virtual", page_icon="🎙️", layout="centered"
)


# ==============================================================================
# CARREGAMENTO DO PIPELINE RAG (COM CACHE DE ALTA PERFORMANCE)
# ==============================================================================
@st.cache_resource
def inicializar_pipeline_rag():
    # ETAPA 1 & 2: Processamento e Extração de Conteúdo
    loader = PyPDFDirectoryLoader("./documentos")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(docs)

    # ETAPA 3: Indexação Vetorial
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", transport="rest"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ETAPA 4: Camada de Recuperação
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    # ETAPA 5: GERAÇÃO E VALIDAÇÃO DE RESPOSTAS
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", transport="rest", temperature=0.1
    )

    system_prompt = (
        "Você é o assistente virtual oficial da escola de canto Voz em Foco.\n"
        "Sua missão é responder às dúvidas dos alunos e interessados de forma clara, "
        "educada, acolhedora e profissional.\n\n"
        "IMPORTANTE: Você atende exclusivamente sobre a escola Voz em Foco e seus cursos. "
        "Quando o usuário disser 'o curso', 'a escola' ou fizer uma pergunta sem indicar outro contexto, "
        "entenda que ele está falando da Voz em Foco. Não pergunte qual curso ou instituição ele quer dizer "
        "em perguntas gerais; responda usando os documentos fornecidos. Só peça esclarecimento se os próprios "
        "documentos apresentarem informações diferentes para cursos específicos e isso mudar a resposta.\n\n"
        "Regras de Ouro e Validação:\n"
        "1. Responda APENAS com base no contexto fornecido abaixo. NÃO invente e não use conhecimentos externos.\n"
        "2. OBRIGATÓRIO CITAR FONTES: Ao final da resposta, liste as fontes consultadas "
        "no formato: 'Fonte: [Nome do Arquivo], Página X'.\n"
        "3. FALLBACK (Quando não souber): Se a informação não estiver descrita no contexto, diga claramente: "
        "'Desculpe, não encontrei essa informação nos nossos documentos oficiais. "
        "Por favor, entre em contato com a Secretaria do Voz em Foco pelo e-mail suporte@vozemfoco.com.br "
        "ou WhatsApp (11) 99999-8888.'\n\n"
        "Contexto dos Documentos:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


# Inicializa o pipeline RAG apenas uma vez na memória do servidor
rag_chain = inicializar_pipeline_rag()

# ==============================================================================
# ETAPA 6: IMPLANTAÇÃO E INTERFACE WEB (STREAMLIT)
# ==============================================================================
st.title("🎙️ Assistente Virtual - Voz em Foco")
st.caption("🤖 Agente Oficial de Atendimento e Consulta aos Documentos")

# Inicialização do Histórico de Mensagens na Sessão
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Sou o assistente virtual da Voz em Foco. Como posso ajudar com dúvidas sobre os cursos, regras e horários?",
        }
    ]

# Renderização do Histórico no Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Caixa de Entrada Interativa
if pergunta := st.chat_input("Digite sua dúvida aqui..."):
    # Exibe a pergunta do usuário no chat e salva na sessão
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Processamento e Resposta do Agente
    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conhecimento oficial..."):
            pergunta_contextualizada = (
                "Sobre a escola de canto Voz em Foco e seus cursos, responda: "
                + pergunta
            )
            resposta = rag_chain.invoke({"input": pergunta_contextualizada})
            conteudo_resposta = resposta["answer"]

            st.markdown(conteudo_resposta)

    # Salva a resposta do assistente no histórico
    st.session_state.messages.append(
        {"role": "assistant", "content": conteudo_resposta}
    )
