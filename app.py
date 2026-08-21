import os
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
# ETAPA 1: COLETA E ORGANIZAÇÃO DE DOCUMENTOS
# Os 5 arquivos PDF foram organizados e salvos na pasta ./documentos
# ==============================================================================


# ==============================================================================
# ETAPA 2: PROCESSAMENTO E EXTRAÇÃO DE CONTEÚDO
# ==============================================================================

# Aponta para a pasta e lê todos os arquivos PDF presentes nela
loader = PyPDFDirectoryLoader("./documentos")
docs = loader.load()

# Divisão do texto em pedaços (chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Cada pedaço terá até 1000 caracteres
    chunk_overlap=150,  # 150 caracteres repetidos entre blocos
)

chunks = text_splitter.split_documents(docs)


# ==============================================================================
# ETAPA 3: INDEXAÇÃO VETORIAL
# ==============================================================================

# Força o uso do transporte HTTP (REST) com o modelo oficial de embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", transport="rest"
)

# Cria o banco vetorial FAISS salvando os chunks e seus respectivos vetores na memória
vectorstore = FAISS.from_documents(chunks, embeddings)

# ==============================================================================
# ETAPA 4: CAMADA DE RECUPERAÇÃO (RAG)
# ==============================================================================

# Configura o retriever para realizar a busca semântica no banco vetorial
# Definimos k=5 para resgatar os 5 trechos mais relevantes do contexto
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# ==============================================================================
# ETAPA 5: GERAÇÃO E VALIDAÇÃO DE RESPOSTAS
# ==============================================================================

# 5.1 - Inicializa o LLM Gemini com temperatura baixa para respostas precisas
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", transport="rest", temperature=0.1
)

# 5.2, 5.3 & 5.4 - Prompt do Sistema com regras de citação, fallback e validação
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

# 5.5 - Montagem da cadeia RAG (Junta o LLM ao Prompt de Contexto)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ==============================================================================
# ETAPA 6: IMPLANTAÇÃO E INTERFACE (CHAT DE ATENDIMENTO)
# ==============================================================================
print("🚀 ETAPA 6: Inicializando a Interface de Atendimento...")


def executar_interface_chat():
    print("\n" + "=" * 60)
    print("🎙️ ASSISTENTE VIRTUAL - VOZ EM FOCO (IA OFICIAL)")
    print("Ambiente de Atendimento e Consulta aos Documentos")
    print("Digite 'sair' a qualquer momento para encerrar a sessão.")
    print("=" * 60 + "\n")

    while True:
        # 6.1 - Campo de entrada da pergunta do colaborador/aluno
        pergunta = input("👤 Aluno/Colaborador: ")

        # Comando para encerrar o chat
        if pergunta.strip().lower() in ["sair", "exit", "quit"]:
            print("\n👋 Atendimento encerrado. Sessão finalizada com sucesso!")
            break

        if not pergunta.strip():
            continue

        print("🤖 Agente pensando e consultando base vetorial...")

        # Mantém o contexto da escola mesmo quando o usuário faz uma pergunta curta
        pergunta_contextualizada = (
            "Sobre a escola de canto Voz em Foco e seus cursos, responda: " + pergunta
        )
        resposta = rag_chain.invoke({"input": pergunta_contextualizada})

        # 6.2 - Exibição clara da resposta com a indicação do agente de IA
        print(f"\n🎙️ Resposta do Agente (Voz em Foco):\n{resposta['answer']}\n")
        print("-" * 60)


# Executa o chat quando o arquivo é chamado diretamente
if __name__ == "__main__":
    executar_interface_chat()
