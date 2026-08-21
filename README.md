# 🎙️ Assistente Virtual - Voz em Foco (RAG com Gemini & LangChain)

Um agente inteligente de atendimento e consulta aos documentos oficiais da escola de canto **Voz em Foco**. A aplicação utiliza a arquitetura **RAG (Retrieval-Augmented Generation)** para responder a dúvidas de alunos e colaboradores com base estrita nos manuais, políticas e guias da instituição.

---

## 📌 Funcionalidades

- **Consulta Inteligente a PDFs:** Processamento e busca vetorial sobre os documentos oficiais presentes na base de conhecimento.
- **Respostas Precisas e sem Alucinação:** Respostas geradas exclusivamente com base no contexto fornecido, prevenindo a invenção de dados externos.
- **Rastreabilidade (Citação de Fontes):** Indicação obrigatória do nome do arquivo fonte e da página consultada ao final de cada resposta.
- **Canal de Fallback:** Encaminhamento direto para a Secretaria quando a informação não estiver presente nos documentos.
- **Interface Web Interativa:** Chat responsivo desenvolvido em Streamlit com histórico de mensagens em tempo real.

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Framework RAG:** LangChain
- **Modelo de Linguagem (LLM):** Google Gemini 3.6 Flash (`langchain-google-genai`)
- **Embeddings:** Google Generative AI Embeddings (`models/gemini-embedding-001`)
- **Banco Vetorial:** FAISS (`faiss-cpu`)
- **Interface Gráfica:** Streamlit
- **Deploy na Nuvem:** Streamlit Community Cloud

---

## 📐 Arquitetura do Pipeline RAG

1. **Ingestão de Documentos:** Carregamento dinâmico dos arquivos em `./documentos` via `PyPDFDirectoryLoader`.
2. **Chunking (Divisão de Texto):** Fragmentação dos textos em blocos de até 1000 caracteres com sobreposição de 150 caracteres usando `RecursiveCharacterTextSplitter`.
3. **Indexação Vetorial:** Conversão dos trechos em embeddings vetoriais e armazenamento no FAISS com cache de alta performance (`@st.cache_resource`).
4. **Camada de Recuperação (Retriever):** Busca por similaridade vetorial resgatando os 5 trechos mais relevantes ($k=5$) para cada pergunta.
5. **Geração e Validação de Respostas:** Processamento do LLM Gemini com *System Prompt* rigoroso focado na escola Voz em Foco, com trava de segurança para ausência de dados e citação de fontes.
6. **Interface de Atendimento:** Aplicação web interativa no Streamlit com estado de sessão mantido.

---

## 📂 Estrutura do Repositório

```text
voz-em-foco-agent/
│
├── documentos/          # Base de conhecimento (PDFs oficiais da escola)
├── app.py               # Aplicação principal (Pipeline RAG + Interface Streamlit)
├── .env                 # Variáveis de ambiente locais (chave da API)
├── .gitignore           # Arquivos e pastas ignorados pelo Git
├── requirements.txt     # Dependências para deploy no Streamlit Cloud
└── README.md            # Documentação oficial do projeto
```

## 🚀 Como Executar o Projeto Localmente
Pré-requisitos

1. `Python 3.10 ou superior`

2. `Chave de API da Google AI Studio (GOOGLE_API_KEY)`

Passo a Passo

1. Clone o repositório
```
Bash

git clone [https://github.com/luaramaciel/voz-em-foco-agent.git](https://github.com/luaramaciel/voz-em-foco-agent.git)
cd voz-em-foco-agent
```

2. Crie e ative um ambiente virtual
```
Bash

# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências
```
Bash

pip install -r requirements.txt
```

4. Configure as Variáveis de Ambiente

Crie um arquivo .env na raiz do projeto contendo a sua chave da API:

```
Bash

GOOGLE_API_KEY="sua_chave_aqui"
```

5. Inicie a aplicação Web:
```
Bash

streamlit run app.py
```
Acesse a interface no navegador em http://localhost:8501.

## 💬 Exemplos de Uso
Consulta com Citação de Fonte

    Usuário: "Como funcionam os treinos de canto e a política de faltas?"

    Agente: "As aulas e treinos ocorrem conforme a grade do curso... \n\nFonte: Guia_de_Treinos.pdf, Página 2"

Fallback para Assuntos fora do Escopo

    Usuário: "Vocês vendem instrumentos musicais na recepção?"

    Agente: "Desculpe, não encontrei essa informação nos nossos documentos oficiais. Por favor, entre em contato com a Secretaria do Voz em Foco pelo e-mail suporte@vozemfoco.com.br ou WhatsApp (11) 99999-8888."

## 🎥 Demonstração em Vídeo

Assista ao vídeo demonstrativo do funcionamento do Assistente Virtual, exibindo a busca semântica em tempo real, citação de fontes e a regra de segurança/fallback:

▶️ Clique aqui para assistir à demonstração em vídeo


## 👩🏾‍💻 Autora do Projeto

Desenvolvido por Luara Maciel como projeto prático de Inteligência Artificial e arquitetura RAG.

    GitHub: @luaramaciel

    Projeto: Voz em Foco Agent

---

## 📄 Licença

Este projeto está sob a licença [MIT](LICENSE). Consulte o arquivo `LICENSE` para mais detalhes.

