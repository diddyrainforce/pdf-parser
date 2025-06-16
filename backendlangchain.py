import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# 🔐 Charger les variables d’environnement              .\env\Scripts\Activate        uvicorn backendlangchain:app --reload
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
#print(openai_api_key)

# Chemin vers Chroma (il doit déjà être initialisé)
CHROMA_PATH = "./chroma_db"

# Initialiser le modèle GPT et l'API
chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
app = FastAPI()

# Initialisation du retriever   
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
retriever = vectordb.as_retriever()

# Autoriser le frontend Loveable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou "https://loveable.io" si hébergé
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#CHATBOT####################################################################################################################################
@app.post("/chat")
async def chat_with_bot(request: Request):
    body = await request.json()
    user_message = body.get("message")
    print(user_message)
    response = chat([HumanMessage(content=user_message)])
    return {"reply": response.content}
##########################################################################################################################################

#PROPOSALASSISTANT#########################################################################################################################################
# Prompt personnalisé
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Tu es un assistant expert en rédaction de projets européens (Horizon Europe, Eurostars, etc.).

Utilise les informations ci-dessous pour rédiger une **proposition structurée et professionnelle** en français. Structure le document avec des sections claires, titres en gras, et un style convaincant.

== Contexte utile (exemples de projets précédents) ==
{context}

== Informations utilisateur ==
{question}

== Consignes ==
- Commence par un résumé exécutif.
- Ajoute ensuite les sections suivantes :
  1. Contexte et objectifs du projet
  2. Description technique
  3. Partenaires et rôles
  4. Budget et plan de financement
- Utilise un langage formel mais accessible.
- Ne répète pas textuellement les entrées utilisateur, reformule.
"""
)

# Chaine RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False
)

@app.post("/generate-pdf")
async def generate_proposal(request: Request):
    data = await request.json()

    user_input = f"""Basic Info: {data['basic_info']}
    Technical: {data['technical']}
    Partners: {data['partners']}
    Budget: {data['budget']}"""

    rag_response = rag_chain.invoke({'query' : user_input})
    print(rag_response) 
    return {"proposal_text": rag_response}
##########################################################################################################################################