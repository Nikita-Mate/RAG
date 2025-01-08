
##Preliminary and dependencies
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
import streamlit as st
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from langchain.schema.runnable import RunnableMap
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

#Loading secrets
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

#Inititalizing Embeddings model
embeddings = AzureOpenAIEmbeddings(azure_deployment="deployment_name",
                                    openai_api_version="api_version")


##Function for formating loaded documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


##Function for reading files and generating knowlege index in local vector store
def file_loader(document):
    #Reading file text onto 'text' using PyPDF2 - PdfReader
    text=""
    
    pdf_reader=PdfReader(document)
    
    for page in pdf_reader.pages:
        text+=page.extract_text()

    #Initilizing langchain CharacterTextSplitter
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len)   
    chunks=text_splitter.split_text(text)

    #Storing into vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("temp-index")


##Function for generating answer based on similarity search of the knowlege index
def chatbot_short(query: str):
    
    #Getting a retriever of the vector store
    folder = os.path.join(os.getcwd(), "temp-index")
    knowledge_index = FAISS.load_local(folder_path=folder, index_name="index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    retriever = knowledge_index.as_retriever(search_type='similarity', search_kwargs={'k':3})

    #Building a RAG prompt
    prompt = ChatPromptTemplate(input_variables=['context', 'question'], 
                                metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt',
                                        'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'},
                                messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], 
                                                                                        template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer then answer normally while informing that you can answer from the read PDF, try to output as bullet points whenever possible. Use two-three sentences maximum, and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])

    #Azure OpenAI model 
    llm = AzureChatOpenAI(openai_api_version="2024-05-01-preview",
                        azure_deployment="deployment_name",
                        temperature=0.2)

    #Building and invoking the RAG chain 
    rag_chain_from_docs = ({'context': lambda input: format_docs(input['documents']),
                            'question': itemgetter('question')}
                            | prompt | llm | StrOutputParser())

    rag_chain_with_source = RunnableMap({'documents': retriever,
                                        'question': RunnablePassthrough()}) | {'documents': lambda input: [doc.metadata for doc in input['documents']],
                                                                                'answer': rag_chain_from_docs}
    invoked_dict = rag_chain_with_source.invoke(query)

    return invoked_dict

def chatbot_long(query: str):
    
    #Getting a retriever of the vector store
    folder = os.path.join(os.getcwd(), "temp-index")

    knowledge_index = FAISS.load_local(folder_path=folder, index_name="index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    retriever = knowledge_index.as_retriever(search_type='similarity', search_kwargs={'k':3})

    #Building a RAG prompt
    prompt = ChatPromptTemplate(input_variables=['context', 'question'], 
                                metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt',
                                        'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'},
                                messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], 
                                                                                        template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer then answer normally while informing that you can answer from the read PDF, try to output as bullet points whenever possible. Use five-six sentences maximum.\nQuestion: {question} \nContext: {context} \nAnswer:"))])

    #Azure OpenAI model 
    llm = AzureChatOpenAI(openai_api_version="2024-05-01-preview",
                        azure_deployment="gpt35neo",
                        temperature=0.2)

    #Building and invoking the RAG chain 
    rag_chain_from_docs = ({'context': lambda input: format_docs(input['documents']),
                            'question': itemgetter('question')}
                            | prompt | llm | StrOutputParser())

    rag_chain_with_source = RunnableMap({'documents': retriever,
                                        'question': RunnablePassthrough()}) | {'documents': lambda input: [doc.metadata for doc in input['documents']],
                                                                                'answer': rag_chain_from_docs}
    invoked_dict = rag_chain_with_source.invoke(query)

    return invoked_dict

##Main method containing streamlit application UI 
def main():
    
    #setting up page configuration
    st.set_page_config(page_title="DocDive", layout='wide')

    st.title("Sample Set Doc Dive")
    st.subheader("(AI Powered PDF Querying Chatbot)\n")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.chat_message("assistant"):
        st.markdown("<h3 style='text-align: center;'>Hi! How can I help you today?</h3>\n\nI am an AI Chatbot capable of answering questions from the uploaded PDF",
                    unsafe_allow_html=True)

    #Setting up chat elements for assistant and user
    for message in st.session_state.messages:
        if message['role'] == 'assistant':
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        else:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    #Sidebar for providing document that is to be read - passing to file_loader() function
    with st.sidebar:
        st.subheader("Upload PDF File")
        doc = st.file_uploader("Upload file here, and click on  the 'Load File' button", accept_multiple_files=False)
        if st.button("Load File"):
            with st.spinner("Loading file..."):
                file_loader(doc)

        ans_type = st.radio("Answer Type:", ['Concise', 'Detailed'], captions=['Shorter, more summarised answers', 'Longer, more detailed answers'])

    #Taking user prompt - passing to chatbot() function
    if prompt := st.chat_input("Enter your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        if ans_type == "Concise":
            response_dict = chatbot_short(prompt)
            response = response_dict['answer']
        else:
            response_dict = chatbot_long(prompt)
            response = response_dict['answer']

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            st.markdown(response)

            #Appending messages so user can see chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
