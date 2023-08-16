import streamlit as st
import requests
from dotenv import load_dotenv
import os

from langchain.embeddings.openai import OpenAIEmbeddings
import os
# import config
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS

load_dotenv()

# Streamlit UI
def main():
    st.title("Medichatbot ")
    # st.markdown("Number of Models Used: 5")

    # Create an input text box
    prompt = st.text_input("Enter text")

    # Create a send button
    if st.button("Send"):
     
        reader = PdfReader('./content/brochure.pdf')
        os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API")
            # read data from the file and put them into a variable called raw_text
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.

        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 100,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()

        docsearch = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        # query = "tell me about PERIOFLOW APPLICATIONS"
        query=prompt
        docs = docsearch.similarity_search(query)
        ans=chain.run(input_documents=docs, question=query)
        print(ans)
        st.write("your answer :", ans)
    

if __name__ == "__main__":
    main()
