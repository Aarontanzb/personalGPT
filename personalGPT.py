from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

def load_model():
    '''
    download Llama model for the first run. 
    Subsequent runs will use the downloaded model.
    '''
    model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"

    tokenizer = LlamaTokenizer.from_pretrained(model_id, cache_dir="./models/")
    model = LlamaForCausalLM.from_pretrained(model_id, cache_dir="./models/")

    hf_pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=2048,
        temperature=0,
        top_p=0.75,
        repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=hf_pipe)

    return local_llm


def main():
    # load the instructorEmbeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                model_kwargs={"device": "cuda"})
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # load the LLM for generating Natural Language responses. 
    llm = load_model()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        print("----------------------------------SOURCE DOCUMENTS---------------------------")

if __name__ == "__main__":
    main()