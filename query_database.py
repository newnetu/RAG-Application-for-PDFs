import argparse
import boto3
import json
from langchain_community.vectorstores import Chroma
from get_embedding_functions import get_embeddings_function

CHROMA_PATH = "chroma"

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")

def main():
    # Create CLI:
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB for the input
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings_function())

    # Search the DB - k=3 - finds top three results with regards to query
    results = db.similarity_search(query_text, k=3)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])  

    # Create the system and message lists for Nova
    system_list = [{
        "text": "You are a helpful assistant. Answer questions based only on the provided context."
    }]  

    message_list = [
        {
            "role":"user",
            "content":[{
                "text": f"""
                Context: {context_text}

                Question: {query_text}

                Answer the question based only on the context provided above.
                """
            }]
        }
    ]

    # Inference Parameters:
    inf_params = {
        "maxTokens": 500,
        "topP": 0.9,
        "topK": 20,
        "temperature": 0.7
    }

    # Request Body
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }

    #Invoking model
    client = get_bedrock_client()
    response = client.invoke_model_with_response_stream(
        modelId= "amazon.nova-lite-v1:0",
        body = json.dumps(request_body)
    )

    # Response stream processing
    stream = response.get("body")
    full_response = ""

    if stream:
        for event in stream:
            chunk = event.get("chunk")

            if chunk:
                chunk_json = json.loads(chunk.get("bytes").decode())
                content_block_delta = chunk_json.get("contentBlockDelta")
                if content_block_delta:
                    text_delta = content_block_delta.get("delta", {}).get("text", "")
                    full_response += text_delta

                    # Print streaming response
                    print(text_delta,end="") 
    else:
        print("No response stream received")

    # Print Sources:
    sources = [doc.metadata.get("source", None) for doc in results]
    for source in sources:
        print("\n",source)



if __name__ == "__main__":
    main()