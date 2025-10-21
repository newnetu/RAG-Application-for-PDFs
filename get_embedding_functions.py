from langchain_aws import BedrockEmbeddings
from botocore.config import Config

def get_embeddings_function():
    boto_config = Config(
        retries = dict(
            max_attempts = 3
        ),
        connect_timeout = 30,
        read_timeout = 30
    )
    
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", 
        region_name="us-east-1",
        model_id="amazon.titan-embed-text-v1", 
        config=boto_config
    )
    return embeddings
