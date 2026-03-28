"""Custom Resource Lambda to create OpenSearch Serverless vector index.

Called by CDK Custom Resource after the AOSS collection is ACTIVE.
Creates the knn vector index with the exact mapping Bedrock KB expects.
"""

import json
import time

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth


def on_event(event, context):
    """CloudFormation Custom Resource entry point."""
    print(json.dumps(event))
    request_type = event["RequestType"]
    if request_type == "Create":
        return on_create(event)
    if request_type == "Update":
        return on_create(event)  # idempotent — recreate if missing
    if request_type == "Delete":
        return on_delete(event)
    raise ValueError(f"Invalid request type: {request_type}")


def on_create(event):
    props = event["ResourceProperties"]
    host = props["AOSSHost"].replace("https://", "")
    index_name = props["AOSSIndexName"]
    dimensions = int(props.get("Dimensions", "1024"))

    client = _get_client(host)

    # Wait briefly for collection to be fully ready
    for attempt in range(5):
        try:
            if client.indices.exists(index=index_name):
                print(f"Index {index_name} already exists, skipping creation")
                return {"Data": {"IndexName": index_name, "Status": "EXISTS"}}
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}: Collection not ready yet: {e}")
            time.sleep(5)

    index_body = {
        "settings": {
            "index.knn": True,
        },
        "mappings": {
            "properties": {
                "bedrock-knowledge-base-default-vector": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "faiss",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16,
                        },
                    },
                },
                "AMAZON_BEDROCK_TEXT_CHUNK": {
                    "type": "text",
                    "index": True,
                },
                "AMAZON_BEDROCK_METADATA": {
                    "type": "text",
                    "index": False,
                },
            }
        },
    }

    response = client.indices.create(index=index_name, body=index_body)
    print(f"Created index {index_name}: {response}")
    return {"Data": {"IndexName": index_name, "Status": "CREATED"}}


def on_delete(event):
    """No-op on delete — let AOSS collection deletion handle cleanup."""
    return {"PhysicalResourceId": event.get("PhysicalResourceId", "kb-index")}


def _get_client(host):
    credentials = boto3.Session().get_credentials()
    region = boto3.session.Session().region_name
    auth = AWSV4SignerAuth(credentials, region, "aoss")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
