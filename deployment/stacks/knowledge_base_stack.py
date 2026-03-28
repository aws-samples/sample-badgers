"""Knowledge Base Stack for BADGERS.

Creates:
- S3 bucket for KB content (main_content/ prefix as data source)
- OpenSearch Serverless collection for vector storage
- Bedrock Knowledge Base with Titan Embed V2, no chunking, 1024-dim float vectors
- S3 data source pointing to main_content/
"""

import json
from pathlib import Path

from aws_cdk import (
    Stack,
    CfnOutput,
    CustomResource,
    Duration,
    RemovalPolicy,
    Tags,
    aws_s3 as s3,
    aws_kms as kms,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_bedrock as bedrock,
    aws_opensearchserverless as aoss,
    custom_resources,
)
from constructs import Construct


class KnowledgeBaseStack(Stack):
    """Stack for Bedrock Knowledge Base with OpenSearch Serverless."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deployment_id: str,
        deployment_tags: dict[str, str],
        s3_kms_key: kms.IKey,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.deployment_id = deployment_id
        self.deployment_tags = deployment_tags

        id_prefix = "badgers"
        kb_name = f"{id_prefix}-knowledge-base-{deployment_id}"
        collection_name = f"{id_prefix}-kb-{deployment_id}"  # max 32 chars
        index_name = "kb-vectors"

        # Apply common tags
        for key, value in deployment_tags.items():
            Tags.of(self).add(key, value)
        # ---------------------------------------------------------------
        # S3 bucket for knowledge base content
        # ---------------------------------------------------------------
        self.kb_bucket = s3.Bucket(
            self,
            "KBContentBucket",
            bucket_name=f"{id_prefix}-knowledge-base-{deployment_id}",
            versioned=True,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=s3_kms_key,
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        Tags.of(self.kb_bucket).add("resource_name", "kb-content-bucket")
        Tags.of(self.kb_bucket).add(
            "resource_description", "S3 bucket for knowledge base content"
        )

        # ---------------------------------------------------------------
        # IAM role for Bedrock KB service
        # ---------------------------------------------------------------
        self.kb_role = iam.Role(
            self,
            "KBServiceRole",
            role_name=f"{id_prefix}-kb-role-{deployment_id}",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            inline_policies={
                "BedrockKBPolicy": iam.PolicyDocument(
                    statements=[
                        # S3 read access for data source
                        iam.PolicyStatement(
                            actions=[
                                "s3:GetObject",
                                "s3:ListBucket",
                            ],
                            resources=[
                                self.kb_bucket.bucket_arn,
                                f"{self.kb_bucket.bucket_arn}/*",
                            ],
                        ),
                        # KMS decrypt for the S3 bucket
                        iam.PolicyStatement(
                            actions=[
                                "kms:Decrypt",
                                "kms:GenerateDataKey",
                            ],
                            resources=[s3_kms_key.key_arn],
                        ),
                        # Bedrock embedding model access
                        iam.PolicyStatement(
                            actions=["bedrock:InvokeModel"],
                            resources=[
                                f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0",
                            ],
                        ),
                    ]
                ),
            },
        )
        # ---------------------------------------------------------------
        # OpenSearch Serverless encryption policy (required before collection)
        # ---------------------------------------------------------------
        encryption_policy = aoss.CfnSecurityPolicy(
            self,
            "KBEncryptionPolicy",
            name=f"{id_prefix}-kb-enc-{deployment_id}",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{collection_name}"],
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )

        # Network policy — allow public access for Bedrock service
        network_policy = aoss.CfnSecurityPolicy(
            self,
            "KBNetworkPolicy",
            name=f"{id_prefix}-kb-net-{deployment_id}",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                            },
                            {
                                "ResourceType": "dashboard",
                                "Resource": [f"collection/{collection_name}"],
                            },
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )

        # ---------------------------------------------------------------
        # OpenSearch Serverless collection
        # ---------------------------------------------------------------
        self.collection = aoss.CfnCollection(
            self,
            "KBCollection",
            name=collection_name,
            type="VECTORSEARCH",
            description="Vector store for BADGERS knowledge base",
        )
        self.collection.add_dependency(encryption_policy)
        self.collection.add_dependency(network_policy)

        # ---------------------------------------------------------------
        # Lambda role for index creation custom resource
        # ---------------------------------------------------------------
        self.index_creator_role = iam.Role(
            self,
            "IndexCreatorRole",
            role_name=f"{id_prefix}-kb-index-creator-{deployment_id}",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
            inline_policies={
                "AOSSAccess": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=["aoss:APIAccessAll"],
                            resources=[self.collection.attr_arn],
                        ),
                    ]
                ),
            },
        )

        # Data access policy — grant KB role, index creator Lambda, and account access
        data_access_policy = aoss.CfnAccessPolicy(
            self,
            "KBDataAccessPolicy",
            name=f"{id_prefix}-kb-dap-{deployment_id}",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "index",
                                "Resource": [f"index/{collection_name}/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                            },
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                                "Permission": [
                                    "aoss:CreateCollectionItems",
                                    "aoss:UpdateCollectionItems",
                                    "aoss:DescribeCollectionItems",
                                ],
                            },
                        ],
                        "Principal": [
                            self.kb_role.role_arn,
                            self.index_creator_role.role_arn,
                            f"arn:aws:iam::{self.account}:root",
                        ],
                    }
                ]
            ),
        )
        data_access_policy.add_dependency(self.collection)

        # Grant KB role AOSS API access
        self.kb_role.add_to_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=[self.collection.attr_arn],
            )
        )

        # ---------------------------------------------------------------
        # Custom Resource: create vector index in AOSS collection
        # ---------------------------------------------------------------
        index_creator_fn = lambda_.Function(
            self,
            "IndexCreatorFunction",
            function_name=f"{id_prefix}-kb-index-creator-{deployment_id}",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.on_event",
            code=lambda_.Code.from_asset(
                str(
                    Path(__file__).parent.parent
                    / "lambdas"
                    / "code"
                    / "kb_index_creator"
                ),
                bundling={
                    "image": lambda_.Runtime.PYTHON_3_12.bundling_image,
                    "command": [
                        "bash",
                        "-c",
                        "pip install -r requirements.txt -t /asset-output && cp -au . /asset-output",
                    ],
                },
            ),
            role=self.index_creator_role,
            timeout=Duration.seconds(60),
            memory_size=256,
        )

        index_creator_provider = custom_resources.Provider(
            self,
            "IndexCreatorProvider",
            on_event_handler=index_creator_fn,
        )

        index_custom_resource = CustomResource(
            self,
            "AOSSIndexCustomResource",
            service_token=index_creator_provider.service_token,
            properties={
                "AOSSHost": self.collection.attr_collection_endpoint,
                "AOSSIndexName": index_name,
                "Dimensions": "1024",
            },
        )
        index_custom_resource.node.add_dependency(self.collection)
        index_custom_resource.node.add_dependency(data_access_policy)
        # ---------------------------------------------------------------
        # Bedrock Knowledge Base
        # ---------------------------------------------------------------
        self.knowledge_base = bedrock.CfnKnowledgeBase(
            self,
            "KnowledgeBase",
            name=kb_name,
            role_arn=self.kb_role.role_arn,
            knowledge_base_configuration=bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                type="VECTOR",
                vector_knowledge_base_configuration=bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                    embedding_model_arn=f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0",
                    embedding_model_configuration=bedrock.CfnKnowledgeBase.EmbeddingModelConfigurationProperty(
                        bedrock_embedding_model_configuration=bedrock.CfnKnowledgeBase.BedrockEmbeddingModelConfigurationProperty(
                            dimensions=1024,
                        )
                    ),
                ),
            ),
            storage_configuration=bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
                type="OPENSEARCH_SERVERLESS",
                opensearch_serverless_configuration=bedrock.CfnKnowledgeBase.OpenSearchServerlessConfigurationProperty(
                    collection_arn=self.collection.attr_arn,
                    vector_index_name=index_name,
                    field_mapping=bedrock.CfnKnowledgeBase.OpenSearchServerlessFieldMappingProperty(
                        metadata_field="AMAZON_BEDROCK_METADATA",
                        text_field="AMAZON_BEDROCK_TEXT_CHUNK",
                        vector_field="bedrock-knowledge-base-default-vector",
                    ),
                ),
            ),
            description="BADGERS knowledge base for grounding context",
        )
        self.knowledge_base.node.add_dependency(data_access_policy)
        self.knowledge_base.node.add_dependency(index_custom_resource)

        # ---------------------------------------------------------------
        # S3 Data Source — main_content/ prefix, NO chunking
        # ---------------------------------------------------------------
        self.data_source = bedrock.CfnDataSource(
            self,
            "KBDataSource",
            name=f"{id_prefix}-kb-datasource-{deployment_id}",
            knowledge_base_id=self.knowledge_base.attr_knowledge_base_id,
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                type="S3",
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=self.kb_bucket.bucket_arn,
                    inclusion_prefixes=["main_content/"],
                ),
            ),
            vector_ingestion_configuration=bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(
                    chunking_strategy="NONE",
                ),
            ),
            description="S3 data source for KB content (main_content/ folder)",
        )

        # ---------------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------------
        CfnOutput(
            self,
            "KBBucketName",
            value=self.kb_bucket.bucket_name,
            description="Knowledge base content bucket",
            export_name=f"{Stack.of(self).stack_name}-KBBucketName",
        )
        CfnOutput(
            self,
            "KnowledgeBaseId",
            value=self.knowledge_base.attr_knowledge_base_id,
            description="Bedrock Knowledge Base ID",
            export_name=f"{Stack.of(self).stack_name}-KnowledgeBaseId",
        )
        CfnOutput(
            self,
            "DataSourceId",
            value=self.data_source.attr_data_source_id,
            description="KB Data Source ID",
            export_name=f"{Stack.of(self).stack_name}-DataSourceId",
        )
        CfnOutput(
            self,
            "CollectionEndpoint",
            value=self.collection.attr_collection_endpoint,
            description="OpenSearch Serverless collection endpoint",
            export_name=f"{Stack.of(self).stack_name}-CollectionEndpoint",
        )
        CfnOutput(
            self,
            "CollectionArn",
            value=self.collection.attr_arn,
            description="OpenSearch Serverless collection ARN",
            export_name=f"{Stack.of(self).stack_name}-CollectionArn",
        )
