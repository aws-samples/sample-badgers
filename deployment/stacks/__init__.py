"""CDK stacks for PDF Extractor POC."""

from .s3_stack import S3Stack
from .iam_stack import IAMStack
from .cognito_stack import CognitoStack
from .lambda_stack import LambdaAnalyzerStack
from .agentcore_ecr_stack import AgentCoreECRStack
from .agentcore_gateway_stack import AgentCoreGatewayStack
from .agentcore_runtime_websocket_stack import AgentCoreRuntimeWebSocketStack
from .agentcore_memory_stack import AgentCoreMemoryStack
from .inference_profiles_stack import InferenceProfilesStack
from .custom_analyzers_stack import CustomAnalyzersStack

__all__ = [
    "S3Stack",
    "IAMStack",
    "CognitoStack",
    "LambdaAnalyzerStack",
    "AgentCoreECRStack",
    "AgentCoreGatewayStack",
    "AgentCoreRuntimeWebSocketStack",
    "AgentCoreMemoryStack",
    "InferenceProfilesStack",
    "CustomAnalyzersStack",
]
