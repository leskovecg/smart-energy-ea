from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

@dataclass
class IterationMetric:
    iteration: int
    metrics: Dict[str, float]  # e.g., {"accuracy": 0.92, "f1_score": 0.88}

@dataclass
class TimestampEntry:
    action: str
    timestamp: datetime

@dataclass
class ActiveLearningSession:
    dataset_uuid: str
    dataset_version: str
    model_type: str
    model_version: str
    hyperparameters: Dict[str, Any]
    training_dataset: List[int]
    validation_dataset: List[int]
    query_strategy: str
    iteration_samples: int
    seed_set: List[int]
    iteration_metrics: List[IterationMetric]
    timestamps: List[TimestampEntry]
    logging_and_auditing: Optional[str] = None  # blob or text log
    version_control: Optional[str] = None       # e.g., git hash
    experiment_parameters: Optional[List[str]] = field(default_factory=list)

@dataclass
class SwarmLearningSession:
    participant_id: str
    node_roles: List[str]  # e.g., ["leader", "member"]
    dataset_uuid: str
    dataset_version: str
    data_sensitivity: str  # e.g., "high", "confidential", etc.
    model_architecture: Union[Dict[str, Any], bytes]  # ONNX as bytes or JSON
    global_hyperparameters: Dict[str, Any]
    global_model_version: str
    aggregation_algorithm: str  # e.g., "federated averaging"
    weight_contributions: Dict[str, float]  # {participant_id: weight}
    aggregation_frequency: str  # e.g., "every 5 epochs"

@dataclass
class LayerDetail:
    layer_index: int
    layer_type: str
    config: Dict[str, Any]
    next_layers: List[int]

@dataclass
class OptimizerConfig:
    type: str  # e.g., "Adam", "SGD"
    config: Dict[str, Any]
    loss_metric: str  # e.g., "cross_entropy"

@dataclass
class NeurosymbolicLearningSession:
    dataset_uuid: str
    dataset_version: str
    transformation_steps: List[str]
    rules_file: str  # Path including training run ID
    model_type: str
    model_version: str
    layer_details: List[LayerDetail]
    hyperparameters: Dict[str, Union[float, int, str]]
    optimizer: OptimizerConfig
    loss_metric: str
    epoch_statistics: Dict[int, float]  # {epoch: loss_value}
    knowledge_representation: str
    satisfiability_scores: Dict[int, float]  # {checkpoint: score}
    seed_value: int
    training_indices: Union[List[int], Dict[str, List[int]]]
    validation_indices: Union[List[int], Dict[str, List[int]]]
    post_training_evaluation_metrics: Dict[str, float]
    training_time: float
    training_time_per_epoch: float
    training_failure_cases: List[int]
    validation_failure_cases: List[int]