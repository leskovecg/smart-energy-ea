# minio-ai-storage

## Data Structures

### 🔄 Active Learning Model Metadata

| Field | Type | Description |
|-------|------|-------------|
| `dataset_uuid` | `string` | UUID of the dataset in the data lake. |
| `dataset_version` | `string` | The version or timestamp of the dataset used. |
| `model_type` | `string` | The type of model (e.g., neural network, SVM, random forest). |
| `model_version` | `string` | Version of the model or training pipeline used. |
| `hyperparameters` | `JSON` | All hyperparameter settings (e.g., learning rate, batch size). |
| `training_dataset` | `[integer]` | Subset of data used for training (indices, IDs). |
| `validation_dataset` | `[integer]` | Subset of data used for validation (indices, IDs). |
| `query_strategy` | `string` | The strategy used to select samples (e.g., uncertainty sampling, diversity sampling), including metrics or criteria (e.g., entropy, margin). |
| `iteration_samples` | `integer` | The number of samples queried in each iteration. |
| `seed_set` | `[integer]` | The initial labelled dataset used to start active learning. |
| `iteration_metrics` | `[ { iteration: integer, metrics: JSON } ]` | Metrics like accuracy, precision, recall, F1 score per iteration. The final record reflects final model performance. |
| `timestamps` *(optional)* | `[{ action: string, timestamp: datetime }]` | Actions including dataset preparation, training, querying, and annotation. |
| `logging_and_auditing` *(optional)* | `string` (blob/text) | Logs of decisions made by the algorithm and any manual overrides. |
| `version_control` *(optional)* | `string` | Git hashes or version identifiers for datasets, models, and code. |
| `experiment_parameters` | `[string]` | Unique IDs or names for each experiment run. |

### 🐝 Swarm Learning Model Metadata

| Field | Type | Description |
|-------|------|-------------|
| `participant_id` | `string` | Unique identifier for each participant in the swarm. |
| `node_roles` | `[string]` | Roles in the swarm (e.g., `leader`, `member`). A node may have multiple roles. |
| `dataset_uuid` | `string` | UUID of the dataset in the data lake. |
| `dataset_version` | `string` | The version or timestamp of the dataset used. |
| `data_sensitivity` | `string` | Privacy and security level of the participant's data (e.g., confidential, restricted). |
| `model_architecture` | `onnx` or `JSON` | Shared model architecture agreed upon by participants, either as ONNX binary or JSON structure. |
| `global_hyperparameters` | `JSON` | Global training settings (e.g., learning rate, batch size). |
| `global_model_version` | `string` | Version of the globally aggregated model. |
| `aggregation_algorithm` | `string` | Aggregation technique used (e.g., federated averaging, weighted aggregation). |
| `weight_contributions` | `JSON` | Proportion of each node's contribution to the global model (e.g., `{ "node1": 0.25, "node2": 0.75 }`). |
| `aggregation_frequency` | `string` | Frequency of aggregation (e.g., every N epochs, after each round). |



### 🧠 Neurosymbolic Learning Model Metadata

| Field | Type | Description |
|-------|------|-------------|
| `dataset_uuid` | `string` | UUID of the dataset in the MinIO data lake. |
| `dataset_version` | `string` | The version or timestamp of the dataset used. |
| `transformation_steps` | `[string]` | Labels representing preprocessing and transformations applied to raw data. |
| `rules_file` | `string` | Path to a JSON file representing symbolic rules. File name includes the training run ID. |
| `model_type` | `string` | Type of neural model (e.g., CNN, RNN, Transformer). |
| `model_version` | `string` | Version of the model from the HumAIne model repository (likely ONNX format). |
| `layer_details` | `[JSON]` | List of objects with layer index, type, configuration arguments, and next layer indices. |
| `hyperparameters` | `JSON` | Training settings (e.g., learning rate, batch size, number of epochs). |
| `optimizer` | `JSON` | Optimizer type (e.g., Adam, SGD), configuration, and associated loss metric. |
| `loss_metric` | `string` | Name of the loss function minimized during training (e.g., cross entropy). |
| `epoch_statistics` | `JSON` | Loss metric values at specific epoch checkpoints. |
| `knowledge_representation` | `string` | Format or method to convert rules into a model-compatible representation (e.g., Probabilistic SDD). |
| `satisfiability_scores` | `JSON` | Ruleset satisfiability scores at specific training checkpoints. |
| `seed_value` | `integer` | Random seed used for training. |
| `training_indices` | `[integer]` or `JSON` | Indices or structured data indicating which instances were used for training (e.g., per fold). |
| `validation_indices` | `[integer]` or `JSON` | Indices or structured data for validation instances (e.g., per fold). |
| `post_training_evaluation_metrics` | `JSON` | Metrics calculated after training on validation sets (e.g., accuracy, F1). |
| `training_time` | `float` | Total time spent during training (in seconds or chosen unit). |
| `training_time_per_epoch` | `float` | Average training time per epoch. |
| `training_failure_cases` | `[integer]` | Indices of mislabeled predictions in the training set. |
| `validation_failure_cases` | `[integer]` | Indices of mislabeled predictions in the validation set. |


## Development

Make editable install in the root directory:
```
pip install -e .
```

Run tests and other stuff from the root:
```pytest tests/```
```python sandbox/tests.py```
