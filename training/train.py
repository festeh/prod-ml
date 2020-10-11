from pathlib import Path

from tfx.components import Trainer
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.proto import trainer_pb2

TRAINING_STEPS = 1000
EVALUATION_STEPS = 100

CURRENT_DIR = Path(__file__).parent


def get_trainer(transform, schema_gen):
    trainer = Trainer(
        module_file=(CURRENT_DIR / "train_run_fn.py").as_posix(),
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        transformed_examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=TRAINING_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=EVALUATION_STEPS),
    )
    return trainer
