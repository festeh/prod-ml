import os

from tfx.components import CsvExampleGen
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tfx.utils.dsl_utils import external_input
from tfx.proto import example_gen_pb2

from constants import DATA_DIR


def csv_to_example_gen(path, pattern='sample.csv'):
    c_input = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='data', pattern=pattern)
    ])
    examples = external_input(path)
    example_gen = CsvExampleGen(examples, input_config=c_input)
    return example_gen


if __name__ == '__main__':
    context = InteractiveContext()
    path_to_csv_file = DATA_DIR / "csv/"
    context.run(csv_to_example_gen(path_to_csv_file))
