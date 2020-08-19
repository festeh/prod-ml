import os

from tfx.components import CsvExampleGen
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tfx.utils.dsl_utils import external_input
from  tfx.proto import example_gen_pb2

from constants import DATA_DIR

if __name__ == '__main__':
    context = InteractiveContext()
    path_to_csv_file = DATA_DIR / "csv/"
    c_input = example_gen_pb2.Input(splits=[
                      example_gen_pb2.Input.Split(name='data', pattern='sample.csv')
                     ])
    example_gen_pb2.Input()
    examples = external_input(path_to_csv_file)
    example_gen = CsvExampleGen(examples, input_config=c_input)
    context.run(example_gen)
