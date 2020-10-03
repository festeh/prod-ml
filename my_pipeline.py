from pathlib import Path

from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform
from tfx.proto import example_gen_pb2
from tfx.utils.dsl_utils import external_input

from constants import COMPLAINTS_DIR


class DataValidationPPl:
    def __init__(self, path: Path):
        self.path = path
        self.example_gen = self.get_example_gen()
        self.stats_gen = self.get_stats_gen()
        self.schema_gen = self.get_schema_gen(True)
        self.example_validator = self.get_example_validator()
        self.preprocessor = self.get_transform()

    def get_example_gen(self):
        pattern = self.path.name
        c_input = example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='data', pattern=pattern)
        ])
        examples = external_input(self.path.parent)
        return CsvExampleGen(input_base=self.path.parent.as_posix(), input_config=c_input)

    def get_stats_gen(self):
        return StatisticsGen(examples=self.example_gen.outputs["examples"])

    def get_schema_gen(self, infer_shape=True):
        return SchemaGen(statistics=self.stats_gen.outputs["statistics"], infer_feature_shape=infer_shape)

    def get_example_validator(self):
        return ExampleValidator(statistics=self.stats_gen.outputs["statistics"],
                                schema=self.schema_gen.outputs["schema"])

    def get_transform(self):
        return Transform(examples=self.example_gen.outputs["examples"],
                         schema=self.schema_gen.outputs["schema"],
                         module_file=(Path(__file__).parent/"preprocessing/complaints_preporcessing.py").as_posix()
                         )


if __name__ == '__main__':
    PATH = COMPLAINTS_DIR / "sample.csv"
    DataValidationPPl(PATH)
