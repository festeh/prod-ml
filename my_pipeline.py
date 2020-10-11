from pathlib import Path

from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform
from tfx.proto import example_gen_pb2

from constants import COMPLAINTS_DIR
from training.train import get_trainer


class DataValidationPPl:
    def __init__(self, path: Path):
        self.path = path
        self.example_gen = self.get_example_gen()
        self.stats_gen = self.get_stats_gen()
        self.schema_gen = self.get_schema_gen(True)
        self.example_validator = self.get_example_validator()
        self.transform = self.get_transform()
        self.trainer = self.get_trainer()

    def get_example_gen(self):
        output = example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(
                splits=[
                    example_gen_pb2.SplitConfig.Split(
                        name="train", hash_buckets=9
                    ),
                    example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
                ]
            )
        )
        return CsvExampleGen(input_base=self.path.as_posix(), output_config=output)

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

    def get_trainer(self):
        return get_trainer(self.transform,
                           self.schema_gen)


if __name__ == '__main__':
    PATH = COMPLAINTS_DIR / "sample.csv"
    DataValidationPPl(PATH)
