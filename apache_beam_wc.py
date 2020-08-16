# Count each word occurrences
import re
from pathlib import Path

from apache_beam import Pipeline, FlatMap, Map, CombinePerKey
from apache_beam.io import ReadFromText, WriteToText


def fomat_wc_item(result):
    word, count = result
    return f"{word}: {count}"


if __name__ == '__main__':
    input_file = "gs://dataflow-samples/shakespeare/kinglear.txt"
    local_input_file = "data/beam_example/kinglear.txt"
    local_wc = "data/beam_example/wc"

    if not Path(local_input_file).exists():
        with Pipeline() as p:
            lines = p | ReadFromText(input_file)
            output = lines | WriteToText(local_input_file, shard_name_template="")

    with Pipeline() as p:
        lines = p | ReadFromText(local_input_file)
        counts = (
                lines
                | 'Split' >> FlatMap(lambda x: re.findall(r'[a-zA-z\']+', x))
                | 'Create uniq' >> Map(lambda x: (x, 1))
                | 'Agg' >> CombinePerKey(sum)
                | 'Format' >> Map(fomat_wc_item)
        )
        counts | WriteToText(local_wc, shard_name_template="")
