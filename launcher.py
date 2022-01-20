import argparse
import abc
from collections import OrderedDict

from generator.generator import TaxiGenerator, CensusGenerator, PlasticcGenerator
from benchmarks.taxi import run as taxi_run
from benchmarks.census import run as census_run
from benchmarks.plasticc import run as plasticc_run


class Benchmark(abc.ABC):
    @abc.abstractmethod
    def run(self, **kwargs):
        pass

    @staticmethod
    def print_result(result: OrderedDict):
        for k, v in result.items():
            print(f"{k}: {v}")


class TaxiBenchmark(Benchmark):
    _datafile = "taxi.csv"
    _records = 20_000_000

    def __init__(self, reuse: bool, **kwargs):
        self._reuse = reuse
        self._records = kwargs.pop("taxi_records", self._records)

    def run(self, **kwargs):
        if not self._reuse:
            print("Generating Taxi data file", self._datafile)
            gen = TaxiGenerator(self._datafile)
            gen.generate(self._records)
        print("Running Taxi benchmark")
        res = taxi_run(self._datafile)
        self.print_result(res)


class CensusBenchmark(Benchmark):
    _datafile = "census.csv"
    _records = 21721923

    def __init__(self, reuse: bool, **kwargs):
        self._reuse = reuse
        self._records = kwargs.pop("census_records", self._records)

    def run(self, **kwargs):
        if not self._reuse:
            print("Generating Census data file", self._datafile)
            gen = CensusGenerator(self._datafile)
            gen.generate(self._records)
        print("Running Census benchmark")
        res = census_run(self._datafile)
        self.print_result(res)


class PlasticcBenchmark(Benchmark):
    _datafile_prefix = "plasticc"
    _training_set_records = 1421705
    _test_set_records = 4536531
    _training_set_metadata_records = 7848
    _test_set_metadata_records = 3492890

    def __init__(self, reuse: bool, **kwargs):
        self._reuse = reuse
        self._training_set_records = kwargs.pop("training_set_records", self._training_set_records)
        self._test_set_records = kwargs.pop("test_set_records", self._test_set_records)
        self._training_set_metadata_records = kwargs.pop("training_set_metadata_records", self._training_set_metadata_records)
        self._test_set_metadata_records = kwargs.pop("test_set_metadata_records", self._test_set_metadata_records)

    def run(self):
        if not self._reuse:
            print("Generating Plasticc data files with prefix", self._datafile_prefix)
            gen = PlasticcGenerator(self._datafile_prefix)
            output_files = list(
                gen.generate(
                    self._training_set_records,
                    self._test_set_records,
                    self._training_set_metadata_records,
                    self._test_set_metadata_records,
                )
            )
        print("Running Plasticc benchmark")
        res = plasticc_run(*output_files)
        self.print_result(res)


def main():
    benchmarks = {
        "taxi": TaxiBenchmark,
        "census": CensusBenchmark,
        "plasticc": PlasticcBenchmark,
    }

    parser = argparse.ArgumentParser(description="Generate dataset for a benchmark.")
    parser.add_argument(
        "-m",
        "--mode",
        choices=list(benchmarks.keys()) + ["all"],
        required=True,
        help="Benchmark to run.",
    )
    parser.add_argument(
        "-tr",
        "--taxi-records",
        required=False,
        type=int,
        default=TaxiBenchmark._records,
        help="Override default number of records for Taxi benchmark.",
    )
    parser.add_argument(
        "-cr",
        "--census-records",
        required=False,
        type=int,
        default=CensusBenchmark._records,
        help="Override default number of records for Census benchmark.",
    )
    parser.add_argument(
        "-trsr",
        "--training-set-records",
        required=False,
        type=int,
        default=PlasticcBenchmark._training_set_records,
        help="Override default number of records to generate for training set in Plasticc benchmark.",
    )
    parser.add_argument(
        "-tesr",
        "--test-set-records",
        required=False,
        type=int,
        default=PlasticcBenchmark._test_set_records,
        help="Override default number of records to generate for test set in Plasticc benchmark.",
    )
    parser.add_argument(
        "-trsmr",
        "--training-set-metadata-records",
        required=False,
        type=int,
        default=PlasticcBenchmark._training_set_metadata_records,
        help="Override default number of records to generate for training set metadata in Plasticc benchmark.",
    )
    parser.add_argument(
        "-tesmr",
        "--test-set-metadata-records",
        required=False,
        type=int,
        default=PlasticcBenchmark._test_set_metadata_records,
        help="Override default number of records to generate for test set metadata in Plasticc benchmark.",
    )
    parser.add_argument(
        "-ru",
        "--reuse-dataset-files",
        required=False,
        default=False,
        help="Skip dataset generation phase and reuse datasets generated on previous runs."
    )

    args = parser.parse_args()
    modes = [benchmarks[args.mode]] if args.mode != "all" else benchmarks.values()

    for benchmark_class in modes:
        kwargs = vars(args)
        benchmark = benchmark_class(args.reuse_dataset_files, **kwargs)
        benchmark.run()


if __name__ == "__main__":
    main()
