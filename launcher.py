import argparse
import abc
import sys
import json
import time
from collections import OrderedDict

import os
# Need to set it early before importing generator classes because they import Modin
# and this variable has to be set already in case modin experimental API is needed later.
os.environ["MODIN_EXPERIMENTAL"] = "true"

print("ENV --------------------------------------------------------------{")
for k,v in os.environ.items():
    print(f"{k}={v}")
import socket
print("Host name = ", socket.gethostname())
print("ENV --------------------------------------------------------------}")

from generator.generator import TaxiGenerator, CensusGenerator, PlasticcGenerator
from benchmarks.taxi import run as taxi_run
from benchmarks.census import run as census_run
from benchmarks.plasticc import run as plasticc_run

import unidist
unidist.init()

class Benchmark(abc.ABC):
    def __init__(self, reuse: bool, parallel: bool, num_cpus: int):
        self._reuse = reuse
        self._parallel = parallel
        self._num_cpus = num_cpus

    @abc.abstractmethod
    def run(self, **kwargs):
        pass


class TaxiBenchmark(Benchmark):
    _datafile = "taxi.csv"
    _records = 20_000_000

    def __init__(self, reuse: bool, parallel: bool, num_cpus: int, **kwargs):
        super().__init__(reuse, parallel, num_cpus)
        self._records = kwargs.pop("taxi_records", self._records)

    def run(self, **kwargs) -> tuple[OrderedDict, float]:
        print(f'{"Reusing" if self._reuse else "Generating"} Taxi data file {self._datafile}')
        gen = TaxiGenerator(self._datafile, self._reuse, self._parallel, self._num_cpus)
        gen.generate(self._records)

        print("Running Taxi benchmark")
        t0 = time.time()
        res = taxi_run(self._datafile)
        t1 = time.time()
        return res, t1 - t0


class CensusBenchmark(Benchmark):
    _datafile = "census.csv"
    _records = 21721923

    def __init__(self, reuse: bool, parallel: bool, num_cpus: int, **kwargs):
        super().__init__(reuse, parallel, num_cpus)
        self._records = kwargs.pop("census_records", self._records)

    def run(self, **kwargs) -> tuple[OrderedDict, float]:
        print(f'{"Reusing" if self._reuse else "Generating"} Census data file {self._datafile}')
        gen = CensusGenerator(self._datafile, self._reuse, self._parallel, self._num_cpus)
        gen.generate(self._records)

        print("Running Census benchmark")
        t0 = time.time()
        res = census_run(self._datafile)
        t1 = time.time()
        return res, t1 - t0


class PlasticcBenchmark(Benchmark):
    _datafile_prefix = "plasticc"
    _training_set_records = 1_421_705
    _test_set_records = 45_365_310
    _training_set_metadata_records = 7848
    _test_set_metadata_records = 349_289

    def __init__(self, reuse: bool, parallel: bool, num_cpus: int, **kwargs):
        super().__init__(reuse, parallel, num_cpus)
        self._training_set_records = kwargs.pop("training_set_records", self._training_set_records)
        self._test_set_records = kwargs.pop("test_set_records", self._test_set_records)
        self._training_set_metadata_records = kwargs.pop("training_set_metadata_records", self._training_set_metadata_records)
        self._test_set_metadata_records = kwargs.pop("test_set_metadata_records", self._test_set_metadata_records)

    def run(self) -> tuple[OrderedDict, float]:
        print(f'{"Reusing" if self._reuse else "Generating"} Plasticc data files with prefix {self._datafile_prefix}')
        gen = PlasticcGenerator(self._datafile_prefix, self._reuse, self._parallel, self._num_cpus)
        output_files = list(
            gen.generate(
                self._training_set_records,
                self._test_set_records,
                self._training_set_metadata_records,
                self._test_set_metadata_records,
            )
        )

        print("Running Plasticc benchmark")
        t0 = time.time()
        res = plasticc_run(*output_files)
        t1 = time.time()
        return res, t1 - t0


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
        action='store_true',
        required=False,
        default=False,
        help="Skip dataset generation phase and reuse datasets generated on previous runs."
    )
    parser.add_argument(
        "-np",
        "--no-parallel",
        action='store_true',
        required=False,
        default=False,
        help="Disable parallel dataset generation."
    )
    parser.add_argument(
        "--cpus",
        required=False,
        type=int,
        help="Specify maximum number of CPU cores to use."
    )
    parser.add_argument(
        "--hdk",
        action='store_true',
        required=False,
        default=False,
        help="Use experimental HDK engine to execute benchmarks."
    )
    parser.add_argument(
        "-j",
        "--json",
        action='store_true',
        required=False,
        default=False,
        help="Output result in JSON format."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        type=str,
        help="Output file to write JSON data to. By default stdout is used."
    )

    args = parser.parse_args()
    modes = [args.mode] if args.mode != "all" else benchmarks.keys()

    if args.cpus is not None:
        os.environ["MODIN_CPUS"] = str(args.cpus)
        print("Using", args.cpus, "number of CPU cores")
    else:
        print("Using maximum available number of CPU cores")

    if args.hdk:
        os.environ["MODIN_STORAGE_FORMAT"] = "hdk"
        os.environ["MODIN_ENGINE"] = "native"

    benchmark_results = OrderedDict()
    for benchmark_name in modes:
        benchmark_class = benchmarks[benchmark_name]
        kwargs = vars(args)
        benchmark = benchmark_class(args.reuse_dataset_files, not args.no_parallel, args.cpus, **kwargs)
        results, total_time = benchmark.run()
        results["Total"] = total_time
        results.move_to_end("Total", last=False)
        benchmark_results[benchmark_name] = results

    with open(args.output, "w") if args.output is not None else sys.stdout as fp:
        if args.json:
            json.dump(benchmark_results, fp, indent=4)
            fp.write("\n")
        else:
            for name, results in benchmark_results.items():
                total_time = results.pop("Total")
                for k, v in results.items():
                    print(f"{k}: {v}", file=fp)
                print(f"Total {name} benchmark execution time: {total_time}", file=fp)


if __name__ == "__main__":
    main()
