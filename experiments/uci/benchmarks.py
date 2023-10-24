"""
Definitions of benchmark suites.

A benchmark suite defines which datasets and models to use; and how to plot the results.
"""
import experiments.uci.datasets as ds
import experiments.uci.models as md
import experiments.uci.plotters as pl
from experiments.uci.benchmark_api import BenchmarkSet, make_benchmark_suite
from experiments.uci.dataset_api import DATASET_FACTORIES
from experiments.uci.grouping import GroupingKey as GK
from experiments.uci.grouping import GroupingSpec
from benchmark.model_api import MODEL_FACTORIES
from experiments.uci.plotter_api import PLOTTERS

make_benchmark_suite(
    name="test",
    description="Suite to test",
    sets=[
        BenchmarkSet(
            name="metrics",
            datasets=[
                ds.synthetic,
                ds.synthetic_asymmetric,
                ds.synthetic_focused,
                ds.synthetic_uniform
            ],
            models=[
                md.gpr,
                md.t_vgp,
                md.rcgpr
            ],
            plots=[
                pl.metrics_box_plot,
            ],
            do_compile=[True],
            do_optimise=[True],
            do_predict=True,
            do_posterior=False,
            file_by=GroupingSpec((GK.DATASET,), minimise=False),
            row_by=GroupingSpec((GK.METRIC,), minimise=False),
            column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
            line_by=None,
            repetitions=5,
        )
    ],
)

make_benchmark_suite(
    name="full",
    description="Suite that runs everything.",
    sets=[
        BenchmarkSet(
            name="metrics",
            datasets=DATASET_FACTORIES.all(),
            models=MODEL_FACTORIES.all(),
            plots=[
                pl.metrics_box_plot,
            ],
            do_compile=[True],
            do_optimise=[True],
            do_predict=True,
            do_posterior=False,
            file_by=GroupingSpec((GK.DATASET,), minimise=False),
            row_by=GroupingSpec((GK.METRIC,), minimise=False),
            column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
            line_by=None,
            repetitions=1,
        )
    ],
)

make_benchmark_suite(
    name="no_outliers",
    description="Suite that runs everything.",
    sets=[
        BenchmarkSet(
            name="metrics",
            datasets=[ds.boston,
                      ds.energy,
                      ds.yacht,
                      ds.synthetic],
            models=MODEL_FACTORIES.all(),
            plots=[
                pl.metrics_box_plot,
            ],
            do_compile=[True],
            do_optimise=[True],
            do_predict=True,
            do_posterior=False,
            file_by=GroupingSpec((GK.DATASET,), minimise=False),
            row_by=GroupingSpec((GK.METRIC,), minimise=False),
            column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
            line_by=None,
            repetitions=10,
        )
    ],
)
make_benchmark_suite(
    name="focused",
    description="Suite that runs everything.",
    sets=[
        BenchmarkSet(
            name="metrics",
            datasets=[ds.boston_focused,
                      ds.energy_focused,
                      ds.yacht_focused,
                      ds.synthetic_focused],
            models=MODEL_FACTORIES.all(),
            plots=[
                pl.metrics_box_plot,
            ],
            do_compile=[True],
            do_optimise=[True],
            do_predict=True,
            do_posterior=False,
            file_by=GroupingSpec((GK.DATASET,), minimise=False),
            row_by=GroupingSpec((GK.METRIC,), minimise=False),
            column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
            line_by=None,
            repetitions=10,
        )
    ],
)

make_benchmark_suite(
    name="asymmetric",
    description="Suite that runs everything.",
    sets=[
        BenchmarkSet(
            name="metrics",
            datasets=[ds.boston_asymmetric,
                      ds.energy_asymmetric,
                      ds.yacht_asymmetric,
                      ds.synthetic_asymmetric],
            models=MODEL_FACTORIES.all(),
            plots=[
                pl.metrics_box_plot,
            ],
            do_compile=[True],
            do_optimise=[True],
            do_predict=True,
            do_posterior=False,
            file_by=GroupingSpec((GK.DATASET,), minimise=False),
            row_by=GroupingSpec((GK.METRIC,), minimise=False),
            column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
            line_by=None,
            repetitions=10,
        )
    ],
)

make_benchmark_suite(
    name="uniform",
    description="Suite that runs everything.",
    sets=[
        BenchmarkSet(
            name="metrics",
            datasets=[ds.boston_uniform,
                      ds.energy_uniform,
                      ds.yacht_uniform,
                      ds.synthetic_uniform],
            models=MODEL_FACTORIES.all(),
            plots=[
                pl.metrics_box_plot,
            ],
            do_compile=[True],
            do_optimise=[True],
            do_predict=True,
            do_posterior=False,
            file_by=GroupingSpec((GK.DATASET,), minimise=False),
            row_by=GroupingSpec((GK.METRIC,), minimise=False),
            column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
            line_by=None,
            repetitions=10,
        )
    ],
)