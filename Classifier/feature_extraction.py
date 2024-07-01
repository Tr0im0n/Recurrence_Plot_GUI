
import numpy as np
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric

def pyrqa(timeseries, m, T, epsilon):
    time_series = TimeSeries(timeseries,
                                   embedding_dimension=m,
                                   time_delay=T)
    settings = Settings(time_series,
                        neighbourhood=FixedRadius(epsilon),
                        similarity_measure=EuclideanMetric(), theiler_corrector=1)
    result = RQAComputation.create(settings).run()
    return np.array([result.recurrence_rate,
                     result.determinism,
                     result.average_diagonal_line,
                     result.trapping_time,
                     result.longest_diagonal_line,
                     result.divergence,
                     result.entropy_diagonal_lines,
                     result.laminarity])