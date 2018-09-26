from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import numpy as np
import os
import yaml
from collections import defaultdict

from ray.tune.log_sync import get_syncer
from ray.tune.result import NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S, \
    TIMESTEPS_TOTAL

import IPython as ip

try:
    import tensorflow as tf
except ImportError:
    tf = None
    print("Couldn't import TensorFlow - this disables TensorBoard logging.")

LINE = "line"
HISTO = "histo"
GLOBAL_HISTO = "global_histo"

class LoggerStat(object):
    """Wrapper class for metric stats to be logged"""

    def __init__(self, value, plot_type=LINE):
        self._value = value
        self._plot_type = plot_type
        assert self._plot_type in [LINE, HISTO, GLOBAL_HISTO], "Plot type '{}' not recognized".format(self._plot_type)

    @property
    def value(self):
        return self._value

    @property
    def plot_type(self):
        return self._plot_type

class Logger(object):
    """Logging interface for ray.tune; specialized implementations follow.

    By default, the UnifiedLogger implementation is used which logs results in
    multiple formats (TensorBoard, rllab/viskit, plain json) at once.
    """

    def __init__(self, config, logdir, upload_uri=None):
        self.config = config
        self.logdir = logdir
        self.uri = upload_uri
        self._init()

    def _init(self):
        pass

    def on_result(self, result):
        """Given a result, appends it to the existing log."""

        raise NotImplementedError

    def close(self):
        """Releases all resources used by this logger."""

        pass

    def flush(self):
        """Flushes all disk writes to storage."""

        pass


class UnifiedLogger(Logger):
    """Unified result logger for TensorBoard, rllab/viskit, plain json.

    This class also periodically syncs output to the given upload uri."""

    def _init(self):
        self._loggers = []
        for cls in [_JsonLogger, _TFLogger, _VisKitLogger]:
            if cls is _TFLogger and tf is None:
                print("TF not installed - cannot log with {}...".format(cls))
                continue
            self._loggers.append(cls(self.config, self.logdir, self.uri))
        self._log_syncer = get_syncer(self.logdir, self.uri)

    def on_result(self, result):
        for logger in self._loggers:
            logger.on_result(result)
        self._log_syncer.set_worker_ip(result.get(NODE_IP))
        self._log_syncer.sync_if_needed()

    def close(self):
        for logger in self._loggers:
            logger.close()
        self._log_syncer.sync_now(force=True)

    def flush(self):
        for logger in self._loggers:
            logger.flush()
        self._log_syncer.sync_now(force=True)
        self._log_syncer.wait()


class NoopLogger(Logger):
    def on_result(self, result):
        pass

class _JsonLogger(Logger):
    def _init(self):
        config_out = os.path.join(self.logdir, "params.json")
        with open(config_out, "w") as f:
            json.dump(self.config, f, sort_keys=True, cls=_SafeFallbackEncoder)
        local_file = os.path.join(self.logdir, "result.json")
        self.local_out = open(local_file, "w")

    def on_result(self, result):
        json.dump(result, self, cls=_SafeFallbackEncoder)
        self.write("\n")

    def write(self, b):
        self.local_out.write(b)
        self.local_out.flush()

    def close(self):
        self.local_out.close()

def build_histogram(values, bins=1000):
    # Taken from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

    # Create histogram using numpy        
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    return hist

class _TFLogger(Logger):
    def _init(self):
        self._file_writer = tf.summary.FileWriter(self.logdir)
        self._global_histograms = defaultdict(list)

    def to_tf_values(self, result, path):
        values = []
        global_values = []
        for attr, value in result.items():
            if value is not None:
                if type(value) is LoggerStat:
                    if value.plot_type == LINE:
                        values.append(tf.Summary.Value(tag="/".join(path + [attr]), simple_value=value.value))
                    elif value.plot_type == HISTO:
                        values.append(tf.Summary.Value(tag="/".join(path + [attr]), histo=build_histogram(np.array(value.value))))
                    elif value.plot_type == GLOBAL_HISTO: 
                        tag = "/".join(path + [attr])
                        self._global_histograms[tag].append(value.value)
                        global_values.append(tf.Summary.Value(tag=tag, histo=build_histogram(np.array(self._global_histograms[tag]))))                          
                elif type(value) is dict:
                    v, gv = self.to_tf_values(value, path + [attr])
                    values.extend(v)
                    global_values.extend(gv)
        return values, global_values

    def on_result(self, result):
        tmp = result.copy()
        for k in [
                "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            del tmp[k]  # not useful to tf log these
        values, global_values = self.to_tf_values(tmp, ["ray", "tune"])
        train_stats = tf.Summary(value=values)
        train_stats_global = tf.Summary(value=global_values)
        t = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        self._file_writer.add_summary(train_stats, t)
        self._file_writer.add_summary(train_stats_global, 0) # global stats are not tied to a particular timestep, so we simulate this by always plotting at t=0
        iteration_value = self.to_tf_values({
            "training_iteration": result[TRAINING_ITERATION]
        }, ["ray", "tune"])[0]
        iteration_stats = tf.Summary(value=iteration_value)
        self._file_writer.add_summary(iteration_stats, t)
        self._file_writer.flush()

    def flush(self):
        self._file_writer.flush()

    def close(self):
        self._file_writer.close()


class _VisKitLogger(Logger):
    def _init(self):
        """CSV outputted with Headers as first set of results."""
        # Note that we assume params.json was already created by JsonLogger
        self._file = open(os.path.join(self.logdir, "progress.csv"), "w")
        self._csv_out = None

    def on_result(self, result):
        if self._csv_out is None:
            self._csv_out = csv.DictWriter(self._file, result.keys())
            self._csv_out.writeheader()
        self._csv_out.writerow(result.copy())

    def close(self):
        self._file.close()


class _SafeFallbackEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(_SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return None
            if np.issubdtype(value, float):
                return float(value)
            if np.issubdtype(value, int):
                return int(value)
        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_print(result):
    result = result.copy()
    result.update(config=None)  # drop config from pretty print
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=_SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)
