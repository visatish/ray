from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import numpy as np
import os
import yaml

from ray.tune.log_sync import get_syncer
from ray.tune.result import NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S, \
    TIMESTEPS_TOTAL

try:
    import tensorflow as tf
except ImportError:
    tf = None
    print("Couldn't import TensorFlow - this disables TensorBoard logging.")

class Logger(object):
    """Logging interface for ray.tune; specialized implementations follow.

    By default, the UnifiedLogger implementation is used which logs results in
    multiple formats (TensorBoard, rllab/viskit, plain json) at once.
    """

    def __init__(self, config, logdir, upload_uri=None, tf_histogram_keys=[]):
        self.config = config
        self.logdir = logdir
        self.uri = upload_uri
        self.tf_histogram_keys = tf_histogram_keys
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
            if cls is _TFLogger:
                if tf is None:
                    print("TF not installed - cannot log with {}...".format(cls))
                    continue
                else:
                    self._loggers.append(cls(self.config, self.logdir, self.uri, tf_histogram_keys=self.tf_histogram_keys))
            else:
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

    def to_tf_values(self, result, path):
        values = []
        for attr, value in result.items():
            if value is not None:
                if isinstance(value, (int, float, np.float32, np.float64, np.int32, list)):
                    if attr in self.tf_histogram_keys:
                        values.append(tf.Summary.Value(tag="/".join(path + [attr]), histo=build_histogram(np.array(value))))
                    else:
                        values.append(tf.Summary.Value(tag="/".join(path + [attr]), simple_value=value))
                elif isinstance(value, dict):
                    values.extend(self.to_tf_values(value, path + [attr]))
        return values

    def on_result(self, result):
        tmp = result.copy()
        for k in [
                "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            del tmp[k]  # not useful to tf log these
        values= self.to_tf_values(tmp, ["ray", "tune"])
        train_stats = tf.Summary(value=values)
        t = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        self._file_writer.add_summary(train_stats, t)
        iteration_value = self.to_tf_values({
            "training_iteration": result[TRAINING_ITERATION]
        }, ["ray", "tune"])
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

def reduce_mean_result_dict(result):
    """Sometimes we want to have arrays or lists in the result dict for logging histograms and such,
but we don't want to print those. Instead we reduce them by the mean."""
    if isinstance(result, (np.ndarray, list)):
        return np.mean(result)
    elif not isinstance(result, dict):
        return result
    else:
        return {key: reduce_mean_result_dict(val) for key, val in result.items()}

def pretty_print(result):
    result = reduce_mean_result_dict(result.copy())
    result.update(config=None)  # drop config from pretty print
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=_SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)
