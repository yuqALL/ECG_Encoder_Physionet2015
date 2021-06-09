import numpy as np
import time
import logging

log = logging.getLogger(__name__)


class SsimMonitor(object):
    def __init__(self, col_suffix="SSIM"):
        self.col_suffix = col_suffix

    def monitor_epoch(self, ):
        return

    def monitor_set(
            self,
            setname,
            all_preds,
            all_losses,
            all_batch_sizes,
            all_targets,
            all_targets_alarm_type
    ):
        all_pred_labels = []
        all_target_labels = []
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        for preds, targets in zip(all_preds, all_targets):
            if preds.ndim == 2:
                pred_labels = np.int32(preds[0, 0] < self.threshold_for_binary_case)
            else:
                pred_labels = np.int32(preds[0] < self.threshold_for_binary_case)
            targets_labels = targets
            all_pred_labels.append(pred_labels)
            all_target_labels.append(targets_labels)

        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape

        misclass = 1 - np.mean(all_target_labels == all_pred_labels)
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(misclass)}


class LossMonitor(object):
    """
    Monitor the examplewise loss.
    """

    def monitor_epoch(self, ):
        return

    def monitor_set(
            self,
            setname,
            all_preds,
            all_losses,
            all_batch_sizes,
            all_targets,
            all_targets_alarm_type
    ):
        batch_weights = np.array(all_batch_sizes) / float(
            np.sum(all_batch_sizes)
        )
        loss_per_batch = [np.mean(loss) for loss in all_losses]
        mean_loss = np.sum(batch_weights * loss_per_batch)
        column_name = "{:s}_loss".format(setname)
        return {column_name: mean_loss}


class RuntimeMonitor(object):
    """
    Monitor the runtime of each epoch.

    First epoch will have runtime 0.
    """

    def __init__(self):
        self.last_call_time = None

    def monitor_epoch(self, ):
        return

    def monitor_set(
            self,
            setname,
            all_preds,
            all_losses,
            all_batch_sizes,
            all_targets,
            all_targets_alarm_type
    ):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        epoch_runtime = cur_time - self.last_call_time
        self.last_call_time = cur_time
        column_name = "{:s}_runtime".format(setname)
        return {column_name: epoch_runtime}
