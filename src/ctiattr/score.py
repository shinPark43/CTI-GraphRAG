"""Score predictions with CTIConnect's official metrics (vendored, unmodified).

Items with no prediction are scored as empty predictions (F1 = 0) rather than
skipped, so failed calls cannot inflate a score.
"""

from __future__ import annotations

from ctiattr.data import QA
from ctiattr.vendor.cticonnect_metrics import aggregate_scores, score_id_item


def score(predictions: dict[str, str], qas: list[QA]) -> dict:
    items = [score_id_item(predictions.get(qa.id, ""), qa.ground_truth,
                           task=qa.task, item_id=qa.id) for qa in qas]
    result = aggregate_scores(items)
    result["n_missing"] = sum(qa.id not in predictions for qa in qas)
    result["mean_pred_ids"] = round(sum(len(i.pred_ids) for i in items) / max(len(items), 1), 3)
    return result
