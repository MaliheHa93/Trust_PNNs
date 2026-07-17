from __future__ import annotations


def mean_or_none(values: list[float]):
    if not values:
        return None
    return sum(values) / len(values)


def summarize_decisions(rows: list[dict]) -> dict:
    n = len(rows)

    if n == 0:
        raise ValueError("Cannot summarize empty rows.")

    bad_items = [x for x in rows if bool(x["bad_output"])]
    good_items = [x for x in rows if not bool(x["bad_output"])]
    accepted_items = [x for x in rows if bool(x["accepted"])]
    rejected_items = [x for x in rows if bool(x["rejected"])]
    forwarded_items = [x for x in rows if bool(x["forwarded"])]

    accepted_bad_items = [
        x for x in rows
        if bool(x["accepted"]) and bool(x["bad_output"])
    ]

    rejected_good_items = [
        x for x in rows
        if bool(x["rejected"]) and not bool(x["bad_output"])
    ]

    return {
        "runs": n,
        "overall_mae": mean_or_none([
            float(x["absolute_error"])
            for x in rows
        ]),
        "accepted_mae": mean_or_none([
            float(x["absolute_error"])
            for x in accepted_items
        ]),
        "bad_output_rate": len(bad_items) / n,
        "acceptance_rate": len(accepted_items) / n,
        "rejection_rate": len(rejected_items) / n,
        "forwarding_rate": len(forwarded_items) / n,
        "false_acceptance_rate_all_runs": len(accepted_bad_items) / n,
        "false_rejection_rate_all_runs": len(rejected_good_items) / n,
        "conditional_false_acceptance_rate": (
            len(accepted_bad_items) / len(bad_items)
            if bad_items else 0.0
        ),
        "conditional_false_rejection_rate": (
            len(rejected_good_items) / len(good_items)
            if good_items else 0.0
        ),
        "bad_rate_among_accepted": (
            len(accepted_bad_items) / len(accepted_items)
            if accepted_items else None
        ),
    }