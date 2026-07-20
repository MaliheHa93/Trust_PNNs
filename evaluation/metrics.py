from __future__ import annotations


def mean_or_none(
    values: list[
        float
    ],
):
    if not values:
        return None

    return (
        sum(
            values
        )
        / len(
            values
        )
    )


def summarize_decisions(
    rows: list[
        dict
    ],
) -> dict:

    n = len(
        rows
    )

    if n == 0:
        raise ValueError(
            "Cannot summarize empty rows."
        )

    bad_items = [
        row
        for row in rows
        if bool(
            row[
                "bad_output"
            ]
        )
    ]

    good_items = [
        row
        for row in rows
        if not bool(
            row[
                "bad_output"
            ]
        )
    ]

    accepted_items = [
        row
        for row in rows
        if bool(
            row[
                "accepted"
            ]
        )
    ]

    rejected_items = [
        row
        for row in rows
        if bool(
            row[
                "rejected"
            ]
        )
    ]

    forwarded_items = [
        row
        for row in rows
        if bool(
            row[
                "forwarded"
            ]
        )
    ]

    accepted_bad_items = [
        row
        for row in rows

        if (
            bool(
                row[
                    "accepted"
                ]
            )

            and bool(
                row[
                    "bad_output"
                ]
            )
        )
    ]

    rejected_good_items = [
        row
        for row in rows

        if (
            bool(
                row[
                    "rejected"
                ]
            )

            and not bool(
                row[
                    "bad_output"
                ]
            )
        )
    ]

    # -------------------------------------------------------------
    # Scientifically explicit metrics
    # -------------------------------------------------------------

    # P(accepted AND bad)
    unsafe_acceptance_frequency = (
        len(
            accepted_bad_items
        )
        / n
    )

    # P(accepted | bad)
    false_acceptance_rate = (

        len(
            accepted_bad_items
        )
        / len(
            bad_items
        )

        if bad_items

        else 0.0
    )

    # P(rejected | good)
    false_rejection_rate = (

        len(
            rejected_good_items
        )
        / len(
            good_items
        )

        if good_items

        else 0.0
    )

    # P(bad | accepted)
    accepted_output_contamination_rate = (

        len(
            accepted_bad_items
        )
        / len(
            accepted_items
        )

        if accepted_items

        else None
    )

    return {

        "runs":
            n,

        "overall_mae":
            mean_or_none(
                [
                    float(
                        row[
                            "absolute_error"
                        ]
                    )

                    for row
                    in rows
                ]
            ),

        "accepted_mae":
            mean_or_none(
                [
                    float(
                        row[
                            "absolute_error"
                        ]
                    )

                    for row
                    in accepted_items
                ]
            ),

        "bad_output_rate":
            (
                len(
                    bad_items
                )
                / n
            ),

        "acceptance_rate":
            (
                len(
                    accepted_items
                )
                / n
            ),

        "rejection_rate":
            (
                len(
                    rejected_items
                )
                / n
            ),

        "forwarding_rate":
            (
                len(
                    forwarded_items
                )
                / n
            ),

        # Recommended terminology.
        "unsafe_acceptance_frequency":
            unsafe_acceptance_frequency,

        "false_acceptance_rate":
            false_acceptance_rate,

        "false_rejection_rate":
            false_rejection_rate,

        "accepted_output_contamination_rate":
            accepted_output_contamination_rate,

        # ---------------------------------------------------------
        # Backward-compatible aliases.
        # Existing plotting code will continue working.
        # ---------------------------------------------------------

        "false_acceptance_rate_all_runs":
            unsafe_acceptance_frequency,

        "false_rejection_rate_all_runs":
            (
                len(
                    rejected_good_items
                )
                / n
            ),

        "conditional_false_acceptance_rate":
            false_acceptance_rate,

        "conditional_false_rejection_rate":
            false_rejection_rate,

        "bad_rate_among_accepted":
            accepted_output_contamination_rate,
    }