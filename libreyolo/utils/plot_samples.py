"""Shared ``plot_samples`` helpers.

``plot_samples`` is the number of validated images kept for the sample-image
plot. It is a plotting budget only: it never changes which images are scored
or the metrics that come out. Both :class:`~libreyolo.training.config.TrainConfig`
and :class:`~libreyolo.validation.config.ValidationConfig` validate through
here, and every validator that collects sample images asks
:func:`wants_more_plot_samples` before keeping another one.
"""

from __future__ import annotations

#: Default number of validated images kept for the sample-image plot.
DEFAULT_PLOT_SAMPLES = 8

#: ``plot_samples`` value meaning "keep every validated image".
PLOT_SAMPLES_ALL = -1


def validate_plot_samples(value) -> int:
    """Validate the sample-image plot budget.

    Accepts a non-negative count, or ``-1`` for every validated image.
    """
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"plot_samples must be an integer >= 0, or -1 for all; got {value!r}"
        ) from None
    if count < PLOT_SAMPLES_ALL:
        raise ValueError(
            f"plot_samples must be >= 0, or -1 for all; got {count}"
        )
    return count


def wants_more_plot_samples(config, collected: int) -> bool:
    """Whether a validator should keep another image for the sample plot.

    ``config`` is any object carrying a ``plot_samples`` attribute (a missing
    attribute falls back to the default budget); ``collected`` is how many
    images are already held. Bounded so a small budget does not hold images in
    memory for the whole run; ``-1`` keeps every validated image.
    """
    budget = getattr(config, "plot_samples", DEFAULT_PLOT_SAMPLES)
    if budget == PLOT_SAMPLES_ALL:
        return True
    return collected < budget
