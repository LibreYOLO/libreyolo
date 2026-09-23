"""Shared ``plot_samples`` / ``plot_errors`` helpers.

``plot_samples`` is the number of validated images kept for the sample-image
plot, and ``plot_errors`` the number of misclassified or mis-detected images
kept for the error-analysis plot (#887). Both are plotting budgets only: they
never change which images are scored or the metrics that come out. Both
:class:`~libreyolo.training.config.TrainConfig` and
:class:`~libreyolo.validation.config.ValidationConfig` validate through here,
and every validator that collects sample images asks
:func:`wants_more_plot_samples` before keeping another one.
"""

from __future__ import annotations

#: Default number of validated images kept for the sample-image plot.
DEFAULT_PLOT_SAMPLES = 8

#: ``plot_samples`` value meaning "keep every validated image".
PLOT_SAMPLES_ALL = -1

#: Default number of error images kept for the error-analysis plot (off).
DEFAULT_PLOT_ERRORS = 0

#: Tasks whose validators draw the error-analysis plot (#887).
PLOT_ERRORS_TASKS = ("detect", "segment", "classify")


def validate_plot_samples(value, name: str = "plot_samples") -> int:
    """Validate a sample-image plot budget.

    Accepts a non-negative count, or ``-1`` for every validated image.
    ``name`` is the option reported in the error message.
    """
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be an integer >= 0, or -1 for all; got {value!r}"
        ) from None
    if count < PLOT_SAMPLES_ALL:
        raise ValueError(
            f"{name} must be >= 0, or -1 for all; got {count}"
        )
    return count


def validate_plot_errors(value, save_plots: bool) -> int:
    """Validate the error-analysis plot budget.

    Same range as ``plot_samples``. The error images are part of the plot
    output, so a non-zero budget without ``save_plots`` would parse and then
    draw nothing; that raises instead.
    """
    count = validate_plot_samples(value, name="plot_errors")
    if count != 0 and not save_plots:
        raise ValueError(
            f"plot_errors={count} needs save_plots=True (plots=True); "
            "the error images are written with the other validation plots"
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


def wants_more_plot_errors(config, collected: int) -> bool:
    """Whether a validator should keep another image for the error plot.

    Same contract as :func:`wants_more_plot_samples` for ``plot_errors``,
    which defaults to off. The error plot is part of the ``save_plots``
    output, so nothing is kept when plots are not being saved.
    """
    if not getattr(config, "save_plots", False):
        return False
    budget = getattr(config, "plot_errors", DEFAULT_PLOT_ERRORS)
    if budget == PLOT_SAMPLES_ALL:
        return True
    return collected < budget
