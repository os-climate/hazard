"""Module for hazard-related utility functions for the self implemented CLI including autocompletion, hazard class discovery, and help text generation."""

from hazard import get_hazards_onboarding


def autocomplete_hazards(incomplete: str):
    """Hazard autocompletion based on user input."""
    hazards = get_hazards_onboarding().keys()
    return [hazard for hazard in hazards if hazard.startswith(incomplete)]


def hazards_help_text() -> str:
    """Generate a comma-separated string of available hazards."""
    hazards = get_hazards_onboarding().keys()
    return ", ".join(hazards)
