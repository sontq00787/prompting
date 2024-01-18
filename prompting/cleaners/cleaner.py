from typing import List, Dict

import bittensor as bt

from prompting.cleaners import RemoveQuotes, RemoveRoles, PruneEnding

SUPPORTED_CLEANERS = {
    "remove_quotes": RemoveQuotes,
    "remove_roles": RemoveRoles,
    "prune_ending": PruneEnding,
}


class CleanerPipeline:
    def __init__(self) -> None:
        pass

    def apply(self, generation: str, cleaning_pipeline: List[Dict]) -> str:
        """Apply cleaning steps to generation listed in cleaning_pipeline.

        Args:
            generation (str): string generated from LLM or otherwise.
            cleaning_pipeline (List[Dict]): List of Dicts that define the cleaning pipeline.
                Dictionaries MUST have the keyword "name" to be valid.
                Example: [{"name": "remove_quotes", "kwargs": {}}, {"name": "prune_ending", "kwargs": {}}]

        Returns:
            str: Clean generated string.
        """
        try:
            for cleaner in cleaning_pipeline:
                if "name" not in cleaner or cleaner["name"] not in SUPPORTED_CLEANERS:
                    raise ValueError(
                        f"Cleaning pipeline step {cleaner} must have a name, or must be in SUPPORTED_CLEANERS."
                    )

                func = SUPPORTED_CLEANERS[cleaner["name"]]

                kwargs = cleaner.get("kwargs", {})
                func = func(**kwargs)  # instantiate the cleaner with the kwargs

                # apply all the filters for the specific task.
                generation = func.apply(generation=generation)

            return generation

        except Exception as E:
            bt.logging.error(f"Failed to apply cleaning pipeline. {E}")
            return generation
