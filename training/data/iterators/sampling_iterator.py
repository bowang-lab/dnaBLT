from typing import Any, Generator, Iterator, Dict
import numpy as np

class SamplingIterator:
    def __init__(
        self,
        *,
        source_to_weight: Dict[str, float],
        source_to_iterator: Dict[str, Iterator],
        seed: int = None
    ):
        """
        Creates an iterator that samples from multiple sources based on weights.
        
        Args:
            source_to_weight: Dictionary mapping source names to their sampling weights
            source_to_iterator: Dictionary mapping source names to their iterators
            seed: Optional random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.source_to_weight = source_to_weight
        self.source_to_iterator = source_to_iterator
        
        # Validate that both dictionaries have the same keys
        if set(source_to_weight.keys()) != set(source_to_iterator.keys()):
            raise ValueError("source_to_weight and source_to_iterator must have the same keys")

    def __iter__(self) -> Generator[Any, None, None]:
        """
        Creates and returns an iterator that samples from the source iterators
        according to their weights.
        """
        possible_sources = list(self.source_to_weight.keys())
        weights = [self.source_to_weight[source] for source in possible_sources]
        
        # Create iterators for each source
        source_to_python_iter = {
            source: iter(self.source_to_iterator[source]) 
            for source in possible_sources
        }
        
        # Normalize weights for numpy's choice function
        norm_weights = np.array(weights) / np.sum(weights)
        n_sources = len(possible_sources)
        
        while True:
            # Sample a source based on weights
            source_choice = possible_sources[self.rng.choice(n_sources, p=norm_weights)]
            try:
                yield next(source_to_python_iter[source_choice])
            except StopIteration:
                # If an iterator is exhausted, remove it and renormalize weights
                del source_to_python_iter[source_choice]
                possible_sources.remove(source_choice)
                weights = [self.source_to_weight[source] for source in possible_sources]
                
                if not possible_sources:
                    return  # All iterators exhausted
                    
                norm_weights = np.array(weights) / np.sum(weights)
                n_sources = len(possible_sources)
    
