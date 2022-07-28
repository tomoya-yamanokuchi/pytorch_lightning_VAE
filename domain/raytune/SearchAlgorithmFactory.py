from ray.tune import suggest
from ray.tune.suggest import ConcurrencyLimiter

class SearchAlgorithmFactory:
    def create(self, name: str, max_concurrent: int, **kwargs):
        search_alg = suggest.SEARCH_ALG_IMPORT[name]()(**kwargs)
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        # search_alg = suggest.SEARCH_ALG_IMPORT[name]()(**kwargs)
        return search_alg

