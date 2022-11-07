from .variational_autoencoder.LitVariationalAutoencoder import LitVariationalAutoencoder
from .disentangled_sequential_variational_autoencoder.LitDisentangledSequentialVariationalAutoencoder import LitDisentangledSequentialVariationalAutoencoder
from .randomized_to_canonical_disentangled_sequential_variational_autoencoder.LitRandomized2CanonicalDisentangledSequentialVariationalAutoencoder import LitRandomized2CanonicalDisentangledSequentialVariationalAutoencoder


class ModelFactory:
    def create(self, name: str):
        prefix = name.split("_")[0]
        if    prefix == "vae"       : return LitVariationalAutoencoder
        elif  prefix == "dsvae"     : return LitDisentangledSequentialVariationalAutoencoder
        elif  prefix == "r2c-dsvae" : return LitRandomized2CanonicalDisentangledSequentialVariationalAutoencoder
        else                        : raise NotImplementedError()
