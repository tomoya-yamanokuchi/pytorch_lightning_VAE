from .variational_autoencoder.LitVariationalAutoencoder import LitVariationalAutoencoder
from .disentangled_sequential_variational_autoencoder.LitDisentangledSequentialVariationalAutoencoder import LitDisentangledSequentialVariationalAutoencoder

class ModelFactory:
    def create(self, name: str, **kwargs):
        if    name == "vae"      : return LitVariationalAutoencoder(**kwargs)
        if    name == "dsvae"    : return LitDisentangledSequentialVariationalAutoencoder(**kwargs)
        else                     : raise NotImplementedError()
