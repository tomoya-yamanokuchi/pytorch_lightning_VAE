from .variational_autoencoder.LitVariationalAutoencoder import LitVariationalAutoencoder
from .disentangled_sequential_variational_autoencoder.LitDisentangledSequentialVariationalAutoencoder import LitDisentangledSequentialVariationalAutoencoder

class ModelFactory:
    def create(self, name: str, **kwargs):
        prefix = name.split("_")[0]
        if  prefix == "vae"  : return LitVariationalAutoencoder(**kwargs)
        if  prefix == "dsvae": return LitDisentangledSequentialVariationalAutoencoder(**kwargs)
        else                 : raise NotImplementedError()
