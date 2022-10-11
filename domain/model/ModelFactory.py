from .variational_autoencoder.LitVariationalAutoencoder import LitVariationalAutoencoder
from .disentangled_sequential_variational_autoencoder.LitDisentangledSequentialVariationalAutoencoder import LitDisentangledSequentialVariationalAutoencoder

class ModelFactory:
    def create(self, name: str):
        prefix = name.split("_")[0]
        if  prefix == "vae"  : return LitVariationalAutoencoder
        if  prefix == "dsvae": return LitDisentangledSequentialVariationalAutoencoder
        else                 : raise NotImplementedError()
