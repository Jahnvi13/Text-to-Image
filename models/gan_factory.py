from models import gan_cls

class gan_factory(object):

    @staticmethod
    def generator_factory():
        return gan_cls.generator()

    @staticmethod
    def discriminator_factory():
        return gan_cls.discriminator()