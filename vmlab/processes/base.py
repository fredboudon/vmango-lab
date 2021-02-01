import xsimlab as xs
import abc
import numpy as np


@xs.process
class BaseGrowthUnitProcess(abc.ABC):
    """Base class for all GU based processes
    """

    __GU = xs.global_ref('GU')
    __nb_gus = 0

    def _resize(self, step):

        nb_gus_now = self.__GU.shape[0]

        if self.__nb_gus > nb_gus_now:
            raise ValueError(f'Number GUs decreased from {self.__nb_gus} to {nb_gus_now}.')

        if self.__nb_gus < nb_gus_now:  # decrease in n should not happen
            for name, attr in xs.filter_variables(self, var_type='variable').items():
                dims = np.array(attr.metadata.get('dims')).flatten()
                # TODO: use default as fill value
                # default = attr.metadata.get('default')
                # print(default)
                var_resized = None
                if len(dims) and np.all(dims == 'GU') and len(dims.shape) < 3:
                    var = getattr(self, name)
                    if len(dims) == 1 and var.shape[0] != nb_gus_now:
                        var_resized = np.zeros(nb_gus_now, dtype=var.dtype)
                        var_resized[0:self.__nb_gus] = var
                        setattr(self, name, var_resized)
                    elif len(dims) == 2 and var.shape != (nb_gus_now, nb_gus_now):
                        var_resized = np.zeros((nb_gus_now, nb_gus_now), dtype=var.dtype)
                        var_resized[0:self.__nb_gus, 0:self.__nb_gus] = var

                    if var_resized is not None:
                        setattr(self, name, var_resized)

            self.__nb_gus = nb_gus_now

    @xs.runtime(args=('nsteps', 'step', 'step_start', 'step_end', 'step_delta'))
    def run_step(self, nsteps, step, step_start, step_end, step_delta):
        self._resize(step)
        self.step(nsteps, step, step_start, step_end, step_delta)

    @abc.abstractmethod
    def step(self, nsteps, step, step_start, step_end, step_delta):
        pass


@xs.process
class BaseCarbonUnitProcess(abc.ABC):
    """Base class for all CU based processes
    """

    __CU = xs.global_ref('CU')
    __nb_cus = 0

    def _resize(self, step):

        nb_cus_now = self.__CU.shape[0]

        if self.__nb_cus > nb_cus_now:
            raise ValueError(f'Number CUs decreased from {self.__nb_cus} to {nb_cus_now}.')

        if self.__nb_cus < nb_cus_now:  # decrease in n should not happen
            for name, attr in xs.filter_variables(self, var_type='variable').items():
                dims = np.array(attr.metadata.get('dims')).flatten()
                var_resized = None
                if len(dims) and np.all(dims == 'CU') and len(dims.shape) < 3:
                    var = getattr(self, name)
                    if len(dims) == 1 and var.shape[0] != nb_cus_now:
                        var_resized = np.zeros(nb_cus_now, dtype=var.dtype)
                        var_resized[0:self.__nb_cus] = var
                    elif len(dims) == 2 and var.shape != (nb_cus_now, nb_cus_now):
                        var_resized = np.zeros((nb_cus_now, nb_cus_now), dtype=var.dtype)
                        var_resized[0:self.__nb_gus, 0:self.__nb_gus] = var

                    if var_resized is not None:
                        setattr(self, name, var_resized)

            self.__nb_cus = nb_cus_now

    @xs.runtime(args=('nsteps', 'step', 'step_start', 'step_end', 'step_delta'))
    def run_step(self, nsteps, step, step_start, step_end, step_delta):
        self._resize(step)
        self.step(nsteps, step, step_start, step_end, step_delta)

    @abc.abstractmethod
    def step(self, nsteps, step, step_start, step_end, step_delta):
        pass
