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

        if self.__nb_gus != self.__GU.shape[0]:
            for name, attr in xs.filter_variables(self, var_type='variable').items():
                dims = np.array(attr.metadata.get('dims')).flatten()
                var_resized = None
                if np.all(dims == 'GU') and len(dims.shape) < 3:
                    var = getattr(self, name)
                    if len(dims) == 1 and var.shape[0] != self.__GU.shape[0]:
                        # print(step, name, self.__class__, var.shape, self.__GU.shape)
                        var_resized = np.append(var, np.zeros(self.__GU.shape[0] - var.shape[0], dtype=var.dtype))
                        setattr(self, name, var_resized)
                    elif len(dims) == 2 and var.shape != (self.__GU.shape[0], self.__GU.shape[0]):
                        # print(step, name, self.__class__, var.shape, self.__GU.shape)
                        # print(var_resized)
                        var_resized = np.vstack((var, np.zeros((self.__GU.shape[0] - var.shape[0], var.shape[1]), dtype=var.dtype)))
                        # print(var_resized)
                        var_resized = np.hstack((var_resized, np.zeros((self.__GU.shape[0], self.__GU.shape[0] - var.shape[0]), dtype=var.dtype)))
                        # print(var_resized)
                    if var_resized is not None:
                        setattr(self, name, var_resized)

            self.__nb_gus = self.__GU.shape[0]

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

        if self.__nb_cus != self.__CU.shape[0]:
            for name, attr in xs.filter_variables(self, var_type='variable').items():
                dims = np.array(attr.metadata.get('dims')).flatten()
                var_resized = None
                if np.all(dims == 'CU') and len(dims.shape) < 3:
                    var = getattr(self, name)
                    if len(dims) == 1 and var.shape[0] != self.__CU.shape[0]:
                        # print(step, name, self.__class__, var.shape, self.__CU.shape)
                        var_resized = np.append(var, np.zeros(self.__CU.shape[0] - var.shape[0], dtype=var.dtype))
                        setattr(self, name, var_resized)
                    elif len(dims) == 2 and var.shape != (self.__CU.shape[0], self.__CU.shape[0]):
                        # print(step, name, self.__class__, var.shape, self.__CU.shape)
                        var_resized = np.vstack((var, np.zeros((self.__CU.shape[0] - var.shape[0], var.shape[1]), dtype=var.dtype)))
                        var_resized = np.hstack((var_resized, np.zeros((self.__CU.shape[0],1), dtype=var.dtype)))
                        # print(var_resized)
                    if var_resized is not None:
                        setattr(self, name, var_resized)

            self.__nb_cus = self.__CU.shape[0]

    @xs.runtime(args=('nsteps', 'step', 'step_start', 'step_end', 'step_delta'))
    def run_step(self, nsteps, step, step_start, step_end, step_delta):
        self._resize(step)
        self.step(nsteps, step, step_start, step_end, step_delta)

    @abc.abstractmethod
    def step(self, nsteps, step, step_start, step_end, step_delta):
        pass
