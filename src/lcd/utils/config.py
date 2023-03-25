import collections
import importlib
import os
import pickle


def import_class(_class):
    if type(_class) is not str:
        return _class
    ## 'diffusion' on standard installs
    repo_name = __name__.split(".")[0]
    ## eg, 'utils'
    module_name = ".".join(_class.split(".")[:-1])
    ## eg, 'Renderer'
    class_name = _class.split(".")[-1]
    ## eg, 'diffusion.utils'
    module = importlib.import_module(f"{repo_name}.{module_name}")
    ## eg, diffusion.utils.Renderer
    _class = getattr(module, class_name)
    print(f"[ utils/config ] Imported {repo_name}.{module_name}:{class_name}")
    return _class


class AttriDict(dict):
    """
    A dict which is accessible via attribute dot notation
    https://stackoverflow.com/a/41514848
    https://stackoverflow.com/a/14620633
    """

    DICT_RESERVED_KEYS = list(vars(dict).keys())

    def __init__(self, *args, **kwargs):
        """
        :param args: multiple dicts ({}, {}, ..)
        :param kwargs: arbitrary keys='value'
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, attr):
        if attr not in AttriDict.DICT_RESERVED_KEYS:
            return self.get(attr)
        return getattr(self, attr)

    def __setattr__(self, key, value):
        if key == "__dict__":
            super().__setattr__(key, value)
            return
        if key in AttriDict.DICT_RESERVED_KEYS:
            raise AttributeError("You cannot set a reserved name as attribute")
        self.__setitem__(key, value)

    def __copy__(self):
        return self.__class__(self)

    def copy(self):
        return self.__copy__()


class Config(collections.abc.Mapping):  # type: ignore
    def __init__(self, _class, verbose=True, savepath=None, device=None, **kwargs):
        self._class = import_class(_class)
        self._device = device
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        if verbose:
            print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
            pickle.dump(self, open(savepath, "wb"))
            print(f"[ utils/config ] Saved config to: {savepath}\n")

    def __repr__(self):
        string = f"\n[utils/config ] Config: {self._class}\n"
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            string += f"    {key}: {val}\n"
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __getattr__(self, attr):
        if attr == "_dict" and "_dict" not in vars(self):
            self._dict = {}
            return self._dict
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def __call__(self, *args, **kwargs):
        instance = self._class(*args, **kwargs, **self._dict)
        if self._device:
            instance = instance.to(self._device)
        return instance
