from configparser import ConfigParser

class Configurable:
    def __init__(self, config_file, extra_args):
        config = ConfigParser()
        config.read(config_file)
        if extra_args is not None and len(extra_args) > 0:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2],extra_args[1::2])])
            for section in config.sections():
                for k, v in config.items(section):
                    if k in extra_args.keys():
                        v = type(v)(extra_args[k])
                        config.set(section,k,v)
        self._config = config
            
        for section in config.sections():
            for k,v in config.items(section):
                print(k, v,end='\t')
                pass


    @property
    def raw_data_dir(self):
        return self._config.get('Data', 'raw_data_dir')

    @property
    def save_data_dir(self):
        return self._config.get('Data', 'save_data_dir')

    @property
    def data_name(self):
        return self._config.get('Data', 'data_name')
    
    @property
    def save_model_path(self):
        return self._config.get('Data', 'save_model_path')

    @property
    def load_dir(self):
        return self._config.get('Data', 'load_dir')

    @property
    def load_model_path(self):
        return self._config.get('Data', 'load_model_path')

    @property
    def save_dir(self):
        return self._config.get('Data', 'save_dir')        

    @property
    def dataseed(self):
        return int(self._config.get('generate', 'dataseed'))

    @property
    def m(self):
        return int(self._config.get('generate', 'm'))

    @property
    def num(self):
        return int(self._config.get('generate', 'num'))

    @property
    def k(self):
        return int(self._config.get('generate', 'k'))
    
    @property
    def every_linear(self):
        return bool(int(self._config.get('Network', 'every_linear')))
   
    @property
    def n_layer(self):
        return int(self._config.get('Network', 'n_layer'))

    @property
    def nchannel(self):
        return int(self._config.get('Network', 'nchannel'))

    @property
    def channel(self):
        return int(self._config.get('Network', 'channel'))

    @property
    def kdim(self):
        return int(self._config.get('Network', 'kdim'))

    @property
    def deck(self):
        return int(self._config.get('Network', 'deck'))

    @property
    def dropout(self):
        return float(self._config.get('Network', 'dropout'))

    @property
    def routit(self):
        return int(self._config.get('Network', 'routit'))

    @property
    def tau(self):
        return float(self._config.get('Network', 'tau'))

    @property
    def nbsz(self):
        return int(self._config.get('Network', 'nbsz'))

    @property
    def include_self(self):
        return bool(int(self._config.get('Network', 'include_self')))

    @property
    def threshold(self):
        return float(self._config.get('Network', 'threshold'))

    @property
    def resample(self):
        return bool(int(self._config.get('Network', 'resample')))


    @property
    def jump(self):
        return bool(int(self._config.get('Network', 'jump')))

    @property
    def seed(self):
        return int(self._config.get('Network', 'seed'))

    @property
    def gnn(self):
        return str(self._config.get('Network', 'gnn'))

    @property
    def lr(self):
        return float(self._config.get('Optimizer', 'lr'))

    @property
    def beta1(self):
        return float(self._config.get('Optimizer', 'beta1'))   

    @property
    def beta2(self):
        return float(self._config.get('Optimizer', 'beta2'))

    @property
    def epsilon(self):
        return float(self._config.get('Optimizer', 'epsilon'))

    @property
    def reg(self):
        return float(self._config.get('Optimizer', 'reg'))

    @property
    def clip(self):
        return float(self._config.get('Optimizer', 'clip'))

    @property
    def epoch(self):
        return int(self._config.get('Run', 'epoch'))

    @property
    def early_stop(self):
        return int(self._config.get('Run', 'early_stop'))

    @property
    def alpha(self):
        return float(self._config.get('Loss', 'alpha'))                 

    @property
    def nrank(self):
        return int(self._config.get('Loss', 'nrank'))                 

    @property
    def mutual_hidden(self):
        return int(self._config.get('Loss', 'mutual_hidden'))

    @property
    def ind_channel(self):
        return int(self._config.get('Loss', 'ind_channel'))                 

    @property
    def mutual_batch(self):
        return int(self._config.get('Loss', 'mutual_batch'))                 

    @property
    def mutual_beta(self):
        return float(self._config.get('Loss', 'mutual_beta'))              