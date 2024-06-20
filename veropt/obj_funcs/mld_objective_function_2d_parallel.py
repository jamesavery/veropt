import xarray as xr
import numpy as np
import math
from veropt import ObjFunction
from veropt.run_veros_3d import activate_venv, run_veros, run_two


class MLD1ObjFun(ObjFunction):
    def __init__(self, param_list, verospath="/opt/code/ocean/veros", datapath='/data/ocean/veropt_results', 
                 experiment='mld_experiment7', float_type="float32"):
        self.venvpath     = f"{verospath}/venv"
        self.datapath     = datapath
        self.experiment   = experiment
        self.runpath      = f"{datapath}/{experiment}"
        self.setup_script = f"{experiment}.py"
        self.param_list   = param_list
        
        n_params = 2
        bounds_lower = [5e-2, 5e-2]
        bounds_upper = [1, 1]
        bounds = [bounds_lower, bounds_upper]
        
        n_objs = 1
        init_vals = None
        stds = None

        super().__init__(self.mldObjFunc, bounds, n_params, n_objs, init_vals, stds)


    def optimized_filename(self, params):
        c_k, c_eps = params
        filename = f"1deg_ck{c_k}_ceps{c_eps}.averages"
        return filename
    

    def correct_1deg_coords(self, optimized_filename, datapath, experiment):
        ds = xr.open_dataset(f'{datapath}/{experiment}/{optimized_filename}.nc')
        ds.coords['xt'] = np.concatenate([np.arange(90.5, 360.5, 1.), np.arange(0.5, 90.5, 1.)])
        ds = ds.sortby(ds.xt)
        ds.coords['xu'] = np.concatenate([np.arange(90.5, 360.5, 1.), np.arange(0.5, 90.5, 1.)])
        ds = ds.sortby(ds.xu)
        ds.to_netcdf(f'{datapath}/{experiment}/{optimized_filename}_corr_coords.nc')


    def mldObjFunc(self, new_x):
        
        # TODO: Make not hacky
        # index 0: latest opt step, should be 0
        # index 1: index of point within opt step (we run 16 points in parallel per opt step)
        # index 2: param index (we run optimization on two params: c_k and c_eps)
        c_k0, c_eps0 = new_x.numpy()[0][0][0], new_x.numpy()[0][0][1]
        c_k1, c_eps1 = new_x.numpy()[0][1][0], new_x.numpy()[0][1][1]

        params0 = c_k0, c_eps0
        params1 = c_k1, c_eps1

        setup_args0 = f"c_k={c_k0}=c_eps={c_eps0}"
        setup_args1 = f"c_k={c_k1}=c_eps={c_eps1}"

        run_two(self.runpath, self.setup_script, setup_args0, setup_args1, self.venvpath)
        
        optimized_filename0 = self.optimized_filename(params0)
        optimized_filename1 = self.optimized_filename(params1)

        self.correct_1deg_coords(optimized_filename0, self.datapath, self.experiment)
        self.correct_1deg_coords(optimized_filename1, self.datapath, self.experiment)

        optimized_dataset0 = xr.open_dataset(f"{self.runpath}/{optimized_filename0}_corr_coords.nc")
        optimized_dataset1 = xr.open_dataset(f"{self.runpath}/{optimized_filename1}_corr_coords.nc")
        # optimized_dataset0 = xr.open_dataset(f"{self.runpath}/{optimized_filename0}.nc")
        # optimized_dataset1 = xr.open_dataset(f"{self.runpath}/{optimized_filename1}.nc")

        target_filepath = "/data/ocean/ifremer/mixed_layer_depth"
        target_filename = f"IFREMER_mld_rho_50S50N"
        # target_filepath = "/data/ocean/simulated_target_mld"
        # target_filename = "1deg_target_mld_rho"
        target_dataset = xr.open_dataset(f"{target_filepath}/{target_filename}.nc")

        y0 = calc_y(optimized_dataset0, target_dataset)
        y1 = calc_y(optimized_dataset1, target_dataset)

        with open(self.param_list, "a") as file:
            file.write(f"\n{c_k0},{c_eps0},{y0}")
            file.write(f"\n{c_k1},{c_eps1},{y1}")

        y = [y0, y1]

        return y


def calc_y(optimized_dataset, target_dataset):    
    mld = optimized_dataset.mld.isel(Time=slice(-2,None)).sel(yt=slice(-50,50)).values
    mld = np.where(mld == 0., -1., mld)

    target_mld = target_dataset.mld.values
    target_mld = np.where(target_mld == 0., -1., target_mld)
    
    MSE = np.nanmean(np.square((mld-target_mld)/target_mld))

    return -np.sqrt(MSE)
