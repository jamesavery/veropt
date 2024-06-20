import xarray as xr
import numpy as np
import math
from veropt import ObjFunction
from veropt.run_veros import activate_venv, run_veros


class MLD1ObjFun(ObjFunction):
    def __init__(self, criterion, verospath="/opt/code/ocean/veros", datapath='/data/ocean/veropt_results', 
                 experiment='mld_experiment1', float_type="float32"):
        self.venvpath     = f"{verospath}/venv"
        self.datapath     = datapath
        self.experiment   = experiment
        self.runpath      = f"{datapath}/{experiment}"
        self.setup_script = f"{experiment}.py"
        self.criterion    = criterion
        
        bounds = [1e-3,1]
        n_params = 1
        n_objs = 1
        init_vals = 0.2
        stds = None

        super().__init__(self.mldObjFunc, bounds, n_params, n_objs, init_vals, stds)


    def optimized_filename(self,c_k):
        filename = f"4deg_nz115_ck{c_k}_30yr.averages.nc"
        return filename


    def save_mld_file(self,c_k):
        filename = f"4deg_nz115_ck{c_k}_30yr_mld_{self.criterion}.nc"
        return filename


    def mldObjFunc(self,c_k):
        
        # TODO: Make not hacky
        c_k = c_k.numpy()[0][0][0]

        run_veros(self.runpath, self.setup_script, f"c_k={c_k}", self.venvpath)
        
        optimized_filename = self.optimized_filename(c_k)
        print(f"{self.runpath}/{optimized_filename}")
        optimized_dataset = xr.open_dataset(f"{self.runpath}/{optimized_filename}")

        #target_filepath = "/data/ocean/simulated_target_mld/"
        #target_filename = f"target_mld_{self.criterion}.nc"
        target_filepath = "/data/ocean/ifremer/mixed_layer_depth"
        target_filename = f"IFREMER_mld_tropics.nc"
        target_dataset = xr.open_dataset(f"{target_filepath}/{target_filename}")

        mld_filename = self.save_mld_file(c_k)
        save_mld_file = f"{self.runpath}/{mld_filename}"

        y = calc_y(optimized_dataset, target_dataset, save_mld_file, self.criterion)

        return y


# calculate mld as in MLD climatology from IFREMER (T smaller by 0.2 deg from its value at 10m)
def calc_mld_temp(temp, varname="temp"):
    # make new variable containing depth
    tempt = temp.assign({"mld_help": (temp[varname].dims, \
                                      np.broadcast_to(temp.zt.values[None,:,None,None], temp[varname].shape))})
    
    # get the two temp and depth values above and below critical value
    t_surf = tempt[varname].sel(zt=-10, method="nearest")
    t_crit = t_surf - 0.2
    ds_above_t = tempt.where((tempt[varname] - t_crit > 0))
    ds_above = ds_above_t.where((ds_above_t.mld_help == \
                                 ds_above_t.mld_help.min(dim="zt", skipna=True))).mean(dim="zt", skipna=True)
    ds_below_t = tempt.where((tempt[varname] - t_crit < 0))
    ds_below = ds_below_t.where((ds_below_t.mld_help == \
                                 ds_below_t.mld_help.max(dim="zt", skipna=True))).mean(dim="zt", skipna=True)
    
    # linear interpolation of depth
    mld = (t_crit - ds_below[varname]) / (ds_above[varname] - ds_below[varname]) \
        * (ds_above.mld_help - ds_below.mld_help) + ds_below.mld_help
    
    return mld


# calculate MLD based on diffusivity above background
def calc_mld_diff(ds, varname="kappaH"):
    # make new variable containing depth
    ds_help = ds.assign({"mld_help": (ds[varname].dims, \
                                      np.broadcast_to(ds.zw.values[None,:,None,None], ds[varname].shape))})
    
    # get depth where diffusivity is above background
    background_diff = 2.e-4
    ds_background_diff = ds_help.where((ds_help[varname] < background_diff))
    mld = ds_background_diff.mld_help.max(dim="zw", skipna=True)
    
    return mld


def calc_y(optimized_dataset, target_dataset, save_mld_file, criterion):
    if criterion == 'temp':
        mld_ds = calc_mld_temp(xr.combine_by_coords([optimized_dataset.temp]))
    
    elif criterion == 'diff':
        mld_ds = calc_mld_diff(xr.combine_by_coords([optimized_dataset.kappaH]))

    mld_ds.to_netcdf(save_mld_file)
    
    # time average between simulated years [20,30]
    mld = mld_ds.isel(Time=slice(-10,None)).sel(yt=slice(-23,23)).mean(dim='Time').values
    mld = np.where(mld == 0., -1., mld)

    target_mld = target_dataset.mld.values
    target_mld = np.where(target_mld == 0., -1., target_mld)
    
    MSE = np.nanmean(np.square((mld-target_mld)/target_mld))

    return -np.sqrt(MSE)
