from veropt import BayesOptimiser
from veropt.obj_funcs.mld_objective_function import *
from veropt.gui import veropt_gui

n_init_points = 1
n_bayes_points = 1
n_evals_per_step = 1

obj_func = MLD1ObjFun()

optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)

veropt_gui.run(optimiser)