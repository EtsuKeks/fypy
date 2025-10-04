import warnings
warnings.filterwarnings('ignore')

from fypy.model.levy.BlackScholes import BlackScholes
from fypy.model.sv.Heston import Heston
from fypy.model.slv.Sabr import Sabr
from fypy.calibrate.FourierModelCalibrator import FourierModelCalibrator
from fypy.calibrate.SabrModelCalibrator import SabrModelCalibrator

def calibrate_black_scholes(surface, prev_model) -> BlackScholes:
    if prev_model:
        model = BlackScholes(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve, sigma=prev_model.vol)
    else:
        model = BlackScholes(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve)
    FourierModelCalibrator(surface=surface, do_vega_weight=False).calibrate(model=model)
    return model


def calibrate_heston(surface, prev_model) -> Heston:
    if prev_model:
        model = Heston(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve,
                      v_0=prev_model.v_0, theta=prev_model.theta, kappa=prev_model.kappa, 
                      sigma_v=prev_model.sigma_v, rho=prev_model.rho)
    else:
        model = Heston(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve)
    FourierModelCalibrator(surface=surface, do_vega_weight=False).calibrate(model=model)
    return model


def calibrate_sabr(surface, prev_model):
    if prev_model:
        model = Sabr(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve,
                    v_0=prev_model.v_0, alpha=prev_model.alpha, beta=prev_model.beta, rho=prev_model.rho)
    else:
        model = Sabr(forwardCurve=surface.forward_curve, discountCurve=surface.discount_curve)
    SabrModelCalibrator(surface=surface).calibrate(model=model)
    return model
