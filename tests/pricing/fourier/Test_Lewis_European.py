import unittest

from fypy.model.levy.MertonJD import MertonJD
from fypy.pricing.fourier.LewisEuropeanPricer import LewisEuropeanPricer
from fypy.model.levy.BlackScholes import *
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward
from fypy.pricing.analytical.black_scholes import black76_price


class Test_Lewis_European(unittest.TestCase):
    def test_price_black_scholes(self):
        S0 = 100
        r = 0.03
        q = 0.01
        T = 1
        sigma = 0.2

        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)
        model = BlackScholes(sigma=sigma, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = LewisEuropeanPricer(model=model, N=2 ** 8)

        for is_call in [True, False]:
            for K in [S0 - 10, S0, S0 + 10]:
                price = pricer.price(T=T, K=K, is_call=is_call)
                true_price = black76_price(F=fwd(T), K=K, is_call=is_call, vol=sigma, disc=disc_curve(T), T=T)
                self.assertAlmostEqual(price, true_price, 13)

        # pres = pricer.price_strikes(T=T, K=np.array([S0-10, S0, S0+10]), is_calls=np.array([True, False, True]))

    def test_mjd(self):
        S0 = 100
        r = 0.05
        q = 0.01
        T = 1

        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)

        model = MertonJD(sigma=0.12, lam=0.4, muj=-0.12, sigj=0.18, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = LewisEuropeanPricer(model=model, N=2 ** 8)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 8.67568463542607, 11)


if __name__ == '__main__':
    unittest.main()
