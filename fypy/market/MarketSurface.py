from typing import Dict, List, Optional
from fypy.market.MarketSlice import MarketSlice, StrikeFilter
from fypy.volatility.implied import ImpliedVolCalculator

from abc import ABC, abstractmethod


class SliceFilter(ABC):
    @abstractmethod
    def keep(self, market_slice: MarketSlice) -> bool:
        raise NotImplementedError


class SliceFilters(SliceFilter):
    def __init__(self, filters: Optional[List[SliceFilter]] = None):
        self._filters = filters or []

    def add_filter(self, slice_filter: SliceFilter):
        self._filters.append(slice_filter)

    def keep(self, market_slice: MarketSlice) -> bool:
        for slice_filter in self._filters:
            if not slice_filter.keep(market_slice):
                return False
        return True


class MarketSurface(object):
    def __init__(self, slices: Dict[float, MarketSlice] = None):
        """
        Container class for an option price surface, composed of individual market slices, one per tenor
        :param slices: dict: {float, MarketSlice}, contains all slices (you can add more later)
        """
        self._slices = slices or {}

    def add_slice(self, ttm: float, market_slice: MarketSlice):
        """
        Add a new market slice (overwrites if same ttm already exists in surface)
        :param ttm: float, time to maturity of the slice (tenor)
        :param market_slice: MarketSlice, the market slice prices object
        :return: self
        """
        self._slices[ttm] = market_slice
        return self

    @property
    def slices(self) -> Dict[float, MarketSlice]:
        """ Access all slices """
        return self._slices

    @property
    def ttms(self):
        """ Get the ttms in the surface """
        return self._slices.keys()

    @property
    def num_slices(self) -> int:
        """ Get number of slice in surface """
        return len(self._slices)

    def fill_implied_vols(self, calculator: ImpliedVolCalculator):
        """
        Fill the implied vols given a calculator. Fills in for each of bid,mid,ask, but only those that have
        corresponding prices
        :param calculator: ImpliedVolCalculator, a calculator used to fill in the vols from prices
        :return: None
        """
        for slice_ in self.slices.values():
            slice_.fill_implied_vols(calculator)

    def filter_slices(self,
                      slice_filter: SliceFilter,
                      strike_filter: Optional[StrikeFilter] = None) -> 'MarketSurface':
        filtered_surface = MarketSurface()
        for ttm, market_slice in self.slices.items():
            if slice_filter.keep(market_slice):
                if strike_filter:
                    filtered_surface.add_slice(ttm, market_slice=market_slice.filter_strikes(strike_filter))
                else:
                    filtered_surface.add_slice(ttm, market_slice=market_slice)

        return filtered_surface
