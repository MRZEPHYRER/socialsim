from analysis.demography import DemographyAnalysis
from analysis.economy import EconomyAnalysis
from analysis.fertility import FertilityAnalysis
from analysis.firm import FirmAnalysis
from analysis.time_semantics import build_analysis_metadata


class Analyzer(
    DemographyAnalysis,
    EconomyAnalysis,
    FertilityAnalysis,
    FirmAnalysis,
):

    def __init__(self, world):
        self.world = world

    def time_semantics_metadata(self, start_global_step=0):
        return build_analysis_metadata(
            start_global_step,
            len(getattr(self.world, "population_history", [])),
        )
