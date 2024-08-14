from src.services.uci_service.rice_data import RiceData
from ucimlrepo import fetch_ucirepo 

class TestInitRiceData:

    def test_get_feature(self):
        riceData = RiceData()
        actual = riceData.getFeatures()

        rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
        expected = rice_cammeo_and_osmancik.data.features
        assert expected.equals(actual)

    def test_get_target(self):
        riceData = RiceData()
        actual = riceData.getTargets()
        rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
        expected = rice_cammeo_and_osmancik.data.targets
        assert expected.equals(actual)
