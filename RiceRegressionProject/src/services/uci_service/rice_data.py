from ucimlrepo import fetch_ucirepo 

class RiceData:

    def __init__(self):
        self.load_data()

    def load_data(self):
        self.rice_cammeo_and_osmancik = fetch_ucirepo(id=545)

    def getFeatures(self):
        return self.rice_cammeo_and_osmancik.data.features

    def getTargets(self):
        return self.rice_cammeo_and_osmancik.data.targets