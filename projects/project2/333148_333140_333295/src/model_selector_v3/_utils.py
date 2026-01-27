class DataMapper:
    def __init__(self, config):
        self.family_defs = config.get('family_definitions', {})
        self.mapping = config.get('data_mapping', {})

    def get_family(self, model_type):
        m_type = model_type.lower()
        for family, keys in self.family_defs.items():
            if any(k in m_type for k in keys):
                return family
        return None

    def get_X_for_model(self, model_type, X_dict):
        family = self.get_family(model_type)
        if not family:
            # Fallback jeśli nie rozpoznano rodziny
            return next(iter(X_dict.values()))

        data_key = self.mapping.get(family)
        if data_key and data_key in X_dict:
            return X_dict[data_key]

        # Fallback jeśli klucz nie istnieje
        return next(iter(X_dict.values()))