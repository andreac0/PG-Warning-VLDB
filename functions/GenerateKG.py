from faker import Faker
import pandas as pd

class friendshipKG:

    def __init__(self, country = 'it_IT', seed = False):
        self.fake = Faker(country)
        if seed:
            self.fake.seed_instance(seed)
        self.country = country

    def get_nodes(self, n):
        nodes = []
        for i in range(n):
            nodes.append({'id':i,
                        'name':self.fake.name(),
                        'address':self.fake.address(),
                        'date':self.fake.date(),
                        'nationality': self.country})
        return pd.DataFrame(nodes)

class transportKG:

    def __init__(self, country = 'it_IT', seed = 4):
        self.fake = Faker(country)
        self.fake.seed_instance(seed)
        self.country = country

    def get_nodes(self, n):
        nodes = []
        for i in range(n):
            nodes.append({'id':i,
                        'city': self.fake.city() + ' (' + self.country.split('_')[1] + ')'})
        return pd.DataFrame(nodes)

def parse_entity(row, exclude = ['']):
    entity_desc = ''

    # .sample()
    try:
        for j in range(1,len(row.columns)):
            if row.columns[j] not in exclude:
                entity_desc += row.columns[j] + ': '+ str(row[row.columns[j]].iloc[0]) +'\n'
    # .iloc[]
    except:
        for j in range(1,len(row.index)):
            if row.index[j] not in exclude:
                entity_desc += row.index[j] + ': '+ str(row[row.index[j]]) +'\n'

    return 'ENTITY: '+ entity_desc