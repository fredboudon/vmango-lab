import pandas as pd
import io

import vmlab
from vmlab.models import fruit_model

# simple tree to debug and test carbon coef and allocation dynamics
tree = pd.read_csv(io.StringIO("""
id,parent_id,arch_dev__pot_flowering_date,arch_dev__pot_nb_inflo,arch_dev__pot_nb_fruit,growth__nb_leaf
0,NA,NAT,0,0,0
1,0,NAT,0,0,10
2,1,2003-09-02,1,1,10
3,1,2003-09-01,2,2,10
"""))

model = fruit_model.drop_processes('geometry')

setup = vmlab.create_setup(
    model=model,
    tree=tree,
    start_date='2003-09-01',
    end_date='2004-06-01',
    setup_toml='fruit_model.toml'
)

vmlab.run(
    setup,
    model,
    progress=True
)
