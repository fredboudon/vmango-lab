import vmlab
from vmlab.models import vmango

model = vmango.drop_processes('geometry')

setup = vmlab.create_setup(
    model=model,
    start_date='2003-06-01',
    end_date='2004-06-01',
    setup_toml='vmango.toml',
    output_vars={
        'harvest__nb_fruit_harvested': None
    }
)

ds = vmlab.run(setup, model)
