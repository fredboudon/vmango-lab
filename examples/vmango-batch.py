import vmlab
from vmlab.models import vmango

model = vmango.drop_processes('geometry')

start_date = '2003-06-01'
end_date = '2004-06-01'
setup = vmlab.create_setup(
    model=model,
    start_date=start_date,
    end_date=end_date,
    setup_toml='share/setup/vmango.toml',
    current_cycle=3,
    input_vars=None,
    output_vars=None
)

ds = vmlab.run(
    setup,
    model,
    batch=('batch', [{'topology__seed': i} for i in range(4)])
)
