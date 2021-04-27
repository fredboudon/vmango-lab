import vmlab
from vmlab.models import vmango

start_date = '2003-06-01'
end_date = '2004-06-01'
setup = vmlab.create_setup(
    model=vmango,
    start_date=start_date,
    end_date=end_date,
    setup_toml='share/setup/vmango.toml',
    current_cycle=3,
    input_vars={
        'geometry__interpretation_freq': 0
    },
    output_vars=None
)

ds = vmlab.run(setup, vmango)
