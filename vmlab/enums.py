from . import DotDict


# Type of positions of units
Position = DotDict({
    'LATERAL': 0.,
    'APICAL': 1.
})

# Type of GU
Nature = DotDict({
    'VEGETATIVE': 0.,
    'PURE_FLOWER': 1.,
    'FRUITING': 2.
})
