def retrieveCurves(globals):
    from openalea.plantgl.all import BezierCurve2D
    # Determine the set of curve representing axis at different time.
    # Look for object in global namespace name axisX
    curves = [(n,v) for n,v in list(globals.items()) if 'axis' in n and type(v) == BezierCurve2D ]

    # sort curves according to their names
    for n, v in curves:
        v.name = n
    curves = [v for n,v in curves]
    curves.sort(key=lambda x : x.name)
    return curves


def ProfileInterpolation(curves, knotlist = None, degree = 3, resolution = 10):
    from openalea.plantgl.all import Point4Matrix, NurbsPatch, NurbsCurve2D, BezierCurve2D
    nbcurves = len(curves)
    if knotlist is None: knotlist = [i/float(nbcurves-1) for i in range(nbcurves)]
    k = [knotlist[0] for i in range(degree-1)]+knotlist+[knotlist[-1] for i in range(degree-1)]
    pts = [[(i.x,i.y,0,1) for i in c.ctrlPointList] for c in curves]
    ppts = Point4Matrix(pts)
    p = NurbsPatch(ppts,udegree=degree,vdegree=3)
    def getSectionAt(t):
        section = p.getIsoUSectionAt(t)
        res = NurbsCurve2D([(i.x,i.y,i.w) for i in section.ctrlPointList], section.knotList,section.degree)
        res.stride = resolution
        return res
    p.getAt = getSectionAt
    return p


class SymbolManager(object):
    def __init__(self, axiscurves, knotlist, maxstage, section, length, dlength, radius = 1, radiusvariation = None):
        self.axiscurves = axiscurves
        self.axisfunc = ProfileInterpolation(axiscurves, knotlist)
        self.maxstage = float(maxstage)
        self.section = section
        self.length = length
        self.dlength = dlength
        self.radius = radius
        self.radiusvariation = radiusvariation
        self.leafsmbdb = dict()
        self.leafsmbfinal = self.sweepSymbol(axiscurves[-1])
        self.leafsmbfinal.name = 'finalleaf'

    def sweepSymbol(self, path):
        from openalea.plantgl.all import PglTurtle
        t = PglTurtle()
        t.start()
        return t.startGC().sweep(path, self.section, self.length, self.dlength, self.radius, self.radiusvariation).stopGC().getScene()[0].geometry

    def __call__(self, nstage = None):
        if nstage is None: return self.leafsmbfinal
        nstage = round(nstage,1)
        if nstage >= self.maxstage :
            return self.leafsmbfinal
        try:
            return self.leafsmbdb[nstage]
        except KeyError as e:
            cleafsmb = self.sweepSymbol(self.axisfunc.getAt(min(nstage / self.maxstage, 1.)))
            self.leafsmbdb[nstage] = cleafsmb
            cleafsmb.name = 'leaf_'+str(nstage).replace('.','_')
            return cleafsmb
