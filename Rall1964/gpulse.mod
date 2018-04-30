
COMMENT

gpulse.mod
conductance with square onset and offset defined by
        i = g * (v - e)      i(nanoamps), g(micromhos);
        where
         g = 0 for t < onset and
         g=gmax
          for t > onset and < (onset + dur)
          
ENDCOMMENT
                                               
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
        POINT_PROCESS gpulse
        RANGE onset, dur, gmax, e, i
        NONSPECIFIC_CURRENT i
}
UNITS {
        (nA) = (nanoamp)
        (mV) = (millivolt)
        (umho) = (micromho)
}

PARAMETER {
        onset = 0  (ms)
        dur	  = 1  (ms)
        gmax = 0   (umho)
        e = 0      (mV)
        v          (mV)
}

ASSIGNED { i (nA)  g (umho) }

UNITSOFF

BREAKPOINT {
        g = cond(t)
        i = g*(v - e)
}


FUNCTION cond(x) {
        if (x < onset) {
                cond = 0
        } else{
                cond = gmax
        }
        if (x >= (onset + dur)) {
        		cond = 0
        }
}

UNITSON
