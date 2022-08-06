/*****
fv.c	Creates a feature vector, useful for gesture classification,
from a sequence of points (e.g. mouse points).
****/

#include <stdio.h>
#include <math.h>
#include "matrix.h" /* contains Vector and associated functions */
#include "fv.h"

/* runtime settable parameters */
double dist_sq_threshold = DIST_SQ_THRESHOLD;
double se_th_rolloff = SE_TH_ROLLOFF;

#define EPS (1.0e-4)

/* allocate an FV struct including feature vector */
FV FvAlloc()
{
    register FV fv = (FV) mallocOrDie(sizeof(struct fv));
    fv->y = NewVector(NFEATURES);
    FvInit(fv);
    return fv;
}

/* free memory associated with an FV struct */
void
FvFree(fv)
FV fv;
{
    FreeVector(fv->y);
    free((char *) fv);
}

/* initialize an FV struct to prepare for incoming gesture points */
void
FvInit(fv)
register FV fv;
{
    register int i;

    fv->npoints = 0;
    fv->initial_sin = fv->initial_cos = 0.0;
    fv->maxv = 0;
    fv->path_r = 0;
    fv->path_th = 0;
    fv->abs_th = 0;
    fv->sharpness = 0;
    fv->maxv = 0;
    for(i = 0; i < NFEATURES; i++)
        fv->y[i] = 0.0;
}

/* update an FV struct to reflect a new input point */
void
FvAddPoint(fv, x, y, t)
register FV fv; int x, y; long t;
{
    double dx1, dy1, magsq1;

    double th, absth, d;
#ifdef PF_MAXV
    long lasttime;
#endif

    ++fv >npoints;
    if(fv->npoints == 1) {/* first point, initialize some vars  */
        fv->starttime = fv->endtime = t;
        fv->startx = fv->endx = fv->minx = fv->maxx = x;
        fv->starty = fv->endy = fv->miny = fv->maxy = y;
        fv->endx = x;
        fv->endy = y;
        return;
    }
    dx1 = x-fv->endx;
    dy1 = y-fv->endy;
    magsq1 = dx1*dx1 + dy1*dy1;

    if(magsq1 <= dist_sq_threshold){
        fv->npoints--;
        return;	/* ignore a point close to the last point */
    }
    if(x < fv->minx) fv->minx = x;
    if(x > fv->maxx) fv->maxx = x;
    if(y < fv->miny) fv->miny = y;
    if(y > fv->maxy) fv->maxy = y;
#ifdef PF_MAXV
    lasttime = fv->endtime;
#endif
    fv->endtime = t;

    d = sqrt(magsq1);
    fv->path_r += d;	/* update path length feature */

    /* calculate initial theta when the third point is seen */
    if(fv->npoints == 3){
        double magsq, dx, dy, recip;
        dx = x-fv->startx; dy = y-fv->starty;
        magsq = dx*dx + dy*dy;
        if(magsq > dist_sq_threshold){
            /* find angle w.r.t. positive x axis e.g. (1,0) */
            recip = 1 / sqrt(magsq);
            fv->initial_cos = dx*recip;
            fv->initial_sin = dy*recip;
        }
    }
    if(fv->npoints >= 3){
        /* update angle based features */
        th = absth = atan2(dx1*fv->dy2-fv->dx2*dy1, dx1*fv->dx2 + dy1*fv->dy2);
        if(absth < 0) absth = -absth;
        fv->path_th += th;
        fv->abs_th += absth;
        fv->sharpness += th*th;

#ifdef PF_MAXV /* compute max velocity */
        if(fv->endtime > lasttime &&  (v = d / (fv->endtime-lasttime)) > fv->maxv)
            fv->maxv = v;
#endif
    }
    /* prepare for next iteration */
    fv->endx = x; fv->endy = y;
    fv->dx2 = dx1; fv->dy2 = dy1;
    fv->magsq2 = magsq1;

    return;
}

/* calculate and return a feature vector */
Vector
FvCalc(fv)
register FV fv;
{
    double bblen, selen, factor;
    if(fv->npoints <= 1)
        return fv->y;	/* a feature vector of all zeros */
    fv->y[PF_INIT_COS] = fv->initial_cos;
    fv->y[PF_INIT_SIN] = fv->initial_sin;
    /* compute the length of the bounding box diagonal */
    bblen = hypot(fv->maxx-fv->minx, fv->maxy-fv->miny);

    fv->y[PF_BB_LEN] = bblen;

    /* the bounding box angle defaults to 0 for small gestures */
    if(bblen*bblen > dist_sq_threshold)
        fv->y[PF_BB_TH] = atan2(fv->maxy-fv->miny,
                                fv->maxx-fv->minx);

    /* compute the length and angle between the first and last points */
    selen = hypot(fv->endx-fv->startx,
                  fv->endy-fv->starty);
    fv->y[PF_SE_LEN] = selen;

    /* when the first and last points are very close,
    the angle features are muted so that they satisfy the stability criterion */
    factor = selen*selen / se_th_rolloff;
    if(factor > 1.0) factor = 1.0;
    factor = selen > EPS ? factor/selen : 0.0;
    fv->y[PF_SE_COS] = (fv->endx-fv->startx)*factor;
    fv->y[PF_SE_SIN] = (fv->endy-fv->starty)*factor;

    /* the remaining features have already been computed */
    fv->y[PF_LEN] = fv->path_r;
    fv->y[PF_TH] = fv->path_th;
    fv->y[PF_ATH] = fv->abs_th;
    fv->y[PF_SQTH] = fv->sharpness;

#ifdef PF_DUR
    fv->y[PF_DUR] = (fv->endtime-fv->starttime)*.01;
#endif

#ifdef PF_MAXV
    fv->y[PF_MAXV] = fv->maxv*10000;
#endif

    return fv->y;
}



