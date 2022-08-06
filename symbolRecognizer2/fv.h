/*****
fv.h	Create a feature vector, useful for gesture classification,
from a sequence of points (e.g. mouse points).
*****/

/*	compile time settable parameters	*/
/* some of these can also be set at runtime, see fv.c */


#undef USE_TIME


/* Define USE_TIME to enable the duration and maximum  */
/* velocity features. When not defined, 0 may be passed */
/* as the time to FvAddPoint. */


#define DIST_SQ_THRESHOLD (3*3)
/* points within sqrt(DIST_SQ_THRESHOLD) */
/* will be ignored to eliminate mouse jitter */

#define SE_TH_ROLLOFF	(4*4)
/* The SE_THETA features (cos and sin of */
/* angle between first and last point) will */
/* be scaled down if the distance between the */
/* points is less than sqrt(SE_TH_ROLLOFF)  */

/*	Interface	*/
typedef struct fv FV;
  /* During gesture collection, an FV holds */
  /* all intermediate results used in the */
  /* calculation of a single feature vector */

FV	FvAlloc();	/* */
void	FvFree();	/* Fv fv */
void	FvInit();	/* FV fv */
void	FvAddPoint();	/* FV fv; int x, y; long time; */
Vector	FvCalc();	/* FV fv;	*/

/*	internal data structure	*/
#define MAXFEATURES 32
  /* maximum number of features, occasionally useful as an array bound  */

/* indices into the feature Vector returned by FvCalc */

#define PF_INIT_COS	0 /* initial angle (cos) */
#define PF_INIT_SIN	1 /* initial angle (sin)  */
#define PF_BB_LEN	2 /* length of bounding box diagonal */
#define PF_BB_TH	3 /* angle of bounding box diagonal  */
#define PF_SE_LEN	4 /* length between start and end points */
#define PF_SE_COS	5 /* cos of angle between start and end points */
#define PF_SE_SIN	6 /* sin of angle between start and end points */
#define PF_LEN	7 /* arc length of path */
#define PF_TH	8 /* total angle traversed */
#define PF_ATH		9 /* sum of abs vals of angles traversed */
#define PF_SQTH	10 /* sum of squares of angles traversed */

#ifndef USE_TIME
#	define NFEATURES	11
#else
#   define PF_DUR     11  /* duration of path */
#   define PF_MAXV    12  /* maximum speed */
#   define NFEATURES 13
#endif

/* structure which holds intermediate results during feature vector calculation */
struct fv{
    /* the following are used in calculating the features */
    double	startx, starty; /* starting point */
    long	starttime;	/* starting time */

    /* these are set after a few points and then left alone */

    double	initial_sin, initial_cos; /* initial angle to x axis */

    /* these are updated incrementally upon every point */
    int	npoints;	/* number of points in path */

    double	dx2, dy2;	/* differences: endx prevx, endy prevy */
    double	magsq2;	/* dx2 dx2 + dy2 dy2 */

    double	endx, endy; /* last point added */
    long	endtime;

    double	minx, maxx, miny, maxy; /* bounding box */

    double	path_r, path_th; /* total length and rotation (in radians) */
    double	abs_th;	/* sum of absolute values of path angles */
    double	sharpness;	/* sum of squares of path angles */
    double	maxv;	/* maximum velocity */

    Vector	y;	/* Actual feature vector */
};

