/*****
sc.h - create single path classifiers from feature vectors of examples,
as well as classifying example feature vectors.
*****/

#define MAXSCLASSES 100 /* maximum number of classes */

typedef	struct sclassifier	*sClassifier;	/* classifier */
typedef	int sClassIndex;    /* per-class index*/
typedef struct sclassdope *sClassDope;  /* per class information */

struct sclassdope{/* per gesture class information within a classifier */
    char	    *name;	    /* name of a class */
    sClassIndex number;	    /* unique index (small integer) of a class */
    int	        nexamples;	/* number of training examples  */
    Vector	    average;	/* average of training examples  */
    Matrix	    sumcov;	    /* covariance matrix of examples */
};

struct sclassifier{/* a classifier */
    int         nfeatures;	    /* number of features in feature vector */
    int	        nclasses;	    /* number of classes known by this classifier  */
    sClassDope	*classdope;     /* array of pointers to per class data */

    Vector      cnst;           /* constant term of discrimination function  */
    Vector	    *w;	            /* array of coefficient weights */
    Matrix	    invavgcov;	    /* inverse covariance matrix */
};

sClassifier sNewClassifier();	/* */
sClassifier sRead();	        /* FILE f */
void	sWrite();	            /* FILE f; sClassifier sc; */
void	sFreeClassifier();	    /* sc */
void	sAddExample();	        /* sc, char classname; Vector y */
void	sDoneAdding();	        /* sc */
sClassDope sClassify();	        /* sc, y */
sClassDope sClassifyAD();	    /* sc, y, double ap; double dp */
sClassDope sClassNameLookup();	/* sc, classname */
double	MahalanobisDistance();  /* Vector v, u; Matrix sigma */
