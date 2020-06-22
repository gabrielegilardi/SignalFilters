
// q <- .C("mrapprox", p=as.double(p), n=as.integer(n), z=as.double(z),
//      desintxb=as.double(desintxb[-1]), ref23=double(n), qq=double(1), 
//      q=double(n), PACKAGE="meboot")$q

void mrapprox(double *p, int *n, double *z, double *desintxb, double *ref23,
              double *qq, double *q)
{
    int i, j, ii1, k;
    double i1, nn;

    nn = *n;
    for(i=0; i < *n; i=i+1){
        q[i] = -99999;
        ref23[i] = -99999;
    }

    j = 0;
    i1 = 1.0;
    for(ii1=0; ii1 < *n-2; ii1=ii1+1){

        j = 0;
        for(i=0; i < *n; i=i+1){

            if( p[i] > i1/nn && p[i] <= (i1+1)/nn ){
                ref23[j] = i;
                j = j+1;
            }

        }

        for(i=0; i < j; i=i+1){

            k = ref23[i];
            qq[0] = z[ii1] + ( (z[ii1+1]- z[ii1]) / ((i1+1)/nn - i1/nn) ) * (p[k] - i1/nn);
            q[k] = qq[0] + desintxb[ii1] - 0.5*(z[ii1] + z[ii1+1]);
        }
        i1 = i1+1;
    }
}
