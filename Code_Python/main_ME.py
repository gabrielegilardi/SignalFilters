import numpy as np
import scipy.stats as st

class MEBOOT:
	def __init__(self,x,trimval=0.1,seed=None):
		'''
		x: multivariate-time series N x T
		trimval: trim value (default 0.1)
		'''

		self.sd = np.random.RandomState(seed)
		m,n = x.shape

		self.meanx = x.mean(axis=1)

		self.sdx =  x.std(axis=1)


		self.ordxx = np.argsort(x,axis=1)

		xx = x.ravel()[self.ordxx.ravel()].reshape(x.shape)

		self.z = 0.5*(xx[:,1:]+xx[:,:-1])

		dv = abs(np.diff(x,axis=1))

		dvtrim = st.trim_mean(dv,trimval,axis=1)

		self.xmin = xx[:, 0]- dvtrim
		self.xmax = xx[:,-1]+ dvtrim

		tmp = np.array([[0.25]*(n-2)+[0.5]*(n-2)+[0.25]*(n-2)])
		cmd = (np.column_stack((xx[:,:n-2],xx[:,1:n-1],xx[:,2:n])) * tmp)

		aux = np.array([cmd[:,i::n-2].sum(axis=1) for i in range(n-2)]).T

		self.desintxb = np.column_stack((0.75 * xx[:,:1] + 0.25 * xx[:,1:2], aux,   0.25 * xx[:,-2:-1] + 0.75 * xx[:,-1:]))


	def _mrapproxpy(self,p,z,desintxb):

		m,n = p.shape

		q = -np.inf*np.ones((n)*m)

		a = (p//(1/n)-1).astype(int)

		hs = np.arange(n-2)

		dz = np.column_stack(([-np.inf]*m,np.diff(z,axis=1)*n,[0]*m)).ravel()

		sz = np.column_stack(([0]*m,(0.5*(z[:,hs+1]+z[:,hs]))[:,hs],[0]*m)).ravel()

		zt = np.column_stack(([-np.inf]*m,z[:,hs],[-np.inf]*m)).ravel()

		dh = np.column_stack(([-np.inf]*m,desintxb[:,hs],[0]*m)).ravel()

		plus = (n*np.arange(m))[np.newaxis].T
		jx = (np.tile(range(n),(m,1))+plus).ravel()

		ixo = a+1
		ix = (ixo+plus).ravel()

		tmp = zt[ix]+dh[ix]- sz[ix]

		q[jx] = dz[ix]*(p.ravel()[jx]-(ixo.ravel())/n)+tmp

		return q.reshape((m,n))


	def _expandSD(self,bt,fiv):

		obt = len(bt.shape)
		
		if obt==2:
			bt = bt[np.newaxis]
		
			   
		sd = self.sdx

		bt = np.swapaxes(bt,0,1)

		sdf = np.column_stack((sd,bt.std(axis=2)))

		sdfa = sdf/sdf[:,:1]

		sdfd = sdf[:,:1]/sdf

		mx = 1+(fiv/100)

		idx = np.where(sdfa<1)

		sdfa[idx] = np.random.uniform(1,mx,size=len(idx[0]))

		sdfdXsdfa = sdfd[:,1:]*sdfa[:,1:]

		bt *= np.moveaxis(sdfdXsdfa[np.newaxis],0,-1)
		bt = np.swapaxes(bt,0,1)
		
		if obt==2:
			bt = bt[0]
			
		return bt

	def _adjust(self,bt):
		zz = np.column_stack((self.xmin[np.newaxis].T,self.z,self.xmax[np.newaxis].T))

		v = np.diff(zz**2,axis=1)/12

		xb = self.meanx[np.newaxis].T

		s1 = ((self.desintxb - xb)**2).sum(axis=1)

		act_sd = np.sqrt( (s1+v.sum(axis=1))/(self.z.shape[1]+1) ) 

		des_sd = self.sdx

		kappa =( des_sd/ act_sd -1)[np.newaxis].T

		bt = bt + kappa* (bt - xb)
		return bt

	def bootstrap(self,fiv=5,adjust_sd=True):
		'''
		Single realization of ME Bootstrap for the multivariate time series.

		fiv: Increment standard deviation (default fiv=5 %)
		adjust_sd: Fix the standard deviation from the observation. 
		'''

		m,n = self.z.shape
		n+=1

		p = self.sd.uniform(0,1,size=(m,n))

		q = self._mrapproxpy(p,self.z,self.desintxb[:,1:])


		f_low =  np.column_stack((self.xmin[np.newaxis].T,self.z[:,0]))
		f_hi =  np.column_stack((self.z[:,-1],self.xmax[np.newaxis].T))

		low = p<1/n
		hi = p>(n-1)/n


		for i in range(m): 
			q[i][low[i]] = np.interp(p[i][low[i]],[0,1/n],f_low[i])  
			q[i][hi[i]] = np.interp(p[i][hi[i]],[(n - 1)/n,1],f_hi[i])  

			qseq = np.sort(q[i])
			q[i][self.ordxx[i]] = qseq
			
		if fiv!=None:
			q = self._expandSD(q,fiv)
		if adjust_sd==True:
			q = self._adjust(q)
		return q

	def bootstrap_clt(self,nt,fiv=5,adjust_sd=True):
		'''
		Multiple ME boostrap copies.
		Force the central limit theorem. Warning it requires to compute all
		 bootstrap copies at once, so it could require a lot of memory.
		 
		 nt: number of bootstrap copies
		 fiv: Increment standard deviation (default fiv=5 %)
		 adjust_sd: Fix the standard deviation from the observation. 
		'''


		bt = np.array([self.bootstrap(fiv=None) for i in range(nt)])
		if fiv!=None:
			
			bt = self._expandSD(bt,fiv)

		bt = np.swapaxes(bt,0,1)

		N,nt,T = bt.shape

		gm = self.meanx
		s = self.sdx

		smean = s/ np.sqrt(nt)

		xbar = bt.mean(axis=2)

		sortxbar = np.sort(xbar,axis=1)

		oo = np.argsort(xbar,axis=1)

		newbar = gm[np.newaxis].T + st.norm.ppf((np.arange(1,nt+1)/(nt+1))[np.newaxis])* smean[np.newaxis].T

		scn = st.zscore(newbar,axis=1)

		newm = scn*smean[np.newaxis].T+gm[np.newaxis].T

		meanfix = newm- sortxbar

		oinv = np.array([np.array(sorted(zip(oo[i],range(len(oo[i])))))[:,1] for i in range(len(oo))])

		out = np.array([(bt[i][oo[i]]+meanfix[i][np.newaxis].T)[oinv[i]] for i in range(bt.shape[0])])
		out = np.swapaxes(out,0,1)

		if adjust_sd==True:
			out = self._adjust(out)

		return out
