

def SummedAreaTable(vals):
    sat = np.zeros(vals.shape)
    sat[0:1,:] = np.cumsum(vals[0:1,:], axis=1)
    sat[:,0:1] = np.cumsum(vals[:,0:1], axis=0)
    for i in range(1, vals.shape[0]):
        for j in range(1, vals.shape[1]):
            sat[i,j] = vals[i,j] + sat[i-1,j] + sat[i,j-1] - sat[i-1,j-1]
    return sat    




def TPI_sat(elevs, w):
    
    # Compute the Summed Area Table (SAT) for the heights
    sat = SummedAreaTable(elevs)
        
    # sum of heights inside w x w window
    r = w//2
    imin = np.maximum(0, np.arange(elevs.shape[0]) - r)
    jmin = np.maximum(0, np.arange(elevs.shape[1]) - r)
    imax = np.concatenate([np.arange(r, elevs.shape[0]), np.full((r,), elevs.shape[0]-1)])
    jmax = np.concatenate([np.arange(r, elevs.shape[1]), np.full((r,), elevs.shape[1]-1)])
    iimin,jjmin = np.meshgrid(imin,jmin, indexing='ij')
    iimax,jjmax = np.meshgrid(imax,jmax, indexing='ij')

    Hsums = sat[iimax,jjmax]
    Hsums[r+1:,:]    -= (sat[iimin-1,jjmax])[r+1:,:]
    Hsums[:,r+1:]    -= (sat[iimax,jjmin-1])[:,r+1:]
    Hsums[r+1:,r+1:] += (sat[iimin-1,jjmin-1])[r+1:,r+1:]    
    Hcnts = (iimax - iimin + 1)*(jjmax - jjmin + 1)
        
    # Compute the TPI values for each cell
    tpi = elevs - Hsums/Hcnts
    
    return tpi

    
    
def LocalVariance_sat(elevs, w):
    
    # summed area tables
    s1Sat = SummedAreaTable(elevs)
    s2Sat = SummedAreaTable(elevs*elevs)
    
    # sums
    r = w//2
    imin = np.maximum(0, np.arange(elevs.shape[0]) - r)
    jmin = np.maximum(0, np.arange(elevs.shape[1]) - r)
    imax = np.concatenate([np.arange(r, elevs.shape[0]), np.full((r,), elevs.shape[0]-1)])
    jmax = np.concatenate([np.arange(r, elevs.shape[1]), np.full((r,), elevs.shape[1]-1)])
    iimin,jjmin = np.meshgrid(imin,jmin, indexing='ij')
    iimax,jjmax = np.meshgrid(imax,jmax, indexing='ij')

    s = s1Sat[iimax,jjmax]
    s[r+1:,:]    -= (s1Sat[iimin-1,jjmax])[r+1:,:]
    s[:,r+1:]    -= (s1Sat[iimax,jjmin-1])[:,r+1:]
    s[r+1:,r+1:] += (s1Sat[iimin-1,jjmin-1])[r+1:,r+1:]    
    
    ss = s2Sat[iimax,jjmax]
    ss[r+1:,:]    -= (s2Sat[iimin-1,jjmax])[r+1:,:]
    ss[:,r+1:]    -= (s2Sat[iimax,jjmin-1])[:,r+1:]
    ss[r+1:,r+1:] += (s2Sat[iimin-1,jjmin-1])[r+1:,r+1:] 
    
    n = w*w
    var = (n*ss - s*s)/(n*(n-1))
    
    return np.sqrt(var)