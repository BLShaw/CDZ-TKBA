import numpy as np

def gaussian_kernel(x, w, sigma):
    """
    Calculates Gaussian Kernel.
    MATLAB: exp(-nrm/(2*sig^2))
    """
    # x: (D,) or (1, D), w: (N, D)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    
    # Squared Euclidean distance
    diff = x - w
    norm_sq = np.sum(diff**2, axis=1)
    
    return np.exp(-norm_sq / (2 * sigma**2))

def cim(x, w, sigma):
    """
    Correntropy Induced Metric (CIM).
    Robust distance metric based on Gaussian Kernel.
    """
    # x: (D,), w: (N, D)
    if w.ndim == 1:
        w = w[np.newaxis, :]
        
    D = x.shape[-1]
    
    # Kernel at 0 (Maximum similarity)
    # TKBA implementation implies GaussKernel(0) = 1 for the simplified version, 
    # or 1/(sqrt(2pi)*sig) for the full PDF.
    # We will use the normalized version to ensure range [0, 1] approx.
    # Note that the MATLAB code `CIM` function calculates `sqrt(ret0 - ret1)`.
    # And `GaussKernel` is just exp(-sub^2...). So ret0 = 1.
    ret0 = 1.0
    
    # Kernel of differences (Component-wise)
    # MATLAB: g_Kernel(:,i) = GaussKernel(X(i)-Y(:,i), sig);
    # we compute the kernel for EACH dimension separately, then mean.
    
    diff = x - w # (N, D) broadcasted
    g_kernel = np.exp(-(diff**2) / (2 * sigma**2))
    
    ret1 = np.mean(g_kernel, axis=1) # (N,)
    
    # Result
    # Use absolute value to avoid negative sqrt due to float precision errors
    val = np.abs(ret0 - ret1)
    return np.sqrt(val)

def kernel_bayes_rule(pattern, weight, count_cluster, sigma, kbr_sigma):
    """
    Selects the best cluster using Kernel Bayes Rule.
    Returns posterior probabilities for each cluster.
    """
    # optimized port of the MATLAB KBR logic
    # pattern: (D,)
    # weight: (N, D)
    # count_cluster: (N,)
    
    N = len(weight)
    if N == 0: return []
    
    # Prior Probability (Empirical)
    total_counts = np.sum(count_cluster)
    if total_counts == 0:
        prior = np.ones(N) / N
    else:
        prior = count_cluster / total_counts
        
    # In the full TKBA, KBR involves complex matrix inversions (Gram Matrices).
    # For this integration, implementing the "Likelihood" part primarily via the Kernel,
    # multiplied by Prior, which is the core of Naive Bayes.
    # Note: Full KBR is O(N^3) due to matrix inversion, which is too slow for our 600+ nodes in real-time.
    
    # Approximate Posterior: Likelihood * Prior
    # Likelihood ~ Gaussian Kernel similarity in feature space
    likelihood = gaussian_kernel(pattern, weight, kbr_sigma)
    
    posterior = likelihood * prior
    
    # Normalize
    if np.sum(posterior) > 0:
        posterior /= np.sum(posterior)
        
    return posterior
