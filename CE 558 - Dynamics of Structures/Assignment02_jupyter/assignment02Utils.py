import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from math import floor, log, ceil

def psdf(x,ts):
    window ='hann'
    nperseg = floor(x.shape[0]/1)
    nfft = 2**(ceil(log(nperseg,2)))
    ts = ts
    fs = 1/ts
    freqs, psd = sig.welch(x, fs=fs, nperseg=nperseg, window=window, nfft=nfft)
    psd = np.convolve(psd, np.ones(20)/20, mode='valid')
    plt.figure()
    plt.semilogy(freqs[9:-10],psd)
    plt.xlabel('Frequency Hz')
    plt.ylabel('Amplitdue')
    return freqs, psd

def nmlize(arr):
    arr = np.array(arr, dtype=float)
    mn = np.min(arr)
    mx = np.max(arr)
    if np.isclose(mx, mn):
        return arr * 0.0
    return (arr - mn)/(mx - mn)

def fdd(system_dict):
    
#     %   input arguments
# %    in.x = accel;
# %    in.DF = 3; %Hz. Frequency interval in which mean values is evaluated.
# %    in.Nfft = 2048;
# %    in.fs = fs;
# %    in.fc = fc;
# %    in.isShowingFigures = 1;
# %    in.Npeaks = 5;
# %    in.curv_cut = 0.3;
# %    in.hw = 10; % half width of sdof
    
    x = system_dict['x']
    DF = 3.0
    Nfft = 2048
    fs = system_dict['fs']
    fc = system_dict['fc']
    isShowingFigures = 1
    Npeaks = 5
    curv_cut = 0.3
    hw = 10.0 # half width of SDOF
    ppm = system_dict.get('ppm', 'manual')
    
    # Cross spectrum -----------------
    Nacc = x.shape[1]
    Gyy = np.zeros((Nacc, Nacc, Nfft//2+1), dtype=complex)
    
    for i in range(Nacc):
        for j in range(Nacc):
            fvals, Pxy = sig.csd(x[:, i], x[:, j], 
                                   fs=fs, window='hann',
                                   nperseg=Nfft, noverlap=Nfft//2,
                                   nfft=Nfft)
            Gyy[i, j, :] = Pxy
    
    # SVD -----------------
    nfreq = Gyy.shape[2]
    sw = np.zeros(nfreq)
    for idx_f in range(nfreq):
        # Gyy at this frequency is an (Nacc x Nacc) matrix
        Gmat = Gyy[:, :, idx_f]
        # SVD
        #   U, S, Vh = np.linalg.svd(Gmat)  # Python returns V^H, not V
        #   Largest singular value = S[0]
        # but for cross-spectra, it's typically Hermitian; we just want S[0].
        S = np.linalg.svd(Gmat, compute_uv=False)
        sw[idx_f] = S[0]

    # Plot singular values vs frequency if desired
    freq_axis = fvals  # from csd call, shape (Nfft//2+1,)
    if isShowingFigures:
        plt.figure()
        plt.semilogy(freq_axis, sw)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Singular Values')
        plt.title('Singular Values of Gyy(jω)')
        plt.grid(True)
        plt.show()

    # --- Peak picking ---------------------------------------------------------
    df = fs / Nfft  # frequency resolution

    if ppm == 'manual':
        
        nextPeak = True
        ithpeak = 0
        ifn = np.zeros(Npeaks, dtype=int)

        # Plot sw vs index (to match the original code’s “index” approach)
        # Or you could plot sw vs freq_axis. We'll do index-based for clarity:
        fig = plt.figure()
        plt.semilogy(sw)
        plt.xlabel('index')
        plt.ylabel('Singular Values')
        plt.title('Select peak regions [Left-Click two X-limits, or Right-Click in reverse to finish]')
        plt.grid(True)
        plt.show(block=False)  # don't block execution, but keep plot open

        while nextPeak:
            print("Pick two points (left then right) to define a peak region...")
            # Wait for user to click two points on the open figure
            rec = plt.ginput(2, timeout=-1)  # returns list of (x, y) pairs

            if len(rec) < 2:
                print("No or insufficient points selected. Finishing peak picking.")
                break

            # rec is something like [(x1, y1), (x2, y2)]
            x1 = rec[0][0]
            x2 = rec[1][0]
            # Convert to integer indices
            idx_f1 = int(np.floor(x1))
            idx_f2 = int(np.floor(x2))

            # Check if user reversed the order to finish
            if x1 > x2:
                print("Peak picking complete.")
                break

            if ithpeak >= Npeaks:
                print("Max number of peaks reached.")
                break

            # Validate range
            if idx_f1 >= idx_f2 or idx_f1 < 0 or idx_f2 >= len(sw):
                print("Invalid region. Please select again.")
                continue

            # Find local max in sw[idx_f1 : idx_f2+1]
            segment = sw[idx_f1:idx_f2+1]
            pv = np.max(segment)
            imax = np.argmax(segment)
            # Global index:
            peak_index = idx_f1 + imax
            ifn[ithpeak] = peak_index
            ithpeak += 1

        # Keep only the ones we used
        ifn = ifn[0:ithpeak]
    else:
        # Automatic approach based on curvature measure
        Nf = int(np.floor(DF / df))  # number of bins per smoothing block

        # 1. Mean of each block
        #    We'll take blocks of length Nf from sw, compute average in each block
        #    so "mean_sw" will have length ~ len(sw)/Nf
        num_blocks = len(sw) // Nf
        mean_sw = np.zeros(num_blocks)
        for i_block in range(num_blocks):
            block_slice = slice(i_block*Nf, i_block*Nf + Nf)
            mean_sw[i_block] = np.mean(sw[block_slice])

        # 2. curvature of the mean (log scale)
        #    curv_sw(i) = log10(m_{i+2}) - 2*log10(m_{i+1}) + log10(m_{i})
        curv_sw = []
        for i_block in range(num_blocks - 2):
            val = (np.log10(mean_sw[i_block+2]) 
                   - 2*np.log10(mean_sw[i_block+1]) 
                   + np.log10(mean_sw[i_block]))
            curv_sw.append(val)
        curv_sw = np.array(curv_sw)

        # remove outliers
        mask = (curv_sw > -1e10) & (curv_sw < 1e10)
        curv_sw = curv_sw[mask]
        # normalize
        curv_sw = nmlize(curv_sw, 0)

        if isShowingFigures:
            plt.figure()
            plt.subplot(211)
            plt.semilogy(mean_sw, 'o-')
            plt.title('Mean of SW (blocks)')
            plt.subplot(212)
            plt.plot(curv_sw, 'o-')
            plt.title('Curvature of log10(mean_sw)')
            plt.tight_layout()
            plt.show()

        # pick peaks from curvature
        idx_peak = np.where(np.abs(curv_sw) > curv_cut)[0]
        # idx_peak is in [0 ... len(curv_sw)-1], which corresponds to block indices
        # but shifted by 1 from original code?
        # Original code does: idx_peak + 1 for "index in mean of sv"
        idx_peak = idx_peak + 1  # so it lines up with mean_sw index
        if len(idx_peak) == 0:
            ifn = []
        else:
            diff_idx = np.diff(idx_peak)
            idx_freq = np.zeros((Npeaks, 2), dtype=int)
            npeak_found = 0

            for i in range(len(diff_idx)):
                if i == 0:
                    # first region start
                    idx_freq[npeak_found, 0] = idx_peak[0]
                else:
                    idx_freq[npeak_found, 1] = idx_peak[i]
                    if diff_idx[i] < 3:
                        # continue region
                        pass
                    else:
                        # new region
                        npeak_found += 1
                        if npeak_found < Npeaks:
                            idx_freq[npeak_found, 0] = idx_peak[i+1]

            # Now, gather the peak indices from the blocks
            ifn_list = []
            for i_row in range(Npeaks):
                if idx_freq[i_row, 0] == 0:
                    break
                i_left_block  = (idx_freq[i_row, 0] - 1)*Nf
                if idx_freq[i_row, 1] == 0:
                    i_right_block = i_left_block + 3*Nf - 1
                else:
                    i_right_block = (idx_freq[i_row, 1] + 1)*Nf

                # clip to array range
                i_left_block  = max(i_left_block, 0)
                i_right_block = min(i_right_block, len(sw)-1)

                block_vals = sw[i_left_block : i_right_block+1]
                if len(block_vals) == 0:
                    continue
                idx_localmax = np.argmax(block_vals)
                peak_idx = i_left_block + idx_localmax
                ifn_list.append(peak_idx)

            ifn = [idx for idx in ifn_list if idx >= 0]
    # end peak-picking

    ifn = np.array(ifn, dtype=int)
    ifn = ifn[ifn > 0]  # remove any leftover zeros
    freq = freq_axis  # same as "0:df:(Nfft/2)*df" in MATLAB

    # The final frequencies from simple “peak index”
    fd_fdd2 = freq[ifn]

    # SDOF curve-fitting to get (f, z)
    fd_fdd, z_fdd = sdof(sw, ifn, fs, hw)

    if isShowingFigures:
        plt.figure()
        plt.semilogy(freq, sw, label='SW')
        plt.semilogy(fd_fdd2, sw[ifn], 'ro', label='Peaks')
        plt.xlim([0, fc])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Singular values of S_{yy}(jω)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # get mode shapes
    # original code: phi and phi_pp from interpolation of SVD
    phi = np.zeros((Nacc, len(ifn)), dtype=complex)
    phi_pp = np.zeros((Nacc, len(ifn)), dtype=complex)

    for i_peak in range(len(ifn)):
        idx_p1 = ifn[i_peak]
        idx_p2 = min(idx_p1 + 1, nfreq-1)

        # two SVDs: at freq idx_p1 and idx_p2
        U1, S1, V1h = np.linalg.svd(Gyy[:, :, idx_p1])
        U2, S2, V2h = np.linalg.svd(Gyy[:, :, idx_p2])

        p1 = U1[:, 0]
        p2 = U2[:, 0]

        # interpolation: phi(:,i) = ...
        # (fd_fdd(i)-freq(ifn(i)))/(freq(ifn(i)+1)-freq(ifn(i))) * (p2 - p1) + p1
        # but we used fd_fdd(i) which might not be exactly freq(idx_p1).
        f1 = freq[idx_p1]
        f2 = freq[idx_p2]
        if np.isclose(f2, f1):
            alpha = 0.0
        else:
            alpha = (fd_fdd[i_peak] - f1)/(f2 - f1)

        phi[:, i_peak] = alpha*(p2 - p1) + p1
        # direct peak picking
        phi_pp[:, i_peak] = p1  # or the leading singular vector at the peak

    # filter modes under fc
    mask_fc = fd_fdd < fc
    fd_fdd  = fd_fdd[mask_fc]
    z_fdd   = z_fdd[mask_fc]
    phi     = phi[:, mask_fc]

    # build output dictionary
    out = {
        'phi': phi,
        'phi_pp': phi_pp,
        'fd': fd_fdd,
        'fd_pp': fd_fdd2,   # not filtered for fc in original code? or you might filter
        'z': z_fdd,
    }

    return out

def sdof(p, idx_list, fs, hw):
    
    p = np.asarray(p).flatten()
    n_peaks = len(idx_list)
    f_out = np.zeros(n_peaks)
    z_out = np.zeros(n_peaks)
    
    df = fs / (len(p) - 1) / 2.0

    for j in range(n_peaks):
        center = idx_list[j]
        
        start_i = int(max(center - hw, 0))
        end_i   = int(min(center + hw, len(p)-1))
        pj = np.zeros_like(p)
        pj[start_i:end_i+1] = p[start_i:end_i+1]

        pj2 = np.concatenate([pj, pj[-2:0:-1]])  # flip ignoring first & last
        # Now the IFFT
        x = np.fft.ifft(pj2)

        # Keep only first ~1/3 for analyzing in fz
        x_sub = x[:len(x)//3]
        f_est, z_est = fz(np.real(x_sub), fs)
        f_pp = df * (center - 1)  # from code: fpp = df*(i(j)-1);
        if abs(f_pp - f_est) > df:
            f_est = f_pp

        f_out[j] = f_est
        z_out[j] = z_est

    return f_out, z_out

def fz(x, fs):
    
    thrshld = 0.1 # 10 percent
    n = x.shape[0]
    time = np.arange(n) / fs
    nzc = 0  # number of zero crossings
    izc = np.zeros(n-1, dtype=int)
    t0 = np.zeros(n-1)
    
    for i in range(1, n):
        if x[i-1] > 0 and x[i] < 0:
            izc[nzc] = i
            nzc += 1
            xa = x[i-1]
            xb = -x[i]
            ta = time[i-1]
            tb = time[i]
            t0[nzc-1] = (ta * xb + tb * xa) / (xa + xb)
        if nzc > 2:
            max_0 = np.max(np.abs(x[izc[0]:izc[1]]))
            max_x = np.max(np.abs(x[izc[nzc-2]:izc[nzc-1]]))
            if (max_x / max_0) < thrshld:
                break
    
    if nzc > 1:
        f = (nzc-1) / (t0[nzc-1] - t0[0])
        max_1 = - np.min(x[izc[0]:izc[1]]) if len(x[izc[0]:izc[1]])>0 else 1
        max_2 = - np.min(x[izc[nzc-2]:izc[nzc-1]]) if len(x[izc[nzc-2]:izc[nzc-1]])>0 else 1
        if max_2 <= 0 or max_1 <=0:
            z = 0.0
        else:
            z = 1.0/(2.0*np.pi) * np.log(max_1/max_2) / (nzc-1)
    
    else:
        
        f = 0
        z = 0
    
    return f, z