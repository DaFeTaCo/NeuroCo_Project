'''
    ----Connectivity Utils----
    
    Utils to be use for static connectivity analysis in rs-fMRI

    Implementation of connectivity measures such as: 
        - Pearson Correlation
        - Partial Correlation
        - Covariance
        - Coherence (median & mean)
        - Coherence Multitaper
        - Dynamic Time Warping (manual and from fastdtw lib)
        - Euclidean Distance
        - Mutual Information
    
    @AlexaLond 
    30/03/2024

'''

# Import of libs

# Files handling
import os
import logging
import warnings
import shutil
from joblib import Memory

# Scientific computation and data analysis
import numpy as np
import pandas as pd
import math
from numpy import savetxt
from scipy.io import loadmat

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch
from nilearn import plotting
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Neuroimaging libs
import nibabel as nib
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.image import load_img
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from sklearn.preprocessing import StandardScaler

brainmask = load_mni152_brain_mask()
mem = Memory('nilearn_cache')

# Stats libs
from scipy.stats import ks_2samp
from scipy.stats import kruskal

# Nitime specific for time series analysis
from nitime import utils
import nitime.algorithms as alg
import nitime.viz
from nitime.viz import drawmatrix_channels

# Signal analysis
from scipy import fftpack, signal
from scipy.spatial import distance
from scipy.stats import entropy
import scipy.spatial.distance as dist
from fastdtw import fastdtw


def corr_nilearn(data_series, labels, kind='correlation', vect=False, discard=False,save_path=None,save=False, group=False): 

    """
    Calculate Pearson correlation matrices, Covariance matrices or Partial Correlation from a numpy array of a parceled BOLD signals [-1, 1]

    Args:
        data_series (array): np array of parceled BOLD time series (each row has size [time series, ROIs]).
        labels (pd.Series): Pandas Series containing labels corresponding to participants code 
        kind (str): three options: correlation, covariance or partial correlation for connectivity measure
        save_matrices (bool): Whether to save correlation matrices as CSV files in the given path

    Returns:
        (array): array of correlation matrices of the given subjects
        (array): array with the mean matrix of the given population

    """
    # standardized_data_list = []

    # # Standardize each subject's data
    # scaler = StandardScaler()
    # for subject_data in data_series:
    #     standardized_data = scaler.fit_transform(subject_data)
    #     standardized_data_list.append(standardized_data)

    # connectome_measure = ConnectivityMeasure(kind=kind, vectorize=vect, discard_diagonal=discard)

    # correlation_matrices = connectome_measure.fit_transform(standardized_data_list)
    # mean_correlation_matrix = connectome_measure.mean_

    # if save:
    #     for i, matrix in enumerate(correlation_matrices):
    #         np.savetxt(f"{save_path}/sub-{labels[i]}_{kind}_matrix.csv", matrix, delimiter=",")
    #     np.savetxt(f"{save_path}/{group}_mean_{kind}_matrix.csv", mean_correlation_matrix, delimiter=",")   

    # return correlation_matrices, mean_correlation_matrix

    connectome_measure = ConnectivityMeasure(kind=kind, vectorize=vect, discard_diagonal=discard)

    # Compute connectivity matrices for each subject
    connectivity_matrices = connectome_measure.fit_transform(data_series)
    
    standardized_matrices_list = []
    
    # Standardize each subject's connectivity matrix
    scaler = StandardScaler()
    for matrix in connectivity_matrices:
        if vect:
            # Vectorized version of the matrix (without diagonal)
            standardized_matrix = scaler.fit_transform(matrix.reshape(-1, 1)).reshape(matrix.shape)
        else:
            # Full matrix version
            standardized_matrix = scaler.fit_transform(matrix)
        standardized_matrices_list.append(standardized_matrix)

    mean_correlation_matrix = np.mean(standardized_matrices_list, axis=0)

    if save:
        for i, matrix in enumerate(standardized_matrices_list):
            np.savetxt(f"{save_path}/sub-{labels[i]}_{kind}_matrix.csv", matrix, delimiter=",")
        np.savetxt(f"{save_path}/{group}_mean_{kind}_matrix.csv", mean_correlation_matrix, delimiter=",")   

    return standardized_matrices_list, mean_correlation_matrix

def coherence_func(time_series_array, fs, nperseg, window_type, overlap): 

    """
        Calculate the median and mean of the coherence similarity measure [0, 1]

    Args: 
        time_series_array(array): parcelled (structural or functional) time series. Shape --> (n, time_series, ROIs)
        fs(int): sample frequency of the coherence [1/TR]
        nperseg(int): samples per seg taken to compute the coherence
        window_type(str): window name between 'hann', 'hamming', 'tukey', etc. [see scipy.signal docs]

    Returns: 
        (array): mean coherence matrices 
        (array): median coherence matrices
    
    """

    num_subjects = len(time_series_array)

    regions = len(time_series_array[0][1])

    coh_mean_all_subjects = []
    coh_med_all_subjects = []


    for subject_idx in range(num_subjects): 

        time_series = time_series_array[subject_idx]
        coh_mean = np.zeros((regions, regions))
        coh_med = np.zeros((regions, regions))

        for i in range(regions):
            for j in range(i):
                # Magnitude squared coherence estimate, of discrete-time signals X and Y 
                # using Welch's method
                f, Cxy = signal.coherence(time_series[:,i], time_series[:,j],
                                          fs=fs, nperseg=nperseg, window=window_type,noverlap=overlap, nfft=None, detrend='constant', axis=-1)

                # Cxy = abs(Pxy)**2/(Pxx*Pyy); where P: power spectral density of X and Y, or xy: cross spectral
                # density

                valid_freq_indices = np.where((f > 0.02) & (f < 0.15)) # BOLD signal range 
                Cxy = Cxy[valid_freq_indices]
            
                mean_Cxy = np.mean(Cxy) #[(f > 0.01) & (f <= 0.1)])#[np.where((f>0.009) & (f<0.09))]) #[np.where((f>0.009) & (f<0.09))]
                median_Cxy = np.median(Cxy) #[np.where((f>0.009) & (f<0.09))]), np.sum(cxy[(f > 4) & (f <= 8)])
                
                coh_mean[i,j] = mean_Cxy
                coh_mean[j,i] = mean_Cxy

                coh_med[i,j] = median_Cxy
                coh_med[j,i] = median_Cxy

        coh_mean_all_subjects.append(coh_mean)
        coh_med_all_subjects.append(coh_med)

    coh_mean_all_subjects = np.array(coh_mean_all_subjects)
    coh_med_all_subjects = np.array(coh_med_all_subjects)

    return coh_mean_all_subjects, coh_med_all_subjects

def coherence_multitaper(global_signals_subjects, TR): 

    """
        Calculate the median and mean of the multitaper coherence similarity measure [0, 1]
        
        ~ Multitaper methods offer advantages in terms of frequency resolution and variance reduction
        by using multiple orthogonal tapers

    Args: 
        global_signals_subjects(array): parcelled (structural or functional) time series. Shape --> (n, time_series, ROIs)
        TR(int): fMRI acquisition repetition time

    Returns: 
        (array): mean multitaper coherence matrices 
        (array): median multitaper coherence matrices
    
    """

    # Normalize the data in each of the ROIs to be in units of % change

    mean_coh_multitaper = []
    med_coh_multitaper = []

    cont = 0

    for signals in global_signals_subjects: 

        print(cont)

        pdata = utils.percent_change(np.transpose(signals))

        NW = 4
        K = 2 * NW - 1
        f_ub = 0.15 # Frequency init
        f_lb = 0.02 # Frequency stop

        n_samples = pdata.shape[1]
        regions = pdata.shape[0]

        print("samples " + str(n_samples),"regions " + str(regions))

        tapers, eigs = alg.dpss_windows(n_samples, NW, K)

        # print('dimensiones tapers: ' + str(tapers.shape))

        tdata = tapers[None, :, :] * pdata[:, None, :]

        print('tdata shape: ', tdata.shape)
        # print('tdta: ', tdata)

        tspectra = fftpack.fft(tdata)

        # print(tspectra.shape)

        L = n_samples // 2 + 1
        sides = 'onesided'

        w = np.empty((regions, K, L))
        for i in range(regions):
            # print('This is regions: ', i)
            # print('This is tspectra: ', tspectra[i])
            
            w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides=sides)
            # print('This is W[I]: ', w[i])

        #csd_mat = np.zeros((nseq, nseq, L), 'D')
        #psd_mat = np.zeros((2, nseq, nseq, L), 'd')
        coh_mat = np.zeros((regions, regions, L), 'd')
        #coh_var = np.zeros_like(coh_mat)

        for i in range(regions):
            for j in range(i):
                #We calculate the multi-tapered cross spectrum between each two time-series:

                sxy = alg.mtm_cross_spectrum(tspectra[i], tspectra[j], (w[i], w[j]), sides='onesided')

                sxx = alg.mtm_cross_spectrum(tspectra[i], tspectra[i], w[i], sides='onesided')
                syy = alg.mtm_cross_spectrum(tspectra[j], tspectra[j], w[j], sides='onesided')

                #psd_mat[0, i, j] = sxx
                #psd_mat[1, i, j] = syy

                coh_mat[i, j] = np.abs(sxy) ** 2
                coh_mat[i, j] /= (sxx * syy)
                coh_mat[j, i] = coh_mat[i, j]
                #csd_mat[i, j] = sxy

        if L < n_samples:
            freqs = np.linspace(0, 1 / (2 * TR), L)
        else:
            freqs = np.linspace(0, 1 / TR, L, endpoint=False)

        # Look only at frequencies between 0.02 and 0.15 (the physiologically relevant band, see http://imaging.mrc-cbu.cam.ac.uk/imaging/DesignEfficiency)
        freq_idx = np.where((freqs > f_lb) * (freqs < f_ub))[0]

        print(coh_mat.shape)

        # Extract the coherence and average over all these frequency bands:
        coh_mean = np.mean(coh_mat[:, :, freq_idx], -1)  # Averaging on the last dimension
        coh_med = np.median(coh_mat[:, :, freq_idx], -1)  # Averaging on the last dimension

        mean_coh_multitaper.append(coh_mean)
        med_coh_multitaper.append(coh_med)

        cont+=1
        
    mean_coh_multitaper = np.array(mean_coh_multitaper)
    med_coh_multitaper = np.array(med_coh_multitaper)

    return mean_coh_multitaper, med_coh_multitaper
        

def eucl_distance(time_series_subjects): 

    """
        Compute the Euclidian Distance as a similiarity measure
    Args: 
        time_series_subjects(array): parcelled (structural or functional) time series. Shape --> (n, time_series, ROIs)

    Returns: 
        (array): Array with Euclidian Distance matrices of the given subjects
    
    """

    scaler = MinMaxScaler() # Normalization

    euclid_mat = []

    for subject in time_series_subjects:

        # Compute pairwise Euclidean distances
        dist_matrix = distance.cdist(np.transpose(subject), np.transpose(subject))

        # Apply scaling to the distance matrix (avoiding re-fitting)
        dist_matrix = scaler.fit_transform(dist_matrix)

        euclid_mat.append(dist_matrix)

    euclid_mat = np.array(euclid_mat)

    return euclid_mat


def mutual_information(subjects_data, num_bins=17): 

    """
        Compute Mutual Information as a similiarity measure for the use of functional connectivity

    Args: 
        subjects_data(array): parcelled (structural or functional) time series. Shape --> (n, time_series, ROIs)
        num_bins(int): number of bins for discretization (Adjust as needed )
    
    Returns: 
        (array): array with matrices based on Mutual Information for the given   

    """

    mutual_info_results = []
    scaler = MinMaxScaler()  # Initialize Min-Max scaler for normalization
    
    for subject_data in subjects_data:

        mutual = []

        for i in range(subject_data.shape[1]):
            for j in range(subject_data.shape[1]):
                hist_1, bin_edges_1 = np.histogram(subject_data[:, i], bins=num_bins)
                hist_2, bin_edges_2 = np.histogram(subject_data[:, j], bins=num_bins)

                joint_histogram, _, _ = np.histogram2d(subject_data[:, i], subject_data[:, j], bins=[bin_edges_1, bin_edges_2])
                joint_entropy = entropy(joint_histogram.flatten(), base=2)
                entropy_1 = entropy(hist_1, base=2)
                entropy_2 = entropy(hist_2, base=2)

                mutual_information = entropy_1 + entropy_2 - joint_entropy
                mutual.append(mutual_information)

        mutual = np.array(mutual).reshape(subject_data.shape[1], subject_data.shape[1])
        normalized_mutual = scaler.fit_transform(mutual)
        mutual_info_results.append(normalized_mutual)

    return np.array(mutual_info_results)



def read_func(dict_subject_folder, verbose=False): 

    """
        Function for read files inside func folder from each subject/each cognitive group

    Args: 
        dict_subject_folder(dict): key: cognitive group path, values: directory path for each subject

    Returns: 
        (dict): key: cognitive group path, values: dswau directory paths for each subject/each cognitive group
    
    """    
    dswau_workpath = {}
    for path, files in dict_subject_folder.items(): 

        for file in files: 

            func_path = os.path.join(path, file+'/func/')

            if os.path.exists(func_path): 

                dswau = [os.path.join(func_path, dswau) for dswau in os.listdir(func_path)]

                dswau_workpath.setdefault(path, []).extend(dswau)
    if verbose: 
        for path, files_func in dswau_workpath.items():
            print(f"files in 'func' folder from path {path}:")
            for file in files_func:
                print(file) 

    return dswau_workpath

def load_matrices(network_types: list, groups: list, corr_workpath: str) -> dict: 

    """
    Reading functional connectivity matrices stored in .csv files and separated by 
    brain parcellation folders and clinical groups

    Args:
        network_types (list): list with atlas names stored in folders, such as 'yeo_7_Nets', 'seitzman' and 'schaefer_4_400ROI'
        groups (list): list with clinical groups under study stored in folders, such as 'AD', 'CNAD', etc.
        corr_workpath (str): folder path where correlation matrices are stored

    Returns:
        dict: functional connectivity matrices, keys =  networks, subkeys = {group}, values= matrices
              example: {'network': {'group': [n, n]}}
    """
    loaded_matrices = {}
    # Iterate over each network type and group
    for network in network_types:
        loaded_matrices[network] = {}
        for group in groups:
            save_path = os.path.join(corr_workpath, f'{network}/{group}/')

            csv_files = [f for f in os.listdir(save_path) if f.startswith('sub') and f.endswith('.csv')]
                
            # Load all matrices in the directory
            matrices = [np.loadtxt(os.path.join(save_path, f), delimiter=',') for f in csv_files]
            
            loaded_matrices[network][group] = matrices

    return loaded_matrices

def upload_csv_matrices(work_path): 

    """
        Load the matrices stored in .csv files

    Args: 
        work_path(str): parent path where the matrices are stored (by group)
        filename_subject(list): list with the names of .csv file of the matrices
    
    Return: 
        (array): .csv matrices files read

    """
    # matrix_array_subs = [np.loadtxt(os.path.join(work_path, file), delimiter=',') for file in filename_subjects]

    all_files = os.listdir(work_path)

    csv_dirs = [os.path.join(work_path,file) for file in all_files if file.endswith('.csv')]

    matrix_array_subjects = [np.loadtxt(file_path, delimiter=',') for file_path in csv_dirs]

    return np.array(matrix_array_subjects)

def plot_distribution_connectivity(site_labels: np.array, loaded_matrices: dict, network: str, path_to_save: str, kruskall: bool = True, kolmo_s: bool =False) -> None:

    """
    Plotting distribution of functional connectivity matrices per clinical group and scanner manufacter

    Args: 
        network (str): network name to be processed, e.g., 'yeo_7_Nets'
        kruskall (bool): True if compute Kruskal-Wallis test
        kolmo (bool): True if compute Kolmogorov-Smirnov test

    Returns: 
        None: distribution connectivity plots using histplot method from seaborn

    """
    site_labels_AD = site_labels[0:64]
    subjects_labels_AD = np.delete(site_labels_AD, 44)

    site_labels_CNAD = site_labels[64:128]
    site_labels_CNMCI = site_labels[128:186]
    site_labels_MCI = site_labels[186::]

    site_labels_dict = {'AD': subjects_labels_AD, 'CNAD': site_labels_CNAD, 'CNMCI': site_labels_CNMCI, 'MCI': site_labels_MCI}
    # Flatten the connectivity matrices and group by group
    flattened_connectivities = {group: {} for group in loaded_matrices[network].keys()}

    for group, matrices in loaded_matrices[network].items():
        for matrix, site in zip(matrices, site_labels_dict[group]):
            if site not in flattened_connectivities[group]:
                flattened_connectivities[group][site] = []
            flattened_connectivities[group][site].extend(matrix[np.triu_indices_from(matrix, k=1)])

    palette = sns.color_palette("viridis", 3)
    print("----Plotting with kdeplot----")

    weights_dict = {group: {} for group in flattened_connectivities.keys()}
    for group, sites in flattened_connectivities.items():
        total_subjects = sum(len(values) for values in sites.values())
        for site, values in sites.items():
            weights_dict[group][site] = np.ones_like(values) / len(values)

    # Plot histograms for each group separately
    for group, sites in flattened_connectivities.items():
        plt.figure(figsize=(12, 6))
        for idx, (site, values) in enumerate(sites.items()):
            weights = weights_dict[group][site]
            # sns.histplot(values, bins=50, kde=True, label=f'Site {site}', alpha=0.5, color=palette[idx])
            sns.kdeplot(values, weights=weights, label=f"{site} KDE", shade=True)
        
        plt.xlabel('Connectivity Values')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Functional Connectivity Values for Group {group} - {network}')
        plt.legend()

        if kruskall: 
            # Perform Kruskal-Wallis test
            values_list = [values for values in sites.values()]
            stat, p_value = kruskal(*values_list)
            
            # Add text if significant
            if p_value < 0.05:
                plt.text(0.5, 0.9, f'Kruskal-Wallis: p={p_value:.3f}', transform=plt.gca().transAxes, fontsize=12, color='red')

        if kolmo_s: 
            site_names = list(sites.keys())
            for i in range(len(site_names)):
                for j in range(i + 1, len(site_names)):
                    site1 = site_names[i]
                    site2 = site_names[j]
                    values1 = sites[site1]
                    values2 = sites[site2]
                    stat, p_value = ks_2samp(values1, values2)
                    if p_value < 0.05:
                        plt.text(0.5, 0.9 - 0.05 * (i + j), f'Site {site1} vs Site {site2}: p={p_value:.3f}', transform=plt.gca().transAxes, fontsize=10, color='red')
        
        fig_name = f'{path_to_save}connectivity_distribution_{group}_{network}.png'
        plt.savefig(fig_name)
        print(f"Figure {fig_name} saved")
        plt.show()


def create_submatrices(correlation_matrix, roi_counts):

    """
        Function to create submatrices for each network (e.g DMN matrix)

    Args: 
        correlation_matrix(array): 
        roi_count(list): number of ROIs per Network

    Returns: 
        (array): submatrix network (n, # ROI, # ROI)
    
    """
    submatrices = []
    start_idx = 0

    for count in roi_counts:

        end_idx = start_idx + count

        # Defining the initial and final index of a specific network based on a parcellation method
        print(f"Start idx: {start_idx}, end idx: {end_idx}")

        # Matrix slicing
        submatrix = correlation_matrix[start_idx:end_idx, start_idx:end_idx]

        submatrices.append(submatrix)

        start_idx = end_idx

    return submatrices

def read_coords(roi_file: str) -> list:

    """
        Parse and validate coordinates from file
    
    Args: 
        roi_file(str): roi file path

    Returns: 
        (list): list of roi's values

    """
    if not roi_file.endswith('.tsv'):
        raise ValueError('Coordinate file must be a tab-separated .tsv file')

    coords = pd.read_table(roi_file)

    # validate columns
    columns = [x for x in coords.columns if x in ['x', 'y', 'z']]
    if (len(columns) != 3) or (len(np.unique(columns)) != 3):
        raise ValueError('Provided coordinates do not have 3 columns with '
                         'names `x`, `y`, and `z`')

    # convert to list of lists for nilearn input
    return coords.values.tolist()


def signal_extract(data: list, atlas: str, t_r: float = None, masker_type: str = 'Maps', config: dict = {}, saveas: str = 'file') -> list:
    
    """
        Extracts BOLD time-series from regions of interest
        These regions can be from coordinates (Spheres), probabilistic maps, binary maps (Maps) or by labeled maps (Labels)
        Please see https://nilearn.github.io/stable/modules/generated/nilearn.maskers

    Args: 
        data(list): Filenames of subjects.
        atlas: Regions or coordinates to extract signals from. Can be your own atlas or from nilearn atlas
        t_r: TR if need it.
        masker_type: Type of masker used to extract BOLD signals. types are : 'Spheres','Maps','Labels'
        config: dictionary with the configurations by masker_type. Following the nilearn parameters.
        saveas: Destination to save and load output (.npz)

    Returns: 
        (array): subject_ts, array-like , 2-D (n_subjects,n_regions)
        (str): roi_img, ROI image or seeds path
        (list): labels (columns names) for time series extracted
        (str): masker_type, name of masker type applied

    """

    if os.path.exists(saveas):
        npzfile = np.load(saveas)
        return npzfile['data'], npzfile['subjects_ts'], npzfile['roi_img'], npzfile['labels'], npzfile['masker_type']

    # Variables setup
    
    radius = config.get('radius', 4)
    allow_overlap = config.get('allow_overlap', True)
    detrend = config.get('detrend', False)
    standardize = config.get('standardize', 'zscore_sample')

    low_pass = config.get('low_pass', None)
    high_pass = config.get('high_pass', None)
    memory_level = config.get('memory_level', 0)
    smoothing_fwhm = config.get('smoothing_fwhm', None)

    resampling_target = config.get('resampling_target', 'data')
    confounds = config.get('confounds', None)
    verbose = config.get('verbose', 0)
    reports = config.get('reports', True)

    if isinstance(atlas, str) and atlas.endswith('.tsv'):
        roi = read_coords(atlas)
        n_rois = len(roi)
        is_coords = True
    elif isinstance(atlas, (list, np.recarray)):
        is_coords = True
    else:
        roi = load_img(atlas)
        n_rois = len(np.unique(roi.get_fdata())) - 1
        is_coords = False

    masker = None
    if masker_type == 'Spheres':
        if is_coords and radius is None:
            warnings.warn('No radius specified for coordinates; setting to default of extracting from a single voxel')
        masker = NiftiSpheresMasker(seeds=atlas, smoothing_fwhm=smoothing_fwhm, radius=radius, allow_overlap=allow_overlap,
                                    detrend=detrend, standardize=standardize, low_pass=low_pass, high_pass=high_pass, t_r=t_r)
    elif masker_type == 'Maps':
        masker = NiftiMapsMasker(maps_img=atlas, standardize=standardize, high_pass=high_pass, low_pass=low_pass,
                                 detrend=detrend, t_r=t_r, memory_level=memory_level, smoothing_fwhm=smoothing_fwhm,
                                 resampling_target=resampling_target, verbose=verbose, reports=reports)
    elif masker_type == 'Labels':
        masker = NiftiLabelsMasker(labels_img=atlas, standardize=standardize, high_pass=high_pass, low_pass=low_pass,
                                   detrend=detrend, t_r=t_r, memory_level=memory_level, smoothing_fwhm=smoothing_fwhm,
                                   resampling_target=resampling_target, verbose=verbose)
    else:
        raise ValueError("Please provide masker type (Spheres, Maps, Labels).")

    global time_series

    def process_file(func_file):
        try:
            print(f"Processing data: {func_file}")
            func_file = nib.load(func_file)
            time_series = masker.fit_transform(func_file, confounds=confounds)
            return time_series
        except Exception as e:
            logging.exception("An exception was thrown!", exc_info=True)
            return None

    # subjects_ts = [process_file(func_file) for func_file in data if process_file(func_file) is not None]
    subjects_ts = [result for func_file in data if (result := process_file(func_file)) is not None]

    if isinstance(masker, NiftiMapsMasker):
        labels = ['map {}'.format(i) for i in range(time_series.shape[1])]
        roi_img = masker.maps_img_
    elif isinstance(masker, NiftiLabelsMasker):
        labels = ['roi {}'.format(i) for i in range(n_rois)]
        roi_img = masker.labels_img_
    elif isinstance(masker, NiftiSpheresMasker):
        labels = ['roi {}'.format(i) for i in range(len(masker.seeds_))]
        roi_img = masker.seeds_

    np.savez(saveas, data=data, subjects_ts=subjects_ts, roi_img=roi_img, labels=labels, masker_type=masker_type)
    return data, subjects_ts, roi_img, labels, masker_type, masker

def extract_fc(subj, code, mask, mask_type, kind, fname, fname_sig, config: dict = {}, save_results=True):
    

    # ts shape (n_subjects), and in each subject shape (vols, rois)
    config = config if config is not None else {}
    fname = fname + f'/{kind.replace(" ", "_")}'
    # ts shape (n_subjects), and in each subject shape (vols, rois)
    ts, roi_img, labels, masker_type = signal_extract(subj, mask, masker_type=mask_type, config=config)

    
    #print(ts)
    row=len(ts)
    column=len(ts[0])
    print(f'Rows:{row}, Column:{column}')
    print("Shape of a list (ts):",len(ts))
    np.savez(fname_sig)
    
    connectome_measure = ConnectivityMeasure(kind=kind)
    correlation_matrices = connectome_measure.fit_transform(ts)
    mean_correlation_matrix = connectome_measure.mean_

    # correlation_matricesReshaped = correlation_matrices.reshape(correlation_matrices.shape[0], -1)
    # correlation_matricesReshaped.to_csv('/home/kevrodz/Documents/Master/scripts/connectome_measure_corr.csv', index=False)

    if save_results:
        # Create the folder
        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            shutil.rmtree(fname)  # Removes all the subdirectories
            os.makedirs(fname)
        # Save the conn matrix into their respective folder
         
        for i in range(correlation_matrices.shape[0]):
            savetxt(f'{fname}/subj__{i}_connectome_measure_{kind.replace(" ", "_")}.csv', correlation_matrices[i], delimiter=',')
        savetxt(f'{fname}/subj_all_connectome_measure_mean_{kind.replace(" ", "_")}.csv', mean_correlation_matrix, delimiter=',')
    #connectome_measure_cov = ConnectivityMeasure(kind='covariance')  # covariance
    #connectome_measure_pcorr = ConnectivityMeasure(kind='partial correlation')  # partial correlation
    return correlation_matrices, mean_correlation_matrix


def atlas_seitzman(objects=True, func_images=None, path=None, description=False, regions=False, networks=False, plotting_atlas=False, 
                   plot_network=None, name_network_plot=None, path_network=None): 
    
    '''
    --- Seitzman atlas 2018 ----
   
        Creation of masker object with Seitzman 300 ROIs

        NiftiSpheresMasker is useful when data from given seeds should be extracted. 
        Use case: Summarize brain signals from seeds that were obtained from prior knowledge.

    Args: 
        objects(bool): flag to determine if compute only the objects. False for fit_transform in func images
        func_images(list): list with nibabel read images
        path(str): dir to save the fig (output_file='SSeitzman_300_ROI_atlas.png')
        description(bool): True if print Seitzman description
        regions(bool): True if print Seitzman regions
        networks(bool): True if print Seitzman regions
        plotting_atlas(bool): True if plot the whole atlas with nilearn
        plot_network(bool): True if plot a specific network from the atlas with nilearn

    Returns: 
        (array): subjects with Seitzman parcellation
        (obj): Seitzman atlas nilearn object
        (objt): Seitzman mask nilearn object

    '''

    atlas_seitzman  = datasets.fetch_coords_seitzman_2018()

    if description: 
        print('Description: ', atlas_seitzman['description'], '\n') 
    
    if regions: 
        print('Regions: ', np.unique(atlas_seitzman['regions']), '\n')
        
    if networks: 
        print('Networks: ', np.unique(atlas_seitzman['networks']), '\n')

    if plotting_atlas: 
        atlas_seitzman_coords = np.vstack((atlas_seitzman.rois['x'], (atlas_seitzman.rois['y'], (atlas_seitzman.rois['z'])))).T # Atlas coordinates

        flat = np.column_stack(np.apply_along_axis(np.unique, 0, atlas_seitzman.networks, return_inverse=True)[1])

        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(set(atlas_seitzman.networks))))
        newcmap = ListedColormap(colors)

        plotting.plot_markers(node_values=flat, node_size=atlas_seitzman.radius, node_coords=atlas_seitzman_coords, title='Seitzman 300 ROI atlas', node_vmin=0, 
                            node_vmax=len(newcmap.colors), node_cmap=newcmap, output_file=path, colorbar=True)
    if plot_network: 

        network = np.where(atlas_seitzman.networks == name_network_plot)

        atlas_seitzman_coords_dmn = np.vstack((atlas_seitzman.rois['x'][network], (atlas_seitzman.rois['y'][network], (atlas_seitzman.rois['z'][network])))).T # Coordinates of atlas
 
        flat = np.column_stack(np.apply_along_axis(np.unique, 0, atlas_seitzman.networks[network], return_inverse=True)[1])

        cmap = plt.get_cmap('viridis')

        colors = cmap(np.linspace(0, 1, len(set(atlas_seitzman.networks))))
        newcmap = ListedColormap(colors)

        plotting.plot_markers(node_values=flat, node_size=atlas_seitzman.radius[network], node_coords=atlas_seitzman_coords_dmn, title=f'Seitzman {name_network_plot}', node_vmin=0, 
                            node_cmap=newcmap, output_file=path_network)
        plt.show()


    max_radius = max(atlas_seitzman.radius)
    
    masker = NiftiSpheresMasker(seeds=atlas_seitzman.rois, radius = max_radius, allow_overlap=True, 
                                standardize=True, verbose=5) #memory= 'nilearn_cache'
    if not objects: 
        subjects_seitzman = masker.fit_transform(func_images) #[masker.fit_transform(func_images[i]) for i in range(len(func_images))] #range(len(func_images))

        return subjects_seitzman, atlas_seitzman, masker
    
    return atlas_seitzman, masker


def atlas_yeo(objects: bool = True, func_images: list = None, path_yeo_net: str = None, path_yeo_MNI: str = None, 
              thickness = 'thick_7', path_output: str = None, description: bool = False, regions: bool =False, plotting_atlas: bool =False): 

    '''
    --- Yeo atlas 2011 ----
   
        Creation of masker object with Yeo - 7 or 17 networks

        NiftiLabelMasker is useful when data from non-overlapping volumes should be extracted
        Use case: Summarize brain signals from clusters that were obtained by prior K-means or Ward clustering

    Args: 
        func_images(list): list with nibabel read images
        thickness(str): path to NifTI file containigin 7 or 17 regions parcellation fitted to thick template cortex segmentations
        path_yeo_net(str): path to Yeo networks-colors in 1000subjects_reference folder (./1000subjects_reference/17NetworksOrderedNames.csv)
        path_yeo_MNI(str): path to Yeo MNI coordinates in 1000subjects_reference folder ('./1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152/Centroid_coordinates/Yeo2011_17Networks_N1000.split_components.FSL_MNI152_1mm.Centroid_RAS.csv')
        description(bool): True if resume the atlas
        regions(bool): True if print Yeo regions and networks
        plotting_atlas(bool): True if plot the atlas

    Returns: 
        (array): subjects with Yeo parcellation
        (obj): Yeo atlas nilearn object
        (objt):  Yeo mask nilearn object
        (list): networks presented in Yeo
        (list): regions involved in Yeo atlas

        subjects_yeo, atlas_func_yeo, masker, yeo_nets, yeo_regions_MNI

    '''
    # https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering#parcellation-release

    atlas_func_yeo = datasets.fetch_atlas_yeo_2011()

    path_yeo_net = "H:/Semestres_UdeA/SEMESTRE_2022_II/Brain_Parcellation_Nilearn/1000subjects_reference/7NetworksOrderedNames.csv"
    path_yeo_MNI = "H:/Semestres_UdeA/SEMESTRE_2022_II/Brain_Parcellation_Nilearn/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152/Centroid_coordinates/Yeo2011_17Networks_N1000.split_components.FSL_MNI152_1mm.Centroid_RAS.csv"

    if description: 
        print('Description: ', atlas_func_yeo['description'], '\n')

    if regions: 
        yeo_nets = pd.read_csv(path_yeo_net) # './1000subjects_reference/17NetworksOrderedNames.csv'
        print(yeo_nets)
        yeo_regions_MNI = pd.read_csv(path_yeo_MNI) # './1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152/Centroid_coordinates/Yeo2011_17Networks_N1000.split_components.FSL_MNI152_1mm.Centroid_RAS.csv'
        print(yeo_regions_MNI)

    masker = NiftiLabelsMasker(labels_img=atlas_func_yeo[thickness], 
                               standardize=True, 
                               memory= 'nilearn_cache',
                               verbose=5)
    if not objects: 
        subjects_yeo = masker.fit_transform(func_images)# [masker.fit_transform(func_images[i]) for i in range(2)] # confounds
        return subjects_yeo, atlas_func_yeo, masker, yeo_nets, yeo_regions_MNI

    if plotting_atlas: 
        plotting.plot_roi(
                atlas_func_yeo.thick_7,
                title="Original Yeo atlas",
                cut_coords=(8, -4, 9),
                colorbar=True,
                output_file=path_output,
                cmap="Paired",
            )
    
    return atlas_func_yeo, masker


def atlas_schaefer(objects: bool = True, func_images: list = None, n_rois: int = 400, resolution_mm: int = 1, yeo_networks: int = 7,
                description: bool = False, regions: bool = False, path_output:str = None,
                networks: bool = False, plotting_atlas: bool = False): 

    '''
    --- Schaefer atlas 2018 ----
   
        Creation of masker object with Schaefer - 7 or 17 networks (ROI annotation according to yeo networks) (n_rois from 100 to 1000, steps 100)

        NiftiLabelMasker is useful when data from non-overlapping volumes should be extracted
        Use case: Summarize brain signals from clusters that were obtained by prior K-means or Ward clustering

    Args: 
        func_images(list): list with nibabel read images
        n_rois(int): number of Regions of Interest. Default 400
        resolution_mm(int): spatial resolution of atlas image in mm. Default 1 mm
        yeo_networks(int):  number of networks. Default 17 networks
        description(bool): True if resume the atlas
        regions(bool): True if print Schaefer regions
        networks(bool): True if print Schaefer networks
        plotting_atlas(bool): True if plot the atlas

    Returns: 
        (array): subjects with Schaefer parcellation
        (obj): Schaefer mask nilearn object
        (objt): Schaefer atlas nilearn object
        (list): Schaefer network names

        subjects_yeo, atlas_func_yeo, masker, yeo_nets, yeo_regions_MNI

    '''

    atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=yeo_networks, resolution_mm=resolution_mm)

    masker = NiftiLabelsMasker(labels_img=atlas_schaefer['maps'], 
                               standardize=True,
                               verbose=5)
    if description: 
        print('Description: ', atlas_schaefer['description'], '\n')

    if networks: 
        with open('C:/Users/user/nilearn_data/schaefer_2018/Schaefer2018_400Parcels_17Networks_order.txt') as f:
            schaefer_names = f.read()
            print(schaefer_names)

    if regions: 
        print(f"Labels: {atlas_schaefer['labels']}")

        
    if plotting_atlas: 
        plotting.plot_stat_map(atlas_schaefer.maps, title='Schaefer 400 atlas',
                  cut_coords=(8, -4, 9), vmax=400, cbar_tick_format='%i', colorbar=True, cmap='Paired', 
                  output_file=path_output)
                  #, output_file='Schaefer_400_atlas.png'
                    
    
    if not objects: 
        subjects_schaefer = masker.fit_transform(func_images)#[masker.fit_transform(func_images[i]) for i in range(2)]
        return subjects_schaefer, masker, atlas_schaefer
    return atlas_schaefer, masker


def dp(dist_mat):

    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.

    Args: 
        (array): distance matrix (ROI, ROI)

    Returns: 
        (list): traceback backup
        (array): cost matrix
    """

    N, M = dist_mat.shape

    # Initialize the cost matrix
    cost_mat = np.zeros((N+1, M+1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i,j],       # match (0)
                cost_mat[i, j +1 ],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # insertion
            i = i - 1
        elif tb_type == 2:
            # deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:,1:]
    
    return (path[::-1], cost_mat)

def dtw_function(time_series, window): 

    '''
        It reflects a distance measure of the amount of 
        warping required to align two time series

        HAVE TO NORMALIZED JUST FOR MEAN = 0 (EACH VOXEL)! INSTEAD OF Z-SCORE (LIKE PC)

        'DTW distances were then multiplied by -1 and demeaned to transform distance measures 
        to similarity estimates that follow a normal distribution around 0, with values below 0 
        reflecting below average similarity of the time series and values above 0 reflecting above 
        average similarity'

    Args:
        time_series(array): BOLD signal time series matrix
        window(int): how many windows to compute DTW

    Returns: 
        (list): path traceback backup
        (array): cost matrix
        (list): similarity computed ROIs
        
    '''
    
    similarity_rois = []
    path_list = []
    costmat_list = []

    for i in range(time_series.shape[0]): 
        for j in range(time_series.shape[1]): 
            N = time_series[:,i].shape[0]
            M = time_series[:,j].shape[0]
            dist_mat = np.zeros((N, M))
            for z in range(N): 
                for k in range(M): 
                    dist_mat[z,k] = abs(time_series[:,i][z] - time_series[:,j][k])

                    path, cost_mat = dp(dist_mat)

                    path_list = np.append(path_list, path)
                    costmat_list = np.append(costmat_list, cost_mat)
                    
                    similarity_rois = np.append(similarity_rois, cost_mat[N - 1, M - 1]/(N + M))
    
    return path, cost_mat, similarity_rois

def dtw(s, t):

    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix

def main_dtw(time_series):

    mat_dtw  = []
    for i in range(time_series.shape[1]): 
        for j in range(time_series.shape[1]): 
            matrix_dtw = dtw(time_series[:, i], time_series[:, j])
            mat_dtw.append(matrix_dtw)

    return np.array(mat_dtw).reshape(time_series.shape[1], time_series.shape[1])

def demean_func(time_series): 

    new_data = []
    for i in range(len(time_series)):
        for j in range(len(time_series[0])): 
            new_data.append(time_series[i, j] - np.mean(time_series))
    
    return np.array(new_data).reshape(164, 300)

def func_dtw_fast(series): 

    dtw_matrix = []

    for i in range(series.shape[1]): 
        for j in range(series.shape[1]): 
            dtw_result = fastdtw(series[:,i], series[:,j])[0]
            dtw_matrix.append(dtw_result)

    array_reshaped = np.array(dtw_matrix).reshape(300, 300)
    min_val = np.min(array_reshaped)
    max_val = np.max(array_reshaped)

    dtw_matrix_normalized = (array_reshaped - min_val) / (max_val - min_val)

    return dtw_matrix_normalized