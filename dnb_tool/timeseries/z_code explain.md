Alright, I'll go through this Python code step-by-step to explain what each part does, without assuming a biology background. This code seems to be implementing parts of the Early Warning Signal (EWS) and Dynamic Network Biomarker (DNB) detection method, specifically focusing on calculating the EWS and then finding a change point within it.

The code defines two main functions: `EWS_DNB` and `CPD_EWS`, along with a helper function `CPDotsu` and a `timer` decorator.

---

### `timer` decorator

Python

```
def timer(func):
    def wrapper(*args, **kwargs):
        time_s = time.time()
        res = func(*args, **kwargs)
        print(f'calculation time ={time.time() - time_s}sec')
        return res
    return wrapper
```

- **What it does:** This is a Python "decorator." Its purpose is to measure how long a function takes to execute.
    
- **How it works:** When you put `@timer` above a function definition (like `@timer def EWS_DNB(...)`), it means that whenever `EWS_DNB` is called, it will actually run the `wrapper` function. The `wrapper` function records the start time, calls the original `EWS_DNB` function, records the end time, prints the difference, and then returns the result of `EWS_DNB`.
    
- **Purpose:** Useful for performance monitoring and debugging.
    

---

### `EWS_DNB(x, window_size, padding='online', normalization='straight')`

This function calculates the Early Warning Signal (EWS) based on the standard deviation or the largest eigenvalue of the covariance matrix, often reflecting "critical slowing down."

Python

```
@timer
def EWS_DNB(x, window_size, padding='online', normalization='straight'):
    print('caluculating time series DNB:')
```

- **`@timer`**: As explained, this will time the execution of this function.
    
- **`x`**: This is your input data, a time series. It could be 1-dimensional (e.g., a single measurement over time) or N-dimensional (e.g., gene expression levels for many genes over time).
    
- **`window_size`**: This is the size of the sliding window used to calculate statistics locally in time. For example, if `window_size` is 10, calculations at time `t` will use data from `t-9` to `t`.
    
- **`padding`**: How to handle the beginning and end of the time series where a full window isn't available.
    
- **`normalization`**: How to scale the input data.
    

#### **Normalization Step:**

Python

```
    # normalization
    if normalization == 'std':
        x = x / x.std(0)
    elif normalization == 'minmax':
        x = x / (x.max(0) - x.min(0))
    elif normalization == 'PCA':
        if x.shape[1] < 10:
            raise NameError('low dimmention.')
            return -1
        pca = PCA(n_components=10)
        x = pca.fit_transform(x)
    elif normalization != 'straight':
        raise NameError('select \'straight\',\'PCA\', \'minmax\', or \'std\'')
        return -1
```

- **`normalization == 'std'`**: Divides the data by its standard deviation along each feature (column). This makes each feature have a standard deviation of 1.
    
- **`normalization == 'minmax'`**: Divides the data by the range (max - min) of each feature. This scales data to a 0-1 range (if minima are 0, which isn't explicitly ensured here, but it scales relative magnitudes).
    
- **`normalization == 'PCA'`**:
    
    - Checks if the number of features (`x.shape[1]`) is less than 10. If so, it raises an error, indicating PCA might not be meaningful for very low-dimensional data.
        
    - If suitable, it performs **Principal Component Analysis (PCA)**. PCA transforms the data into a new set of orthogonal (uncorrelated) variables called principal components. Here, it keeps only the top 10 principal components.
        
    - **Purpose:** PCA is used for dimensionality reduction and to decorrelate the data. This means the subsequent covariance calculation might be more robust or focus on the main modes of variation.
        
- **`normalization == 'straight'`**: No normalization is applied; the data is used as is.
    
- **`else`**: If an unknown normalization method is specified, it raises an error.
    

#### **Core EWS Calculation (Sliding Window):**

Python

```
    # 1 dim or n dim
    if len(x.shape) == 1:
        xs = sliding_window(x, window_size)
        cov_time_tmp = xs.std(1)
    else:
        x = x.reshape(x.shape[0], -1) # Ensure x is 2D (time_points, features)
        xs = sliding_window(x, (window_size, x.shape[1])).reshape(-1,
                                                         window_size, x.shape[1])
        cov_time_tmp = np.zeros(xs.shape[0])
        for i in tqdm.tqdm(range(0, cov_time_tmp.shape[0])):
            sigmas, _ = np.linalg.eigh(np.cov(xs[i].T)) # Transpose for np.cov input
            cov_time_tmp[i] = sigmas.max()
```

- **`if len(x.shape) == 1`**: This block handles a 1-dimensional time series (e.g., just one gene's expression over time).
    
    - `xs = sliding_window(x, window_size)`: Creates a view of the data where each "row" is a window of `window_size` consecutive data points.
        
    - `cov_time_tmp = xs.std(1)`: Calculates the **standard deviation** within each window. In 1D, standard deviation itself can serve as an EWS, indicating increasing fluctuations.
        
- **`else`**: This block handles N-dimensional time series (multiple features/genes over time).
    
    - `x = x.reshape(x.shape[0], -1)`: Ensures the input `x` is reshaped into a 2D array (time points x features), even if it was originally higher-dimensional.
        
    - `xs = sliding_window(x, (window_size, x.shape[1])).reshape(-1, window_size, x.shape[1])`: This is creating the sliding windows for multi-dimensional data. Each element in `xs` (e.g., `xs[i]`) will be a block of data with shape `(window_size, num_features)`.
        
    - `for i in tqdm.tqdm(range(0, cov_time_tmp.shape[0]))`: Loops through each window. `tqdm` provides a progress bar.
        
    - `np.cov(xs[i].T)`: Calculates the **covariance matrix** for the data within the current window `xs[i]`. `xs[i]` has shape `(window_size, num_features)`. For `np.cov`, variables should be rows, so `xs[i].T` (transpose) is used, making it `(num_features, window_size)`.
        
    - `sigmas, _ = np.linalg.eigh(...)`: Calculates the eigenvalues of the covariance matrix. `np.linalg.eigh` is used for symmetric matrices (like covariance matrices), providing better numerical stability and performance than `eig`. `sigmas` will contain the eigenvalues, sorted (usually ascending).
        
    - `cov_time_tmp[i] = sigmas.max()`: Stores the **largest eigenvalue** of the covariance matrix for the current window. This is the common EWS metric for multi-variate data, as it indicates the direction of largest variance, which is expected to increase drastically near a bifurcation (critical slowing down).
        

#### **Padding Step:**

Python

```
    if padding == 'same':
        # padding marage data using the edge
        cov_time = np.zeros(x.shape[0])
        cov_time[window_size//2:-window_size//2+1] = cov_time_tmp
        cov_time[:window_size//2] = cov_time_tmp[0]
        cov_time[-window_size//2+1:] = cov_time_tmp[-1]
        return cov_time
    elif padding == 'online':
        # cov_time[t] is calculated as time-sereis data t - window_size : t
        cov_time = np.zeros(x.shape[0])
        cov_time[:window_size-1] = cov_time_tmp[0]
        cov_time[window_size-1:] = cov_time_tmp
        return cov_time
    elif padding == 'valid':
        return cov_time_tmp
    else:
        raise NameError('select \'same\', \'online\', or \'valid\' ')
        return -1
```

- The `cov_time_tmp` calculated above has `N - window_size + 1` data points (for `N` total time points). Padding extends this to the original length of `x` (`x.shape[0]`).
    
- **`padding == 'same'`**: Centers the `cov_time_tmp` results and extends the values at the edges. The first `window_size // 2` values are filled with the first calculated value, and the last `window_size // 2` values are filled with the last calculated value.
    
- **`padding == 'online'`**: This is a common choice for online (real-time) detection. It fills the initial `window_size - 1` positions with the first calculated value. This means `cov_time[t]` represents the EWS calculated using data up to time `t`.
    
- **`padding == 'valid'`**: No padding is applied. Only the calculated `cov_time_tmp` (which has a shorter length) is returned.
    
- **Purpose:** To align the EWS time series with the original time series, making it easier to interpret when an EWS signal appears relative to the full data.
    

---

### `CPDotsu(ews)` (Helper for Change Point Detection)

This function implements Otsu's method to find an optimal threshold in the EWS time series, which is then used to identify a change point.

Python

```
def CPDotsu(ews):
    def OtsuScore(data, thresh):
        w_0 = np.sum(data <= thresh)/data.shape[0]
        w_1 = np.sum(data > thresh)/data.shape[0]
        # check ideal case
        if (w_0 == 0) | (w_1 == 0):
            return 0
        mean_all = data.mean()
        mean_0 = data[data <= thresh].mean()
        mean_1 = data[data > thresh].mean()
        sigma2_b = w_0 * ((mean_0 - mean_all)**2) + \
            w_1 * ((mean_1 - mean_all)**2)

        return sigma2_b
    ths = (ews.max() - ews.min()) * np.arange(0, 1, 1e-4) + ews.min()
    scores = np.zeros(ths.shape[0])
    for i in range(ths.shape[0]):
        scores[i] = OtsuScore(ews, ths[i])
    y_ews = np.zeros(ews.shape[0])
    y_ews[ews > ths[scores.argmax()]] = 1

    max_time = ews.argmax()
    cp = max_time - y_ews[:max_time][::-1].argmin()
    return cp
```

- **`OtsuScore(data, thresh)`**: This inner function calculates the "between-class variance" for a given threshold, which is the core of Otsu's method.
    
    - `w_0`, `w_1`: Proportions of data below and above the `thresh`.
        
    - `mean_0`, `mean_1`: Means of data below and above the `thresh`.
        
    - `mean_all`: Overall mean.
        
    - `sigma2_b`: The between-class variance. Otsu's method aims to maximize this value.
        
- **`ths = (ews.max() - ews.min()) * np.arange(0, 1, 1e-4) + ews.min()`**: Generates a range of possible threshold values to test, spanning from the minimum to the maximum of the EWS, with small increments.
    
- **`scores = np.zeros(ths.shape[0])`... `scores[i] = OtsuScore(ews, ths[i])`**: Calculates the Otsu score for each possible threshold.
    
- **`y_ews = np.zeros(ews.shape[0])`... `y_ews[ews > ths[scores.argmax()]] = 1`**: Binarizes the EWS time series. Values above the optimal Otsu threshold (the one with the `scores.argmax()`) are set to 1, others to 0. This creates a binary signal indicating where the EWS is "high."
    
- **`max_time = ews.argmax()`**: Finds the index (time point) where the EWS reaches its maximum value.
    
- **`cp = max_time - y_ews[:max_time][::-1].argmin()`**: This is the actual change point detection.
    
    - `y_ews[:max_time]`: Takes the binary EWS up to the peak.
        
    - `[::-1]`: Reverses this segment.
        
    - `argmin()`: Finds the first occurrence of `0` (meaning the EWS dropped below the threshold) when scanning _backwards_ from the peak.
        
    - `max_time - ...`: Calculates the actual index in the original `ews` array.
        
    - **Purpose:** This method tries to find the time point _before_ the peak of the EWS where the signal significantly rises above a background level, indicating the onset of the pre-disease state.
        

---

### `CPD_EWS(ews, cfg={'type': 'ar', 'dim': 2}, scope_range=np.inf)`

This function orchestrates different change point detection algorithms.

Python

```
@timer
def CPD_EWS(ews, cfg={'type': 'ar', 'dim': 2}, scope_range=np.inf):
    print('caluculating change point:')
    max_time = ews.argmax()
    window_size = min(scope_range, max_time)
    ews_calc = ews[max_time - window_size:max_time] / \
        ews[max_time - window_size:max_time].std()
```

- **`ews`**: The Early Warning Signal time series (output from `EWS_DNB`).
    
- **`cfg`**: A dictionary specifying the type of change point detection algorithm and its parameters.
    
- **`scope_range`**: Limits the portion of the EWS where the change point is searched, relative to the peak.
    
- **`max_time = ews.argmax()`**: Finds the peak of the EWS.
    
- **`ews_calc = ...`**: Selects a segment of the EWS _before_ its peak (from `max_time - window_size` to `max_time`) and normalizes its standard deviation to 1. This normalized segment is what the change point algorithms will operate on.
    
- **Purpose:** To find the exact time point where the EWS signal began to significantly deviate from its "normal" state, indicating the onset of the pre-disease state.
    

#### **Change Point Detection Algorithms:**

Python

```
    if cfg['type'] == 'peak':
        cp = max_time
    elif cfg['type'] == 'ohtsu':
        cp = max_time - window_size + CPDotsu(ews_calc)
    elif cfg['type'] == 'linear':
        algo = rpt.Dynp(model='linear', min_size=1, jump=1).fit(
            ews_calc.reshape(-1, 1))
        cp = max_time - window_size + algo.predict(1)[0]
    elif cfg['type'] == 'ar':
        algo = rpt.Dynp(model='ar', params={"order": cfg['dim']}).fit(ews_calc)
        cp = max_time - window_size + algo.predict(1)[0]
    else:
        raise NameError('select \'peak\',\'linear\',\'ar\', \'ohtsu\',')
        return -1
    return cp
```

- **`cfg['type'] == 'peak'`**: Simplest method. The change point is just the time of the EWS peak. This is likely a baseline or a fallback.
    
- **`cfg['type'] == 'ohtsu'`**: Uses the `CPDotsu` function described above on the `ews_calc` segment.
    
- **`cfg['type'] == 'linear'`**: Uses the `ruptures` library, specifically the `Dynp` (Dynamic Programming) algorithm with a `linear` model. This model assumes that the mean of the signal changes.
    
    - `fit(...)`: Trains the algorithm on the `ews_calc` segment.
        
    - `predict(1)`: Predicts 1 change point.
        
- **`cfg['type'] == 'ar'`**: Uses `ruptures.Dynp` with an `ar` (autoregressive) model. This model assumes that the autoregressive properties (how future values depend on past values) of the signal change.
    
    - `params={"order": cfg['dim']}`: Specifies the order of the AR model (e.g., how many previous time points influence the current one).
        
- **Purpose of `max_time - window_size + ...`**: The `ruptures` library (and `CPDotsu` within `CPD_EWS`) returns an index relative to the _start_ of the `ews_calc` segment. This calculation converts that relative index back to the absolute index in the original `ews` time series.
    

---

In essence, this Python script provides tools to:

1. Calculate an Early Warning Signal (EWS) for a time series, often represented by the largest eigenvalue of the covariance matrix within a sliding window.
    
2. Apply various change point detection algorithms (including one based on Otsu's method and others from the `ruptures` library) to this EWS time series to pinpoint when the "pre-disease state" likely began.
    

The `x.std(0)` and `x.max(0) - x.min(0)` parts, along with `np.cov(xs[i].T)`, confirm that `x` is expected to be a 2D array where columns are features (like genes) and rows are time points.
