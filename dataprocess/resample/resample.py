import rasterio
import rioxarray
from rasterio.warp import Resampling
import numpy as np
import pandas as pd
import xarray as xr


def resampling(
    data_array,
    target_resolution,
    new_shape=None,
    target_crs="EPSG:4326",
    resampler=Resampling.cubic,
):
    """
    Resamples a given DataArray to a new resolution and coordinate reference system (CRS).

    Parameters
    ----------
    data_array : xarray.DataArray
        The input DataArray that needs to be resampled, typically containing spatial data.
    target_resolution : float
        The target resolution for resampling. Defines the spatial resolution (e.g., in degrees or meters).
    new_shape : tuple of int, optional
        The shape (height, width) of the output array after resampling. If None, it is automatically calculated
        based on `target_resolution`.
    target_crs : str, optional
        The target coordinate reference system in EPSG format. Common codes include "EPSG:4326" (WGS 84) and
        "EPSG:3857" (Web Mercator). Default is "EPSG:4326".
    resampler : rasterio.enums.Resampling, optional
        The resampling algorithm to use. Options include nearest, bilinear, cubic, cubic_spline, lanczos,
        average, mode, gauss, max, min, med, sum, etc. Default is `Resampling.cubic`.
        See https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling for more details.

    Returns
    -------
    xarray.DataArray
        The resampled DataArray with the specified resolution and CRS.
    """

    # Assign a CRS to the DataArray
    data_array = data_array.rio.write_crs(target_crs)

    # Calculate the new shape
    if new_shape is None:
        scale_factor = (
            data_array.rio.resolution()[0] / target_resolution,
            data_array.rio.resolution()[1] / target_resolution,
        )
        new_shape = (
            int(data_array.shape[1] * scale_factor[0]),
            int(data_array.shape[2] * scale_factor[1]),
        )

    # Reproject the DataArray
    reprojected_data_array = data_array.rio.reproject(
        data_array.rio.crs,
        shape=new_shape,
        resampling=resampler,
    )

    reprojected_data_array = reprojected_data_array.rename(
        {"x": "longitude", "y": "latitude"}
    )
    for coord in set(reprojected_data_array.coords) - set(
        ["longitude", "latitude", "time"]
    ):
        reprojected_data_array = reprojected_data_array.drop_vars(coord)

    return reprojected_data_array


def yearly_resample(
    data_array,
    target_resolution,
    new_shape=None,
    target_crs="EPSG:4326",
    resampler=Resampling.cubic,
):
    """
    Cut large datasets by year, resample separately, and then combined.
    This approach is used to prevent memory explosions caused by large data sets.

    Parameters
    ----------
        The same as `resample` function.

    Returns
    -------
    xarray.Dataset
        The resampled Dataset with the specified resolution and CRS.
    """
    years = np.unique(data_array.time.dt.year.values)
    resampled_datasets = []
    org_sum, new_sum = 0.0, 0.0
    for year in years:
        yearly_data = data_array.sel(time=str(year))
        org_sum += yearly_data.values.sum()
        resampled_yearly_data = resampling(
            yearly_data,
            target_resolution,
            new_shape,
            target_crs,
            resampler,
        )
        new_sum += resampled_yearly_data.values.sum()
        resampled_datasets.append(resampled_yearly_data)
    print("sum before resample: ", org_sum)
    print("sum after resample: ", new_sum)
    resampled_datasets = xr.concat(resampled_datasets, dim="time")
    resampled_datasets = xr.Dataset({data_array.name: resampled_datasets})
    for coord in set(resampled_datasets.coords) - set(
        ["longitude", "latitude", "time"]
    ):
        resampled_datasets = resampled_datasets.drop_vars(coord)
    return resampled_datasets


def days_in_month(time):
    """
    Returns the number of days in the month for each timestamp in the provided time array.

    Parameters
    ----------
    time : xarray.DataArray, pandas.Series, or similar
        An array-like object with datetime-like data. The object should have a `dt` accessor, typically
        available in xarray DataArrays or pandas Series.

    Returns
    -------
    xarray.DataArray, pandas.Series, or similar
        An array-like object of the same type as the input, containing the number of days in the month
        corresponding to each timestamp in the input `time`.
    """
    return time.dt.days_in_month


def interpolate_daily(dataset):
    """
    Interpolates a dataset with monthly time resolution to a daily time resolution, distributing the monthly
    data evenly across each month.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        The input dataset with a "time" coordinate, where the time resolution is monthly.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        A new dataset reindexed to daily resolution, with values interpolated such that the monthly totals
        are evenly distributed across the days of each month. Missing days are filled using the nearest
        available data.
    """
    # define time span
    time_start = pd.Timestamp(dataset["time"].values.min())
    time_end = pd.Timestamp(dataset["time"].values.max())
    time_start = f"{str(time_start.year)}-01-01"
    time_end = f"{str(time_end.year)}-12-31"
    daily_time = pd.date_range(start=time_start, end=time_end, freq="D")
    # calculate average emissions monthly
    days = days_in_month(dataset["time"])
    days = xr.DataArray(days, coords=[dataset["time"]], name="days_in_month")
    new_dataset = dataset / days
    # reindex dataset to daily and fillna
    new_dataset = new_dataset.reindex(time=daily_time, method="nearest")
    return new_dataset
