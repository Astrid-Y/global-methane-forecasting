import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm


def snapshot(dataset, time_point, title, cmap="viridis", norm=None):
    """
    Plots a geographical heatmap snapshot of the dataset at a specific time point.

    Parameters
    ----------
    dataset : xarray.DataArray
        The dataset containing the data to be plotted. It must include coordinates for longitude and latitude.
    time_point : datetime-like
        The specific time point at which to extract and plot the data from the dataset.
    title : str
        The title of the plot.
    cmap : str, optional
        The colormap to be used for the heatmap. Default is 'viridis'.
    norm : matplotlib.colors.Normalize, optional
        The normalization to apply to the colormap. If None, none normalization is used.

    Raises
    ------
    ValueError
        If the dataset does not contain longitude and latitude coordinates.
    """

    if ("longitude" in dataset.coords) & ("latitude" in dataset.coords):
        lon = dataset["longitude"]
        lat = dataset["latitude"]
    elif ("lon" in dataset.coords) and ("lat" in dataset.coords):
        lon = dataset["lon"]
        lat = dataset["lat"]
    else:
        raise ValueError("Missing longitude or latitude in dataset")
    data_time_point = dataset.sel(time=time_point)
    # apply projection
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # draw heatmap
    heatmap = ax.pcolormesh(
        lon, lat, data_time_point, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm
    )
    # add coastlines and borders
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    # add colorbar and title
    cbar = plt.colorbar(heatmap, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label(f"{dataset.units}")
    time_str = time_point.strftime("%Y-%m-%d %H:00")
    plt.title(f"{title} on {time_str}")
    plt.show()


def snapshot_anima(dataset, time_points, title, cmap="viridis", norm=None):
    """
    Creates an animation of geographical heatmap snapshots from a dataset over specified time points.

    Parameters
    ----------
    dataset : xarray.DataArray
        The dataset containing the data to be animated. It must include coordinates for longitude and latitude.
    time_points : array-like of datetime-like
        A list or array of time points at which to extract and plot the data from the dataset.
    title : str
        The title for the animation. Each frame will include the title along with the specific time point.
    cmap : str, optional
        The colormap to be used for the heatmaps. Default is 'viridis'.
    norm : matplotlib.colors.Normalize, optional
        The normalization to apply to the colormap. If None, none normalization is used.

    Raises
    ------
    ValueError
        If the dataset does not contain longitude and latitude coordinates.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object that can be displayed or saved to a file.
    """

    if ("longitude" in dataset.coords) & ("latitude" in dataset.coords):
        lon = dataset["longitude"]
        lat = dataset["latitude"]
    elif ("lon" in dataset.coords) and ("lat" in dataset.coords):
        lon = dataset["lon"]
        lat = dataset["lat"]
    else:
        raise ValueError("Missing longitude or latitude in dataset")

    # apply projection
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # draw heatmap
    heatmap = ax.pcolormesh(
        lon,
        lat,
        dataset.isel(time=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )
    # add coastlines and borders
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    # add colorbar and title
    cbar = plt.colorbar(heatmap, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label(f"{dataset.units}")

    def update(frame):
        time_str = time_points[frame].strftime("%Y-%m-%d %H:00")
        ax.clear()
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        heatmap = ax.pcolormesh(
            lon,
            lat,
            dataset.isel(time=frame),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(f"{title} on {time_str}")
        return heatmap

    ani = FuncAnimation(fig, update, frames=len(time_points), blit=False)
    plt.ioff()
    plt.close(fig)
    return ani


def snapshot_muti(lon, lat, diff, title, cmap="Reds", units="kg m-2"):
    """
    Creates a series of geographical heatmaps for a series of days and displays them in a grid.

    Parameters
    ----------
    lon : xarray.DataArray or numpy.ndarray
        Longitude coordinates corresponding to the data.
    lat : xarray.DataArray or numpy.ndarray
        Latitude coordinates corresponding to the data.
    diff : dict of xarray.DataArray or numpy.ndarray
        Dictionary where keys are labels and values are 2D arrays representing the data to be plotted.
    title : str
        The overall title for the entire figure.
    cmap : str, optional
        The colormap to be used for the heatmaps. Default is 'Reds'.
    units : str, optional
        The units of the plotted data, which will be displayed on the colorbar. Default is 'kg m-2'.
    """
    row = (len(diff) + 2) // 3
    fig, axes = plt.subplots(
        row,
        3,
        figsize=(18, int(4.5 * row)),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for ax, k in zip(axes.ravel(), diff.keys()):
        heatmap = ax.pcolormesh(
            lon, lat, diff[k], transform=ccrs.PlateCarree(), cmap=cmap
        )
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_title(f"{k}")
        # Add a colorbar to the figure
        cbar = fig.colorbar(
            heatmap, ax=ax, orientation="horizontal", pad=0.05, aspect=50
        )
        cbar.set_label(f"{units}")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def anima_results(datasets, time_points, err="abs", anima=True):
    """
    Creates an animation or static plot of geographical heatmaps for ground truth, predictions, and errors over time.

    Parameters
    ----------
    datasets : list of xarray.DataArray
        A list of three xarray DataArrays, where each DataArray represents the data to be plotted: ground truth, prediction, and error, respectively.
    time_points : array-like of datetime-like
        A list or array of time points at which to extract and plot the data from the datasets.
    err : str, optional
        Specifies the type of error visualization. If 'abs', the absolute error is used with a red colormap.
    anima : bool, optional
        If True, generates an animation of the heatmaps over time. If False, creates a static plot showing the heatmaps at the initial time point. Default is True.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation or None
        If `anima` is True, returns a `FuncAnimation` object for the animation. If `anima` is False, displays a static plot and returns None.
    """
    lon = datasets[0]["longitude"]
    lat = datasets[0]["latitude"]

    titles = ["Ground truth", "Prediction", "Error"]
    if err == "abs":
        cmaps = ["rainbow", "rainbow", "Reds"]
    else:
        cmaps = ["rainbow", "rainbow", "coolwarm"]

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    # Initialize the heatmaps
    heatmaps = []
    for i, (ax, dataset, cmap) in enumerate(zip(axes, datasets, cmaps)):
        # if i == 2:  # Third plot
        #     heatmap = ax.pcolormesh(
        #         lon,
        #         lat,
        #         dataset.isel(time=0),
        #         transform=ccrs.PlateCarree(),
        #         cmap=cmap,
        #         vmin=-0.002,
        #         vmax=0.002,
        #     )
        # else:
        heatmap = ax.pcolormesh(
            lon, lat, dataset.isel(time=0), transform=ccrs.PlateCarree(), cmap=cmap
        )
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        heatmaps.append(heatmap)

        # Add a colorbar to the figure
        cbar = fig.colorbar(
            heatmap, ax=ax, orientation="horizontal", pad=0.05, aspect=50
        )
        cbar.set_label(f"{dataset.units}")

    if anima == False:
        plt.tight_layout()
        plt.show()
        return None

    def update(frame):
        time_str = time_points[frame].strftime("%Y-%m-%d %H:00")
        for i, (ax, heatmap, dataset, title, cmap) in enumerate(
            zip(axes, heatmaps, datasets, titles, cmaps)
        ):
            ax.clear()
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            # if i == 2:
            #     heatmap = ax.pcolormesh(
            #         lon,
            #         lat,
            #         dataset.isel(time=frame),
            #         transform=ccrs.PlateCarree(),
            #         cmap=cmap,
            #         vmin=0,
            #         vmax=0.0015,
            #     )
            # else:
            heatmap = ax.pcolormesh(
                lon,
                lat,
                dataset.isel(time=frame),
                transform=ccrs.PlateCarree(),
                cmap=cmap,
            )
            ax.set_title(f"{title} on {time_str}")
        return heatmaps

    ani = FuncAnimation(fig, update, frames=len(time_points), blit=False)
    plt.tight_layout()
    plt.ioff()
    plt.close(fig)
    return ani


def visual_results(time_point, ds_pred, ds_gt, days=15, hours=24):
    """
    Creates an animation of geographical heatmaps comparing predicted and ground truth data for a specified time period.

    Parameters
    ----------
    time_point : datetime-like
        The starting time point for which the results are visualized.
    ds_pred : xarray.Dataset
        The dataset containing the predicted values.
    ds_gt : xarray.Dataset
        The dataset containing the ground truth values.
    days : int, optional
        The number of days to include in the visualization, starting from the time point. Default is 15.
    hours : int, optional
        The number of hours of the next time point from the starting time point. Default is 24.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object showing the ground truth, predictions, and absolute errors over the specified time period.
    """
    time_point = pd.to_datetime(time_point)
    start_time = time_point + pd.Timedelta(hours=hours)
    end_time = start_time + pd.Timedelta(days=days - 1)
    ds_gt = ds_gt.sel(time=slice(start_time, end_time))
    ds_gt = ds_gt.sel(time=ds_gt["time"].dt.hour == time_point.hour)
    ds_pred = ds_pred.sel(time=time_point)
    time_points = pd.to_datetime(ds_gt["time"].values)
    pred_arr = np.zeros((days, 90, 180))
    for i in range(days):
        pred_arr[i] = ds_pred[f"d{i+1}"].values
    pred_arr = xr.DataArray(
        pred_arr,
        coords=[ds_gt["time"], ds_gt["latitude"], ds_gt["longitude"]],
        dims=["time", "latitude", "longitude"],
        name="cc",
        attrs={"units": "kg m-2"},
    )
    diff = ds_gt.copy()
    # diff['tc_ch4'].values = pred_arr.values - diff['tc_ch4'].values
    diff["tc_ch4"].values = np.abs(pred_arr.values - diff["tc_ch4"].values)
    print(diff["tc_ch4"].values.max(), diff["tc_ch4"].values.min())
    ani = anima_results([ds_gt["tc_ch4"], pred_arr, diff["tc_ch4"]], time_points)
    return ani


def visualize_loss(epoches, results, ax=None, title="Loss", log=False, yrange=None):
    """
    Plots loss values over epochs for training or validation, with optional logarithmic scaling and y-axis range.

    Parameters
    ----------
    epoches : array-like
        A sequence of epoch numbers corresponding to the loss values.
    results : dict or array-like
        If a dictionary, keys represent different loss types or metrics, and values are sequences of loss values for each type. If an array-like object, it is a sequence of loss values for a single metric.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the loss values. If None, a new figure and axes are created. Default is None.
    title : str, optional
        The title of the plot. Default is "Loss".
    log : bool, optional
        If True, sets the y-axis to a logarithmic scale. Default is False.
    yrange : tuple of float, optional
        A tuple specifying the y-axis limits as (min, max). If None, the y-axis limits are automatically determined. Default is None.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if isinstance(results, dict):
        for k, v in results.items():
            ax.plot(epoches, v, label=k)
        ax.legend()
    else:
        ax.plot(epoches, results)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if log:
        ax.set_yscale("log")
    if yrange:
        ax.set_ylim(yrange)
    ax.set_title(title)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)


def visualize_eval(days, results, title="Error", ax=None, errband=None):
    """
    Plots evaluation results over days with optional error bands.

    Parameters
    ----------
    days : array-like
        A sequence of days corresponding to the evaluation results.
    results : dict
        A dictionary where keys represent different metrics or types, and values are sequences of evaluation results for each metric.
    title : str, optional
        The title of the plot. Default is "Error".
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the results. If None, a new figure and axes are created. Default is None.
    errband : dict or None, optional
        A dictionary where keys match those in `results` and values are sequences of error bands (e.g., standard deviations) corresponding to the evaluation results. If None, no error bands are plotted. Default is None.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if errband is None:
        for k, v in results.items():
            x = np.linspace(days[0], days[-1], len(v))
            ax.plot(x, v, marker="o", markersize=4, label=k)
    else:
        for k in results.keys():
            v = np.array(results[k])
            e = np.array(errband[k])
            x = np.linspace(days[0], days[-1], len(v))
            ax.plot(x, v, marker="o", markersize=4, label=k)
            ax.fill_between(
                x,
                v - e,
                v + e,
                alpha=0.2,
            )
    ax.set_xlabel("Days")
    ax.set_ylabel(title)
    ax.set_title(f"{title} accumulated")
    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
