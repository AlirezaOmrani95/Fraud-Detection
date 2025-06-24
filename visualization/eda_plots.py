"""
File: eda_plots.py

Author: Ali Reza (ARO) Omrani
Email: omrani.alireza95@gmail.com
Date: 4th March 2025

Description:
------
This file contains a collection of functions to create various plots for Exploratory Data Analysis (EDA).
These plots are designed to help analyze and understand the structure, distribution, and relationships in a dataset.

Functions:
------
- histogram(data, column, hist_bin, xlabel, ylabel): Plots the distribution of a single variable.
- box_plot(data, column, sorted_by, xlabel, ylabel): Visualize a box plot to show the distributions and detect outliers.
- scatter_plot(data, x, y, xlabel, ylabel): Plots a scatter plot for two continues variables.
- heatmaps(data): Display the correlation between variables.
- pair_plot(data): Generate pairwise plots to show relationships between variables.
- bar_plot(data, column): Displays a bar plot to compare categories based on their values.
- line_plot(data, x, y, xlabel, ylabel): Shows trends over time or continuous variables with a line plot.
- violet_plot(data, x, y, xlabel, ylabel): Generates a violet plot to show the distribution of data.
- count_plot(data, x, xlabel): Visualizes the frequency of categories in a categorical variable.
- piechart(data, column, label): creates a pie chart to visualize the proportions of categories.

Requirements:
------
- matplotlib
- seaborn

Usage Example:
------
>>> from visialization.eda_plots import histogram, pair_plot
>>> histogram(data,column_name,hist_bin)
>>> pair_plot(data)

Notes:
------
- Functions are designed to handle pandas Dataframes.
- Some visualizations may take time for larger datasets.
"""

from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def histogram(
    data: pd.DataFrame,
    column: str,
    hist_bin: int,
    xlabel: str = "values",
    ylabel: str = "frequency",
) -> None:
    """
    Display the histogram of a specific column.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.
        column (str): target column to display.
        hist_bin (int): The number of bins to display histograms
        xlabel (str): x_axis label
        ylabel (str): y_axis label

    Returns:
    -------
        None

    Example:
    -------
        >>> histogram(data, 'category', 30)
    """

    col_name = column
    data[col_name].hist(bins=hist_bin)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(f"histogram of {col_name}")
    plt.show()


def box_plot(
    data: pd.DataFrame,
    column: str,
    sorted_by: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Display the box plot of a specific column sorted by another column.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.
        column (str): target column to display.
        sorted_by (str): based on which feature sort the column.
        xlabel (Optional[str]): x_axis label, default value is None.
        ylabel (Optional[str]): y_axis label, default value is None.

    Returns:
    -------
        None

    Example:
    -------
        >>> box_plot(data,'category',30)

    Notes:
    ------
        - If don't set anything for xlabel and ylabel, column and sorted_by value are replaced with them repectively.
    """

    data.boxplot(column=column, by=sorted_by)
    plt.xlabel(sorted_by if xlabel == None else xlabel)
    plt.ylabel(column if ylabel == None else ylabel)
    plt.title(
        f"scatter plot of {sorted_by if xlabel == None else xlabel} vs {column if ylabel == None else ylabel}"
    )
    plt.show()


def scatter_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Display the scatter plot based on two columns.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.
        x (str): The column name to be used as horizontal coordinates for each point.
        y (str): The column name to be used as vertical coordinates for each point.
        xlabel (Optional[str]): x_axis label, default value is None.
        ylabel (Optional[str]): y_axis label, default value is None.

    Returns:
    -------
        None

    Example:
    -------
        >>> scatter_plot(data, 'amt', 'is_fraud')

    Notes:
    ------
        - If don't set anything for xlabel and ylabel, x and y value are replaced with them repectively.
    """

    data.plot.scatter(x=x, y=y)
    plt.xlabel(x if xlabel == None else xlabel)
    plt.ylabel(y if ylabel == None else ylabel)
    plt.title(
        f"scatter plot {x if xlabel == None else xlabel} vs {y if ylabel == None else ylabel}"
    )
    plt.show()


def heatmaps(data: pd.DataFrame) -> None:
    """
    Display Heatmaps of the dataframe.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.

    Returns:
    -------
        None
    """

    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("correlation heatmap")
    plt.show()


def pair_plot(data: pd.DataFrame) -> None:
    """
    Plot pairwise relationships in a dataset.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.

    Returns:
    -------
        None
    """

    sns.pairplot(data)
    plt.title("pairplot of dataframe")
    plt.show()


def bar_plot(data: pd.DataFrame, column: str) -> None:
    """
    Display bar plot for a column.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.
        column (str): The categorical column name.

    Returns:
    --------
        None

    Example:
    -------
        >>> bar_plot(data, 'category')
    """

    data[column].value_counts().plot.bar()
    plt.xlabel(column)
    plt.ylabel("count")
    plt.title(f"barplot of {column}")
    plt.show()


def line_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Plot line plot from DataFrame.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.
        x (str): The column label for the x-axis.
        y (str): The column label for the y-axis.
        xlabel (Optional[str]): x_axis label, default value is None.
        ylabel (Optional[str]): y_axis label, default value is None

    Returns:
    -------
        None

    Example:
    -------
        >>> line_plot(data,'trans_date_trans_time', 'amt')

    Notes:
    ------
        - If don't set anything for xlabel and ylabel, x and y value are replaced with them repectively.
    """

    data.plot.line(x, y)
    plt.xlabel(x if xlabel == None else xlabel)
    plt.ylabel(y if ylabel == None else ylabel)
    plt.title(f"Line Plot of Value over Time")
    plt.show()


def violet_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Plot violine plot from DataFrame.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.
        x (str): The column label for the x-axis.
        y (str): The column label for the y-axis.
        xlabel (Optional[str]): x_axis label, default value is None.
        ylabel (Optional[str]): y_axis label, default value is None

    Returns:
    -------
        None

    Example:
    -------
        >>> violet_plot(data, 'category','amt')

    Notes:
    ------
        - If don't set anything for xlabel and ylabel, x and y value are replaced with them repectively.
    """

    sns.violinplot(x, y, data=data)
    plt.xlabel(x if xlabel == None else xlabel)
    plt.ylabel(y if ylabel == None else ylabel)
    plt.title("violin plot")
    plt.show()


def count_plot(data: pd.DataFrame, x: str, xlabel: Optional[str] = None) -> None:
    """
    Plots the counts of observations in each categorical bin using bars.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.
        x (str): The column label for the x-axis.
        xlabel (Optional[str]): x_axis label, default value is None.

    Returns:
    --------
        None

    Example:
    -------
        >>> count_plot(data,'category')

    Notes:
    ------
        - If don't set anything for xlabel, x value is replaced with them repectively.
    """

    sns.countplot(x, data=data)
    plt.xlabel(x if xlabel == None else xlabel)
    plt.ylabel("count")
    plt.title("count plot")
    plt.show()


def piechart(data: pd.DataFrame, column: str, label: Optional[str] = None) -> None:
    """
    Plots the counts of observations in each categorical bin using bars.

    Parameters:
    ----------
        data (pd.Dataframe): The pandas dataframe containing the data.
        column (str): The categorical column label.
        label (Optional[str]): label, default value is None.

    Returns:
    --------
        None

    Example:
    -------
        >>> count_plot(data,'category')

    Notes:
    ------
       - If don't set anything for label, '' is replaced with them repectively.
    """

    data[column].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("pie chart")
    plt.ylabel("" if label == None else label)
    plt.show()
