import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import (
    PowerTransformer,
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from wordcloud import WordCloud, STOPWORDS
from IPython.display import display, HTML


def display_html(size=3, content="content"):
  display(HTML(f"<h{size}>{content}</h{size}>"))


def rotate_xlabels(ax, angle=35):
  ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=angle,
        ha="right"
    )
  

def rotate_ylabels(ax, angle=0):
  ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=angle
  )


# pair plots
def pair_plots(data,
               height=3,
               aspect=1.5,
               hue=None,
               legend=False):
  display_html(2, "Pair Plots")

  pair_grid = sns.PairGrid(
      data=data,
      aspect=aspect,
      height=height,
      hue=hue,
      corner=True
  )
  pair_grid.map_lower(sns.scatterplot)

  if legend:
    pair_grid.add_legend()


# correlation matrix heatmap
def correlation_heatmap(data,
                        figsize=(12, 6),
                        method="spearman",
                        cmap="RdBu"):
  cm = data.corr(method=method, numeric_only=True)

  mask = np.zeros_like(cm, dtype=bool)
  mask[np.triu_indices_from(mask)] = True

  fig, ax = plt.subplots(figsize=figsize)
  hm = sns.heatmap(
      cm,
      vmin=-1,
      vmax=1,
      cmap=cmap,
      center=0,
      annot=True,
      fmt=".2f",
      linewidths=1.5,
      square=True,
      mask=mask,
      ax=ax
  )
  rotate_xlabels(ax)
  rotate_ylabels(ax)
  ax.set(title=f"{method.title()} Correlation Matrix Heatmap")


# gives detailed summary of numeric features
def num_summary(data, var):
  import warnings
  warnings.filterwarnings("ignore")

  # title
  col = data.loc[:, var].copy()
  display_html(size=2, content=var)

  # quick glance
  display_html(3, "Quick Glance:")
  display(col)

  # meta-data
  display_html(3, "Meta-data:")
  print(f"{'Data Type':15}: {col.dtype}")
  print(f"{'Missing Data':15}: {col.isna().sum():,} rows ({col.isna().mean() * 100:.2f} %)")
  print(f"{'Available Data':15}: {col.count():,} / {len(col):,} rows")

  # quantiles
  display_html(3, "Percentiles:")
  display(
      col
      .quantile([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
      .rename(index=lambda val: f"{val * 100:.0f}")
      .rename("value")
      .rename_axis(index="percentile")
      .to_frame()
  )

  # central tendancy
  display_html(3, "Central Tendancy:")
  display(
      pd
      .Series({"mean": col.mean(),
               "trimmed mean (5%)": stats.trim_mean(col.values, 0.05),
               "trimmed mean (10%)": stats.trim_mean(col.values, 0.1),
               "median": col.median()})
      .rename("value")
      .to_frame()
  )

  # spread
  display_html(3, "Measure of Spread:")
  std = col.std()
  iqr = col.quantile(0.75) - col.quantile(0.25)
  display(
      pd
      .Series({
          "var": col.var(),
          "std": std,
          "IQR": iqr,
          "mad": stats.median_abs_deviation(col.dropna()),
          "coef_variance": std / col.mean()
      })
      .rename("value")
      .to_frame()
  )

  # skewness and kurtosis
  display_html(3, "Skewness and Kurtosis:")
  display(
      pd
      .Series({
          "skewness": col.skew(),
          "kurtosis": col.kurtosis()
      })
      .rename("value")
      .to_frame()
  )

  alpha = 0.05
  # test for normality
  display_html(3, "Hypothesis Testing for Normality:")
  # shapiro-wilk test
  display_html(4, "Shapiro-Wilk Test:")
  sw_test = stats.shapiro(col.dropna().values)
  sw_statistic = sw_test.statistic
  sw_pvalue = sw_test.pvalue
  print(f"{'Significance Level':21}: {alpha}")
  print(f"{'Null Hypothesis':21}: The data is normally distributed")
  print(f"{'Alternate Hypothesis':21}: The data is not normally distributed")
  print(f"{'p-value':21}: {sw_pvalue}")
  print(f"{'Test Statistic':21}: {sw_statistic}")
  if sw_pvalue < alpha:
    print(f"- Since p-value is less than alpha ({alpha}), we Reject the Null Hypothesis at {alpha * 100}% significance level")
    print("- CONCLUSION: We conclude that the data sample is not normally distributed")
  else:
    print(f"- Since p-value is greater than alpha ({alpha}), we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
    print("- CONCLUSION: We conclude that the data sample is normally distributed")

  #anderson-darling test
  display_html(4, "Anderson-Darling Test:")
  ad_test = stats.anderson(col.dropna().values, dist="norm")
  ad_statistic = ad_test.statistic
  ad_critical = ad_test.critical_values[2]
  print(f"{'Significance Level':21}: {alpha}")
  print(f"{'Null Hypothesis':21}: The data is normally distributed")
  print(f"{'Alternate Hypothesis':21}: The data is not normally distributed")
  print(f"{'Critical Value':21}: {ad_critical}")
  print(f"{'Test Statistic':21}: {ad_statistic}")
  if ad_statistic >= ad_critical:
    print(f"- Since the Test-statistic is greater than Critical Value, we Reject the Null Hypothesis at {alpha * 100}% significance level")
    print("- CONCLUSION: We conclude that the data sample is not normally distributed")
  else:
    print(f"- Since the Test-statistic is less than Critical Value, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
    print("- CONCLUSION: We conclude that the data sample is normally distributed")


# hypothesis testing for association between 2 numeric variables
def num_num_hyp_testing(data, var1, var2, alpha=0.05):
  display_html(2, f"Hypothesis Test for Association between {var1} and {var2}")

  temp = (
      data
      .dropna(subset=[var1, var2], how="any")
      .copy()
  )

  # pearson test
  pearson = stats.pearsonr(temp[var1].values, temp[var2].values)
  pvalue = pearson.pvalue
  statistic = pearson.statistic
  display_html(3, "Pearson Test")
  print(f"- {'Significance Level':21}: {alpha * 100}%")
  print(f"- {'Null Hypothesis':21}: The samples are uncorrelated")
  print(f"- {'Alternate Hypothesis':21}: The samples are correlated")
  print(f"- {'Test Statistic':21}: {statistic}")
  print(f"- {'p-value':21}: {pvalue}")
  if pvalue < alpha:
    print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {var1} and {var2} are correlated")
  else:
    print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {var1} and {var2} are uncorrelated")

  # spearman test
  spearman = stats.spearmanr(temp[var1].values, temp[var2].values)
  pvalue = spearman.pvalue
  statistic = spearman.statistic
  display_html(3, "Spearman Test")
  print(f"- {'Significance Level':21}: {alpha * 100}%")
  print(f"- {'Null Hypothesis':21}: The samples are uncorrelated")
  print(f"- {'Alternate Hypothesis':21}: The samples are correlated")
  print(f"- {'Test Statistic':21}: {statistic}")
  print(f"- {'p-value':21}: {pvalue}")
  if pvalue < alpha:
    print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {var1} and {var2} are correlated")
  else:
    print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {var1} and {var2} are uncorrelated")


# univariate plots for numeric variables
def num_univar_plots(data, var, bins=10, figsize=(15, 7)):
  display_html(2, f"Univariate Analysis of {var}")
  display_html(content="")
  col = data.loc[:, var].copy()

  fig, axes = plt.subplots(2, 3, figsize=figsize)
  axes = axes.ravel()

  #histogram
  sns.histplot(
      data,
      x=var,
      bins=bins,
      kde=True,
      color="#1973bd",
      ax=axes[0],
  )
  sns.rugplot(
      data,
      x=var,
      color="black",
      height=0.035,
      ax=axes[0]
  )
  axes[0].set(title="Histogram")

  # cdf
  sns.ecdfplot(
      data,
      x=var,
      ax=axes[1],
      color="red"
  )
  axes[1].set(title="CDF")

  # power transform
  data = data.assign(**{
      f"{var}_pwt": (
          PowerTransformer()
          .fit_transform(data.loc[:, [var]])
          .ravel()
      )
  })
  sns.kdeplot(
      data,
      x=f"{var}_pwt",
      fill=True,
      color="#f2b02c",
      ax=axes[2]
  )
  sns.rugplot(
      data,
      x=f"{var}_pwt",
      color="black",
      height=0.035,
      ax=axes[2]
  )
  axes[2].set(title="Power Transformed")

  # box plot
  sns.boxplot(
      data,
      x=var,
      color="#4cd138",
      ax=axes[3]
  )
  axes[3].set(title="Box Plot")

  # violin plot
  sns.violinplot(
      data,
      x=var,
      color="#ed68b4",
      ax=axes[4]
  )
  axes[4].set(title="Violin Plot")

  # qq plot
  sm.qqplot(
      col.dropna(),
      line="45",
      fit=True,
      ax=axes[5]
  )
  axes[5].set(title="QQ Plot")

  plt.tight_layout()
  plt.show()


# cramers-v corrrelation heatmap
def cramers_v(data, var1, var2):
  ct = pd.crosstab(
      data.loc[:, var1],
      data.loc[:, var2]
  )
  r, c = ct.shape
  n = ct.sum().sum()
  chi2 = stats.chi2_contingency(ct).statistic
  phi2 = chi2 / n

  # bias correction
  phi2_ = max(0, phi2 - ((r - 1) * (c - 1) / (n - 1)))
  r_ = r - (((r - 1) ** 2) / (n - 1))
  c_ = c - (((c - 1) ** 2) / (n - 1))

  return np.sqrt(phi2_ / min(r_ - 1, c_ - 1))


def cramersV_heatmap(data, figsize=(12, 6), cmap="Blues"):
  cols = data.select_dtypes(include="O").columns.to_list()

  matrix = (
      pd
      .DataFrame(data=np.ones((len(cols), len(cols))))
      .set_axis(cols, axis=0)
      .set_axis(cols, axis=1)
  )

  for col1 in cols:
    for col2 in cols:
      if col1 != col2:
        matrix.loc[col1, col2] = cramers_v(data, col1, col2)

  mask = np.zeros_like(matrix, dtype=bool)
  mask[np.triu_indices_from(mask)] = True
  
  fig, ax = plt.subplots(figsize=figsize)
  hm = sns.heatmap(
      matrix,
      vmin=0,
      vmax=1,
      cmap=cmap,
      annot=True,
      fmt=".2f",
      square=True,
      linewidths=1.5,
      mask=mask,
      ax=ax
  )
  ax.set(title="Cramer's V Correlation Matrix Heatmap")
  rotate_xlabels(ax)
  rotate_ylabels(ax)


# bivariate plots between 2 numeric variables
def num_bivar_plots(data, var_x, var_y, figsize=(12, 4.5), scatter_kwargs=dict(), hexbin_kwargs=dict()):
  display_html(2, f"Bi-variate Analysis between {var_x} and {var_y}")
  display_html(content="")

  fig, axes = plt.subplots(1, 2, figsize=figsize)

  # scatter plot
  sns.scatterplot(
      data,
      x=var_x,
      y=var_y,
      ax=axes[0],
      edgecolors="black",
      **scatter_kwargs
  )
  axes[0].set(title="Scatter Plot")

  # hexbin plot
  col_x = data.loc[:, var_x]
  col_y = data.loc[:, var_y]
  hexbin = axes[1].hexbin(
      x=col_x,
      y=col_y,
      **hexbin_kwargs
  )
  axes[1].set(
      title="Hexbin Plot",
      xlabel=var_x,
      xlim=(col_x.min(), col_x.max()),
      ylim=(col_y.min(), col_y.max())
  )
  cb = plt.colorbar(
      hexbin,
      label="Count"
  )

  plt.tight_layout()
  plt.show()


# gives detailed summary of categorical features
def cat_summary(data, var):
  import warnings
  warnings.filterwarnings("ignore")

  # title
  col = data.loc[:, var].copy()
  display_html(2, var)

  # quick glance
  display_html(3, "Quick Glance:")
  display(col)

  # meta-data
  display_html(3, "Meta-data:")
  print(f"{'Data Type':15}: {col.dtype}")
  print(f"{'Cardinality':15}: {col.nunique(dropna=True)} categories")
  print(f"{'Missing Data':15}: {col.isna().sum():,} rows ({col.isna().mean() * 100:.2f} %)")
  print(f"{'Available Data':15}: {col.count():,} / {len(col):,} rows")

  # summary
  display_html(3, "Summary:")
  display(
      col
      .describe()
      .rename("")
      .to_frame()
  )

  # categories
  display_html(3, "Categories Distribution:")
  with pd.option_context("display.max_rows", None):
    display(
        col
        .value_counts()
        .pipe(lambda ser: pd.concat(
            [
                ser,
                col.value_counts(normalize=True)
            ],
            axis=1
        ))
        .set_axis(["count", "percentage"], axis=1)
        .rename_axis(index="category")
    )


# hypothesis testing for association between numeric and categorical variable
def num_cat_hyp_testing(data, num_var, cat_var, alpha=0.05):
  display_html(2, f"Hypothesis Test for Association between {num_var} and {cat_var}")

  groups_df = (
      data
      .dropna(subset=[num_var])
      .groupby(cat_var)
  )
  groups = [group[num_var].values for _, group in groups_df]

  # anova test
  anova = stats.f_oneway(*groups)
  statistic = anova[0]
  pvalue = anova[1]
  display_html(3, "ANOVA Test")
  print(f"- {'Significance Level':21}: {alpha * 100}%")
  print(f"- {'Null Hypothesis':21}: The groups have similar population mean")
  print(f"- {'Alternate Hypothesis':21}: The groups don't have similar population mean")
  print(f"- {'Test Statistic':21}: {statistic}")
  print(f"- {'p-value':21}: {pvalue}")
  if pvalue < alpha:
    print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {num_var} and {cat_var} are associated to each other")
  else:
    print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {num_var} and {cat_var} are not associated to each other")

  # kruskal-wallis test
  kruskal = stats.kruskal(*groups)
  statistic = kruskal[0]
  pvalue = kruskal[1]
  display_html(3, "Kruskal-Wallis Test")
  print(f"- {'Significance Level':21}: {alpha * 100}%")
  print(f"- {'Null Hypothesis':21}: The groups have similar population median")
  print(f"- {'Alternate Hypothesis':21}: The groups don't have similar population median")
  print(f"- {'Test Statistic':21}: {statistic}")
  print(f"- {'p-value':21}: {pvalue}")
  if pvalue < alpha:
    print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {num_var} and {cat_var} are associated to each other")
  else:
    print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {num_var} and {cat_var} are not associated to each other")

  
#univariate plots for categorical variables
#helper functions
def get_top_k(data, var, k):
  col = data.loc[:, var].copy()
  cardinality = col.nunique(dropna=True)
  if k >= cardinality:
    raise ValueError(f"Cardinality of {var} is {cardinality}. K must be less than {cardinality}.")
  else:
    top_categories = (
        col
        .value_counts(dropna=True)
        .index[:k]
    )
    data = data.assign(**{
        var: np.where(
            col.isin(top_categories),
            col,
            "Other"
        )
    })
    return data
  

def pie_chart(counts, colors, ax):
  pie = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%.2f%%",
        colors=colors,
        wedgeprops=dict(alpha=0.7, edgecolor="black"),
  )

  ax.set_title("Pie Chart")

  ax.legend(
      loc="upper left",
      bbox_to_anchor=(1.02, 1),
      title="Categories",
      title_fontproperties=dict(weight="bold", size=10)
  )

  plt.setp(
      pie[2],
      weight="bold",
      color="white"
  )


def bar_chart(counts, colors, ax):
  barplot = ax.bar(
        x=range(len(counts)),
        height=counts.values,
        tick_label=counts.index,
        color=colors,
        edgecolor="black",
        alpha=0.7
  )

  ax.bar_label(
      barplot,
      padding=5,
      color="black"
  )

  ax.set(
      title="Bar Chart",
      xlabel="Categories",
      ylabel="Count"
  )

  ax.set_xticklabels(
      ax.get_xticklabels(),
      rotation=45,
      ha="right"
  )


def cat_univar_plots(data,
                     var,
                     k=None,
                     order=None,
                     show_wordcloud=True,
                     figsize=(12, 8.5)):
  display_html(2, f"Univariate Analysis of {var}")
  display_html(content="")

  fig = plt.figure(figsize=figsize)
  gs = GridSpec(2, 2, figure=fig)
  ax1 = fig.add_subplot(gs[0, 0]) # bar-chart
  ax2 = fig.add_subplot(gs[0, 1]) # pie-chart
  ax3 = fig.add_subplot(gs[1, :]) # word-cloud

  if k is None:
    counts = (
        data
        .loc[:, var]
        .value_counts()
        .reindex(index=order)
    )
  else:
    temp = get_top_k(
        data,
        var,
        k=k
    )
    counts = (
        temp
        .loc[:, var]
        .value_counts()
    )

  colors = [tuple(np.random.choice(256, size=3) / 255) for _ in range(len(counts))]

  # bar-chart
  bar_chart(
      counts,
      colors,
      ax1
  )

  # pie_chart
  pie_chart(
      counts,
      colors,
      ax2
  )

  # word-cloud
  if show_wordcloud:
    var_string = " ".join(
        data
        .loc[:, var]
        .dropna()
        .str.replace(" ", "_")
        .to_list()
    )

    word_cloud = WordCloud(
        width=2000,
        height=700,
        random_state=42,
        background_color="black",
        colormap="Set2",
        stopwords=STOPWORDS
    ).generate(var_string)

    ax3.imshow(word_cloud)
    ax3.axis("off")
    ax3.set_title("Word Cloud")
  else:
    ax3.remove()

  plt.tight_layout()
  plt.show()


# bivariate plots between numeric and categorical variable 
def num_cat_bivar_plots(data,
                        num_var,
                        cat_var,
                        k=None,
                        estimator="mean",
                        orient="v",
                        order=None,
                        figsize=(15, 4)):

  def get_values(data,
                 num_var,
                 cat_var,
                 estimator,
                 order=None):
    return (
        data
        .groupby(cat_var)
        .agg(estimator, numeric_only=True)
        .loc[:, num_var]
        .dropna()
        .sort_values()
        .reindex(index=order)
    )

  import warnings
  warnings.filterwarnings("ignore")

  display_html(2, f"Bi-variate Analysis between {cat_var} and {num_var}")
  display_html(content="")

  if k is None:
    temp = get_values(
        data,
        num_var,
        cat_var,
        estimator,
        order=order
    )
  else:
    data = get_top_k(
        data,
        cat_var,
        k=k
    )
    temp = get_values(
        data,
        num_var,
        cat_var,
        estimator
    )

  if orient == "v":
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # bar plot
    bar = sns.barplot(
      x=temp.index,
      y=temp.values,
      color="#d92b2b",
      ax=axes[0],
      edgecolor="black",
      alpha=0.5
    )
    axes[0].set(
        title="Bar Plot",
        xlabel=cat_var,
        ylabel=num_var
    )
    rotate_xlabels(axes[0])

    # box plot
    sns.boxplot(
      data,
      x=cat_var,
      y=num_var,
      color="lightgreen",
      order=temp.index,
      ax=axes[1]
    )
    axes[1].set(
        title="Box Plot",
        xlabel=cat_var,
        ylabel=""
    )
    rotate_xlabels(axes[1])

    # violin plot
    sns.violinplot(
      data,
      x=cat_var,
      y=num_var,
      color="#0630c9",
      order=temp.index,
      ax=axes[2],
      alpha=0.5
    )
    axes[2].set(
        title="Violin Plot",
        xlabel=cat_var,
        ylabel=""
    )
    rotate_xlabels(axes[2])
  else:
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # bar plot
    bar = sns.barplot(
      y=temp.index,
      x=temp.values,
      color="#d92b2b",
      ax=axes[0],
      edgecolor="black",
      alpha=0.5
    )
    axes[0].set(
        title="Bar Plot",
        xlabel="",
        ylabel=cat_var
    )

    # box plot
    sns.boxplot(
      data,
      y=cat_var,
      x=num_var,
      color="lightgreen",
      order=temp.index,
      ax=axes[1]
    )
    axes[1].set(
        title="Box Plot",
        xlabel="",
        ylabel=cat_var
    )

    # violin plot
    sns.violinplot(
      data,
      y=cat_var,
      x=num_var,
      color="#0630c9",
      order=temp.index,
      ax=axes[2],
      alpha=0.5
    )
    axes[2].set(
        title="Violin Plot",
        xlabel=num_var,
        ylabel=cat_var
    )

  plt.tight_layout()
  plt.show()


# categorical bivariate plots
def cat_heat_map(data, mask=True, **kwargs):
  if mask:
    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
  else:
    mask = None

  return sns.heatmap(
      data=data,
      mask=mask,
      annot=True,
      linewidths=1.5,
      linecolor="white",
      square=True,
      **kwargs
  )


# bivariate analysis between 2 categorical variables
def cat_bivar_plots(data,
                    var1,
                    var2,
                    k1=None,
                    k2=None,
                    order1=None,
                    order2=None,
                    figsize=(12, 8.5)):

  import warnings
  warnings.filterwarnings("ignore")

  display_html(2, f"Bi-variate Analysis between {var1} and {var2}")
  display_html(content="")

  if k1 is not None:
    data = get_top_k(
        data,
        var1,
        k=k1
    )

  if k2 is not None:
    data = get_top_k(
        data,
        var2,
        k=k2
    )

  fig, axes = plt.subplots(2, 2, figsize=figsize)
  axes = axes.ravel()

  # cross-tab heatmap
  ct = (
      pd
      .crosstab(
          index=data.loc[:, var1],
          columns=data.loc[:, var2]
      )
      .reindex(
          index=order1,
          columns=order2
      )
  )
  hm = cat_heat_map(
      ct,
      mask=False,
      vmin=ct.values.min(),
      vmax=ct.values.max(),
      fmt="d",
      cmap="Blues",
      cbar_kws=dict(location="top", label="Counts"),
      ax=axes[0]
  )
  rotate_ylabels(axes[0])
  rotate_xlabels(axes[0])

  # normalized cross-tab heatmap
  norm_ct = (
      pd
      .crosstab(
          index=data.loc[:, var1],
          columns=data.loc[:, var2],
          normalize="index"
      )
      .reindex(
          index=order1,
          columns=order2
      )
  )
  norm_hm = cat_heat_map(
      norm_ct,
      mask=False,
      vmin=0,
      vmax=1,
      fmt=".2f",
      cmap="Greens",
      cbar_kws=dict(location="top", label="Normalized"),
      ax=axes[1]
  )
  axes[1].set(ylabel="")
  rotate_ylabels(axes[1])
  rotate_xlabels(axes[1])

  # bar plot
  (
      ct
      .plot
      .bar(
          ax=axes[2],
          title="Bar Plot",
          legend=False
      )
  )
  rotate_xlabels(axes[2])

  # stacked bar plot
  (
      norm_ct
      .plot
      .bar(
          ax=axes[3],
          title="Stacked Bar Plot",
          stacked=True
      )
  )
  rotate_xlabels(axes[3])
  axes[3].legend(
      loc="upper left",
      bbox_to_anchor=(1, 1),
      title=var2
  )

  plt.tight_layout()
  plt.show()


# hypothesis testing between 2 categorical variables
def hyp_cat_cat(data, var1, var2, alpha=0.05):
  display_html(2, f"Hypothesis Test for Association between {var1} and {var2}")

  ct = pd.crosstab(
      data.loc[:, var1],
      data.loc[:, var2]
  )

  display_html(3, "Chi-square Test")
  chi2 = stats.chi2_contingency(ct)
  statistic = chi2.statistic
  pvalue = chi2.pvalue
  print(f"- {'Cramers V':21}: {cramers_v(data, var1, var2)}")
  print(f"- {'Significance Level':21}: {alpha * 100}%")
  print(f"- {'Null Hypothesis':21}: The samples are uncorrelated")
  print(f"- {'Alternate Hypothesis':21}: The samples are correlated")
  print(f"- {'Test Statistic':21}: {statistic}")
  print(f"- {'p-value':21}: {pvalue}")
  if pvalue < alpha:
    print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {var1} and {var2} are correlated")
  else:
    print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
    print(f"- CONCLUSION: The variables {var1} and {var2} are uncorrelated")

# missing values
def missing_info(data):
  na_cols = [col for col in data.columns if data[col].isna().any()]
  na_counts = [data[col].isna().sum() for col in na_cols]
  na_pct = [(data[col].isna().mean() * 100) for col in na_cols]

  return (
      pd
      .DataFrame(data={
          "variable": na_cols,
          "count": na_counts,
          "percentage": na_pct
      })
      .sort_values(by="count", ascending=False)
      .set_index("variable")
  )


def plot_missing_info(data, bar_label_params=dict(), figsize=(10, 4)):
  na_data = missing_info(data)
  fig, ax = plt.subplots(1, 1, figsize=figsize)
  bar = ax.bar(
      range(len(na_data)),
      height=na_data["count"].values,
      color="#1eba47",
      edgecolor="black",
      tick_label=na_data.index.to_list(),
      alpha=0.7
  )
  ax.bar_label(
      bar,
      **bar_label_params
  )
  ax.set(
      xlabel="Variable",
      ylabel="Count",
      title="Missing Data Counts per Variable"
  )
  rotate_xlabels(ax)
  plt.tight_layout()
  plt.show()


# iqr outliers
def get_iqr_outliers(data, var, band=1.5):
    q1, q3 = (
      data
      .loc[:, var]
      .quantile([0.25, 0.75])
      .values
    )

    iqr = q3 - q1
    lower_limit = q1 - (band * iqr)
    upper_limit = q3 + (band * iqr)

    display_html(3, f"{var} - IQR Limits:")
    print(f"{'Lower Limit':12}: {lower_limit}")
    print(f"{'Upper Limit':12}: {upper_limit}")

    return (
      data
      .query(f"{var} > @upper_limit | {var} < @lower_limit")
      .sort_values(var)
    )


# univariate plots for date-time variables
def dt_univar_plots(data, var, target=None, bins="auto"):
  display_html(3, f"Univariate plots of {var}")
  col = data.loc[:, var].copy()

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

  # histogram
  sns.histplot(
    data=data,
    x=var,
    bins=bins,
    color="#1973bd",
    ax=ax1
  )
  sns.rugplot(
    data=data,
    x=var,
    color="darkblue",
    height=0.035,
    ax=ax1
  )
  ax1.set(title="Histogram")
  rotate_xlabels(ax1)

  # line-plot
  sns.lineplot(
    data=data,
    x=var,
    y=target,
    color="#d92b2b",
    ax=ax2
  )
  rotate_xlabels(ax2)
  ax2.set(title="Line Plot")