import sys
import os

# Load your CSV file: use first CLI arg if present (verify it exists), otherwise default to "table.csv"
infile = sys.argv[1] if len(sys.argv) > 1 else 'table.csv'
if not os.path.exists(infile):
    print(f"Error: input file '{infile}' not found.", file=sys.stderr)
    sys.exit(1)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# support CSV and Excel inputs
if infile.lower().endswith(('.xlsx', '.xls')):
    df = pd.read_excel(infile)
else:
    df = pd.read_csv(infile)

# Clean and convert columns
monthly_cols = [col for col in df.columns if 'Pushes' in col and '-' in col]
for col in monthly_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert date columns
for col in ['FirstVisit', 'LatestVisit']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Compute total pushes and visits per user
df['TotalPushes'] = df[monthly_cols].sum(axis=1, skipna=True)
df['TotalVisits'] = pd.to_numeric(df['Visits'], errors='coerce')

# Segment users by median push count
median_pushes = df['TotalPushes'].median()
df['Segment'] = df['TotalPushes'].apply(lambda x: 'High' if x > median_pushes else 'Low')

# Regional stats
region_avg = df.groupby('Locale')['TotalPushes'].mean()
user_count_by_region = df['Locale'].value_counts()
# canonical region order (used by multiple plots)
region_order = region_avg.sort_values(ascending=False).index

# Monthly average pushes
monthly_avg = df[monthly_cols].mean()

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
ax = axes.flatten()

# 1. Segmentation histogram
sns.histplot(data=df, x='TotalPushes', hue='Segment', multiple='stack', ax=ax[0])
median_label = f"Median = {int(median_pushes)}" if pd.notnull(median_pushes) else "Median = NA"
ax[0].axvline(median_pushes, color='red', linestyle='--', label=median_label)
ax[0].set_title('User Segmentation by Total Pushes')
ax[0].legend()

# 2. Monthly usage trends
# plot per-region monthly averages (lighter lines) and the overall average (bold)
region_monthly = df.groupby('Locale')[monthly_cols].mean().fillna(0)
region_order_local = region_avg.sort_values(ascending=False).index
palette = sns.color_palette('tab10', n_colors=max(10, len(region_order_local)))
for i, region in enumerate(region_order_local):
    if region in region_monthly.index:
        ax[1].plot(monthly_cols, region_monthly.loc[region, monthly_cols].values,
                   label=region, color=palette[i % len(palette)], alpha=0.6)
# overall average (bold)
ax[1].plot(monthly_cols, monthly_avg.values, marker='o', color='k', linewidth=3, label='Overall')
ax[1].set_title('Average Monthly Pushes')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Average Pushes')
# shorten month labels (remove ' Pushes')
month_labels = [m.replace(' Pushes', '') for m in monthly_cols]
plt.sca(ax[1])
plt.xticks(range(len(month_labels)), month_labels, rotation=45)
ax[1].grid(axis='y', linestyle='--', alpha=0.4)
ax[1].tick_params(axis='x', rotation=45)
ax[1].legend(fontsize='small', ncol=2, loc='upper left')

# 3. Median monthly pushes by region (heatmap)
region_month_med = df.groupby('Locale')[monthly_cols].median().reindex(region_order).fillna(0)
sns.heatmap(region_month_med, cmap='YlGnBu', ax=ax[2], cbar_kws={'label': 'Median pushes'})
ax[2].set_title('Median monthly pushes by Region')
ax[2].set_ylabel('Locale')
# shorten month labels (remove ' Pushes')
month_labels = [c.replace(' Pushes', '') for c in region_month_med.columns]
ax[2].set_xticks(np.arange(len(month_labels)))
ax[2].set_xticklabels(month_labels, rotation=45)
ax[2].set_xlabel('Month')

# 4. Active users per month (stacked area)
active_counts = df.groupby('Locale')[monthly_cols].apply(lambda x: (x > 0).sum()).reindex(region_order).fillna(0).astype(int)
active_by_month = active_counts.T
# cleaned month labels (remove suffix like ' Pushes')
month_labels = [m.replace(' Pushes', '') for m in active_by_month.index]

colors = sns.color_palette('tab10', n_colors=max(10, len(region_order)))[:len(region_order)]
active_by_month.plot.area(ax=ax[3], stacked=True, color=colors)
ax[3].set_title('Active users per month by Region (stacked)')
ax[3].set_xlabel('Month')
ax[3].set_ylabel('Number of active users')
ax[3].set_xticks(range(len(month_labels)))
ax[3].set_xticklabels(month_labels, rotation=45)
ax[3].grid(axis='y', linestyle='--', alpha=0.4)
ax[3].legend(title='Locale', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

# 5. Recency analysis
if 'DaysSince\nLatestVisit' in df.columns:
    x = pd.to_numeric(df['DaysSince\nLatestVisit'], errors='coerce')
    y = df['TotalPushes']
    sns.scatterplot(x=x, y=y, ax=ax[4])
    ax[4].set_title('Recency vs Total Pushes')
    ax[4].set_xlabel('Days Since Latest Visit')
    ax[4].set_ylabel('Total Pushes')

# 7. Early engagement (monthly figures)
import math

n = len(monthly_cols)
cols = min(4, n)
rows = math.ceil(n / cols)

fig2, axs2 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
if hasattr(axs2, 'flatten'):
    axs_flat = axs2.flatten()
else:
    axs_flat = [axs2]

for i, col in enumerate(monthly_cols):
    axm = axs_flat[i]
    data = pd.to_numeric(df[col], errors='coerce').dropna()

    if data.empty:
        axm.text(0.5, 0.5, 'No data', ha='center', va='center')
        axm.set_title(col)
        axm.set_xlabel('Pushes')
        axm.set_ylabel('Number of users')
        continue

    sns.histplot(data, bins=30, kde=True, ax=axm, color='C0')
    median_val = data.median()
    axm.axvline(median_val, color='red', linestyle='--', label=f'Median = {median_val:.0f}')
    axm.set_title(col)
    axm.set_xlabel('Pushes')
    axm.set_ylabel('Number of users')
    axm.set_xlim(left=0)
    axm.legend(fontsize='small')

    q = data.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    axm.text(
        0.98, 0.98,
        f"25/50/75/90/99: {int(q.loc[0.25])}/{int(q.loc[0.5])}/{int(q.loc[0.75])}/{int(q.loc[0.9])}/{int(q.loc[0.99])}",
        transform=axm.transAxes,
        fontsize=8, ha='right', va='top',
        bbox=dict(boxstyle='round', fc='white', alpha=0.7)
    )

# turn off any extra axes
for j in range(n, len(axs_flat)):
    try:
        axs_flat[j].axis('off')
    except Exception:
        pass

plt.tight_layout()
fig2.savefig("early_engagement_by_month.png")
plt.close(fig2)

# bubble plot in dashboard (move here so it's included in user_profile_dashboard.png)
region_median = df.groupby('Locale')['TotalPushes'].median().reindex(region_order).fillna(0)
region_total = df.groupby('Locale')['TotalPushes'].sum().reindex(region_order).fillna(0)
region_counts = user_count_by_region.reindex(region_order).fillna(0)

# scale bubble sizes for visibility
max_total = max(1, region_total.values.max())
sizes = (region_total.values / max_total) * 1500

ax[5].scatter(region_counts.values, region_median.values, s=sizes, alpha=0.7, color='C2', edgecolor='k', linewidth=0.5)
for i, loc in enumerate(region_order):
    ax[5].annotate(loc, (region_counts.values[i], region_median.values[i]), fontsize=8, xytext=(4, 4), textcoords='offset points')
ax[5].set_xlabel('User count')
ax[5].set_ylabel('Median TotalPushes')
ax[5].set_title('Median pushes vs user count (bubble size = total pushes)')
ax[5].grid(alpha=0.2)

# save dashboard now that `fig` is complete
fig.tight_layout()
fig.savefig("user_profile_dashboard.png")

# Additional regional diagnostics: boxplot, heatmap, power-user share, median vs count
import math
fig3, axs3 = plt.subplots(2, 2, figsize=(16, 10))
axs = axs3.flatten()

# consistent region order
region_order = region_avg.sort_values(ascending=False).index

# 1) Boxplot of TotalPushes by region (robust view)
sns.boxplot(x='Locale', y='TotalPushes', data=df[df['Locale'].isin(region_order)], order=region_order, ax=axs[0])
axs[0].set_title('TotalPushes distribution by Region')
axs[0].tick_params(axis='x', rotation=45)

# 2) Average pushes by region (show mean ± SE)
region_means = region_avg.loc[region_order]
region_counts_local = user_count_by_region.reindex(region_order).fillna(0).astype(int)
region_std_local = df.groupby('Locale')['TotalPushes'].std().reindex(region_order)
# use global std to avoid zero/NaN for tiny groups
global_std_local = df['TotalPushes'].std(ddof=1)
region_std_local = region_std_local.fillna(global_std_local)

se_local = region_std_local / np.sqrt(region_counts_local.replace(0, np.nan))
se_local = se_local.fillna(global_std_local)

x = np.arange(len(region_order))
width = 0.6
axs[1].bar(x, region_means.values, width, color='C0', yerr=se_local.values, capsize=5)
axs[1].set_title('Average Pushes by Region (mean ± SE)')
axs[1].set_xticks(x)
axs[1].set_xticklabels(region_order, rotation=45)
axs[1].tick_params(axis='x', rotation=45)
axs[1].set_ylabel('Average Pushes')

# 3) Concentration metrics: Gini coefficient and Top‑10% share by region (with bootstrap 95% CIs)

def gini_coef(s):
    a = np.asarray(s, dtype=np.float64) if not isinstance(s, np.ndarray) else s.astype(np.float64)
    # accept pandas Series or numpy array
    if hasattr(s, 'dropna'):
        a = np.asarray(s.dropna(), dtype=np.float64)
    else:
        a = a[~np.isnan(a)]
    if a.size == 0 or a.sum() == 0:
        return 0.0
    a = np.sort(a)
    n = a.size
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * a) / (n * a.sum())) - (n + 1.0) / n


def top_share_pct(s, pct=0.10):
    # accepts Series or array; returns percent
    if hasattr(s, 'dropna'):
        arr = s.dropna().values
    else:
        arr = np.asarray(s)
        arr = arr[~np.isnan(arr)]
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    cutoff = max(1, math.ceil(arr.size * pct))
    return np.sort(arr)[-cutoff:].sum() / arr.sum() * 100

# point estimates for reference
gini_series = df.groupby('Locale')['TotalPushes'].apply(lambda s: gini_coef(s)).reindex(region_order).fillna(0)
top10_series_pt = df.groupby('Locale')['TotalPushes'].apply(lambda s: top_share_pct(s, pct=0.10)).reindex(region_order).fillna(0)

# bootstrap settings
n_boot = 1000
alpha = 0.05
seed = 123
rng = np.random.default_rng(seed)

# containers for bootstrap results
gini_mean = []
gini_lo = []
gini_hi = []
top10_mean = []
top10_lo = []
top10_hi = []

for region in region_order:
    arr = df.loc[df['Locale'] == region, 'TotalPushes'].dropna().values
    if arr.size == 0:
        # no data
        gini_mean.append(np.nan); gini_lo.append(np.nan); gini_hi.append(np.nan)
        top10_mean.append(np.nan); top10_lo.append(np.nan); top10_hi.append(np.nan)
        continue
    if arr.size == 1:
        # single-user region: use point estimates and zero-width CI so the region remains visible
        gpt = gini_coef(arr)
        tpt = top_share_pct(arr, pct=0.10)
        gini_mean.append(gpt); gini_lo.append(gpt); gini_hi.append(gpt)
        top10_mean.append(tpt); top10_lo.append(tpt); top10_hi.append(tpt)
        continue

    boots_g = np.empty(n_boot)
    boots_t = np.empty(n_boot)
    n = arr.size
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots_g[i] = gini_coef(sample)
        boots_t[i] = top_share_pct(sample, pct=0.10)

    gini_mean.append(boots_g.mean())
    gini_lo.append(np.percentile(boots_g, 100 * (alpha / 2)))
    gini_hi.append(np.percentile(boots_g, 100 * (1 - alpha / 2)))

    top10_mean.append(boots_t.mean())
    top10_lo.append(np.percentile(boots_t, 100 * (alpha / 2)))
    top10_hi.append(np.percentile(boots_t, 100 * (1 - alpha / 2)))

# convert to arrays
gini_mean = np.array(gini_mean)
gini_lo = np.array(gini_lo)
gini_hi = np.array(gini_hi)

top10_mean = np.array(top10_mean)
top10_lo = np.array(top10_lo)
top10_hi = np.array(top10_hi)

# clip CIs to valid ranges: Gini in [0,1], Top10 lower bound >= 0
gini_lo = np.maximum(gini_lo, 0.0)
gini_hi = np.minimum(gini_hi, 1.0)
top10_lo = np.maximum(top10_lo, 0.0)

# convert Gini to percent for plotting
gini_mean_pct = gini_mean * 100
gini_err_lower = (gini_mean - gini_lo) * 100
gini_err_upper = (gini_hi - gini_mean) * 100

top10_err_lower = top10_mean - top10_lo
top10_err_upper = top10_hi - top10_mean

# plot side-by-side bars with bootstrap 95% CI
x = np.arange(len(region_order))
width = 0.35
bars_top10 = axs[2].bar(x - width/2, top10_mean, width,
                        yerr=[top10_err_lower, top10_err_upper], capsize=4,
                        color='C2', label='Top 10% (bootstrap mean ±95% CI)')
bars_gini = axs[2].bar(x + width/2, gini_mean_pct, width,
                       yerr=[gini_err_lower, gini_err_upper], capsize=4,
                       color='C4', label='Gini ×100 (bootstrap mean ±95% CI)')

axs[2].set_title('Top 10% share and Gini by Region (bootstrap 95% CI)')
axs[2].set_xticks(x)
axs[2].set_xticklabels(region_order, rotation=45)
axs[2].set_ylabel('Percent')
axs[2].legend(fontsize='small')

# annotate point estimates
for i, (b, val) in enumerate(zip(bars_top10, top10_mean)):
    if np.isnan(val):
        continue
    axs[2].annotate(f"{val:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')
for i, (b, val) in enumerate(zip(bars_gini, gini_mean_pct)):
    if np.isnan(val):
        continue
    axs[2].annotate(f"{val:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')

axs[2].text(0.01, 0.02, f'Bootstrap n={n_boot}, 95% CI; small n → wide/unreliable intervals', transform=axs[2].transAxes, fontsize=7, va='bottom')

# 4) User count and Total Pushes by Region (normalized)
counts_local = user_count_by_region.reindex(region_order).fillna(0).astype(int)
region_total_local = df.groupby('Locale')['TotalPushes'].sum().reindex(region_order).fillna(0).astype(int)

max_count_local = counts_local.max() if counts_local.max() > 0 else 1
max_total_local = region_total_local.max() if region_total_local.max() > 0 else 1
norm_counts_local = counts_local / max_count_local
norm_total_local = region_total_local / max_total_local

x = np.arange(len(region_order))
width = 0.35
bars_counts_local = axs[3].bar(x - width/2, norm_counts_local.values, width, color='C0', label='User count (normalized)')
bars_total_local = axs[3].bar(x + width/2, norm_total_local.values, width, color='C1', label='Total pushes (normalized)')

axs[3].set_xticks(x)
axs[3].set_xticklabels(region_order, rotation=45)
axs[3].set_title('User count and Total Pushes by Region (normalized)')
axs[3].set_ylabel('Normalized (0-1)')
axs[3].legend(fontsize='small')

# annotate actual values above each bar for clarity
for b, val in zip(bars_counts_local, counts_local.values):
    axs[3].annotate(f"{int(val)}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points', zorder=9)
for b, val in zip(bars_total_local, region_total_local.values):
    axs[3].annotate(f"{int(val)}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points', zorder=9)

plt.tight_layout()
fig3.savefig('region_analysis_extra.png')
plt.close(fig3)

# Overlaid Lorenz curves: top/bottom regions by Gini
try:
    gini_series_local = df.groupby('Locale')['TotalPushes'].apply(lambda s: gini_coef(s)).reindex(region_order).fillna(0)
except Exception:
    gini_series_local = df.groupby('Locale')['TotalPushes'].apply(lambda s: gini_coef(s if hasattr(s,'dropna') else pd.Series(s))).reindex(region_order).fillna(0)

top_n = 3
top_regions = gini_series_local.sort_values(ascending=False).head(top_n).index.tolist()
bottom_regions = gini_series_local.sort_values(ascending=True).head(top_n).index.tolist()

selected_regions = []
for r in top_regions + bottom_regions:
    if r not in selected_regions:
        selected_regions.append(r)

fig5, ax5 = plt.subplots(1, 1, figsize=(8, 6))
colors = sns.color_palette('tab10', n_colors=max(6, len(selected_regions)))

# overall Lorenz for reference
all_vals = df['TotalPushes'].dropna().values
if all_vals.size > 0 and all_vals.sum() > 0:
    a_all = np.sort(all_vals)
    pop_all = np.concatenate(([0.0], np.arange(1, a_all.size + 1) / a_all.size))
    cum_all = np.concatenate(([0.0], np.cumsum(a_all) / a_all.sum()))
    ax5.plot(pop_all, cum_all, color='k', linestyle='--', linewidth=1.5, label='All users (reference)')

for i, region in enumerate(selected_regions):
    s = df.loc[df['Locale'] == region, 'TotalPushes'].dropna().values
    if s.size == 0 or s.sum() == 0:
        continue
    a = np.sort(s)
    pop = np.concatenate(([0.0], np.arange(1, a.size + 1) / a.size))
    cum = np.concatenate(([0.0], np.cumsum(a) / a.sum()))

    if region in top_regions:
        ls = '-' ; lw = 2.0
    else:
        ls = '--' ; lw = 1.5

    ax5.plot(pop, cum, label=f'{region} (G={gini_series_local.loc[region]:.2f})', color=colors[i % len(colors)], linestyle=ls, linewidth=lw, alpha=0.9)

ax5.plot([0, 1], [0, 1], ':', color='gray', linewidth=1)
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.set_xlabel('Cumulative share of users')
ax5.set_ylabel('Cumulative share of pushes')
ax5.set_title(f'Lorenz curves — top {top_n} and bottom {top_n} regions by Gini')
ax5.legend(loc='lower right', fontsize='small')
fig5.tight_layout()
fig5.savefig('lorenz_curves.png')
plt.close(fig5)
