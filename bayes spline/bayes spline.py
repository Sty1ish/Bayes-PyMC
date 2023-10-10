from pathlib import Path

import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from patsy import dmatrix

RANDOM_SEED = 8927
az.style.use("arviz-darkgrid")

#%%
os.chdir(r'C:\Users\stylish1379_2bytesco\Desktop\spline')
sample_data = pd.read_csv('sg spline.csv')
sample_data.head(n=10)

sample_data.shape

sample_data.columns

# 플롯 라벨 그리기 위해 변수 저장.
sample_data.DAY = pd.to_datetime(sample_data.DAY)
sample_data = sample_data.reset_index()
total_len = len(sample_data.DAY)


#%%
'''
아래 변수는 기간으로 "index"를 사용하고, 평활값을  "Wishlists" 를 사용하였음
sample_data.columns ctrl+f로 필요한 값 변경해서 사용할 것.
'''

#%%
# 스케터 플롯

x = sample_data.plot.scatter(
    "index", "Wishlists", color="cornflowerblue", s=10, title="Cherry Blossom Data", ylabel="Days in bloom", rot = 0
);
x.set_xticks(np.arange(0, total_len + 1, 60))

#%%
# num_knots => 섹션의 개수

num_knots = 15
knot_list = np.quantile(sample_data.index, np.linspace(0, 1, num_knots))
knot_list

# 섹션 플롯
x = sample_data.plot.scatter(
    "index", "Wishlists", color="cornflowerblue", s=10, title="Cherry Blossom Data", ylabel="Days in bloom", rot = 0
);
x.set_xticks(np.arange(0, total_len + 1, 60))

for knot in knot_list:
    plt.gca().axvline(knot, color="grey", alpha=0.4);
    

#%%
# B 행렬을 이용해서 B-spline을 회귀를 위해 사용할 수 있음. 아래는 3차
B = dmatrix(
    "bs(index, knots=knots, degree=3, include_intercept=True) - 1",
    {"index": sample_data.index.values, "knots": knot_list[1:-1]},
)
B


# B행렬이 N차원임을 고려할때 뭘 고를래? 참고 그래프.
'''
The b-spline basis is plotted below, showing the domain of each piece of the spline. 
The height of each curve indicates how influential the corresponding model covariate (one per spline region) 
will be on model’s inference of that region. The overlapping regions represent the knots, 
showing how the smooth transition from one region to the next is formed.

아래에는 스플라인의 각 부분의 영역을 보여주는 B-스플라인 기준이 나와 있습니다. 
각 곡선의 높이는 해당 모델 공변량(스플라인 영역당 하나)이 해당 영역에 대한 모델의 추론에 얼마나 영향을
 미치는지를 나타냅니다. 겹치는 영역은 매듭을 나타내며, 한 영역에서 다음 영역으로의 부드러운 전환이
 어떻게 형성되는지를 보여줍니다.
'''

spline_df = pd.DataFrame(B).assign(index=sample_data.index.values).melt("index", var_name="spline_i", value_name="value")

color = plt.cm.magma(np.linspace(0, 0.80, len(spline_df.spline_i.unique())))

fig = plt.figure()
for i, c in enumerate(color):
    subset = spline_df.query(f"spline_i == {i}")
    subset.plot("index", "value", c=c, ax=plt.gca(), label=i)
plt.legend(title="Spline Index", loc="upper center", fontsize=8, ncol=6);
    
#%%
# 사전분포 사후분포 확인

COORDS = {"splines": np.arange(B.shape[1])}
with pm.Model(coords=COORDS) as spline_model:
    a = pm.Normal("a", 100, 5)
    w = pm.Normal("w", mu=0, sigma=3, size=B.shape[1], dims="splines")
    mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
    sigma = pm.Exponential("sigma", 1)
    D = pm.Normal("D", mu=mu, sigma=sigma, observed=sample_data.Wishlists) # this is observed / 관측값
    
pm.model_to_graphviz(spline_model)

with spline_model:
    idata = pm.sample_prior_predictive()
    idata.extend(pm.sample(draws=1000, tune=1000, random_seed=RANDOM_SEED, chains=4))
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# 계수 해석방법은 다음 참고
# 시그마 값이 작고, 신뢰구간이 작고.  결정계수가 클수록 좋음
az.summary(idata, var_names=["a", "w", "sigma"])
az.plot_forest(idata, var_names=["w"], combined=False, r_hat=True);


#%%
wp = idata.posterior["w"].mean(("chain", "draw")).values

spline_df = (
    pd.DataFrame(B * wp.T)
    .assign(index=sample_data.index.values)
    .melt("index", var_name="spline_i", value_name="value")
)

spline_df_merged = (
    pd.DataFrame(np.dot(B, wp.T))
    .assign(index=sample_data.index.values)
    .melt("index", var_name="spline_i", value_name="value")
)

# 스플라인 잘라진 애들이 각각 어떤 형태로 예측했는지
# 분할된 애들마다 어떤 예측했는지는 번호고, 까만 실선은 실제 예측

color = plt.cm.rainbow(np.linspace(0, 1, len(spline_df.spline_i.unique())))
fig = plt.figure()
for i, c in enumerate(color):
    subset = spline_df.query(f"spline_i == {i}")
    subset.plot("index", "value", c=c, ax=plt.gca(), label=i)
spline_df_merged.plot("index", "value", c="black", lw=2, ax=plt.gca())
plt.legend(title="Spline Index", loc="lower center", fontsize=8, ncol=6)

for knot in knot_list:
    plt.gca().axvline(knot, color="grey", alpha=0.4);
    
#%% 예측으로 시각화 하기

post_pred = az.summary(idata, var_names=["mu"]).reset_index(drop=True)
sample_data_post = sample_data.copy().reset_index(drop=True)
sample_data_post["pred_mean"] = post_pred["mean"]
sample_data_post["pred_hdi_lower"] = post_pred["hdi_3%"]
sample_data_post["pred_hdi_upper"] = post_pred["hdi_97%"]
sample_data.plot.scatter(
    "index",
    "Wishlists",
    color="cornflowerblue",
    s=10,
    title="Cherry blossom data with posterior predictions",
    ylabel="Days in bloom",
)
for knot in knot_list:
    plt.gca().axvline(knot, color="grey", alpha=0.4)

sample_data_post.plot("index", "pred_mean", ax=plt.gca(), lw=3, color="firebrick")
plt.fill_between(
    sample_data_post.index,
    sample_data_post.pred_hdi_lower,
    sample_data_post.pred_hdi_upper,
    color="firebrick",
    alpha=0.4,
);
