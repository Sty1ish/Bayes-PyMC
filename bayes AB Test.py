#%% Default Setting
from dataclasses import dataclass
from typing import Dict, List, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy.stats import bernoulli, expon

RANDOM_SEED = 4000
rng = np.random.default_rng(RANDOM_SEED)

plotting_defaults = dict(
    bins=50,
    kind="hist",
    textsize=10,
)

@dataclass
class BetaPrior:
    alpha: float
    beta: float
    
@dataclass
class BinomialData:
    trials: int
    successes: int
    
class ConversionModelTwoVariant:
    def __init__(self, priors: BetaPrior):
        self.priors = priors

    def create_model(self, data: List[BinomialData]) -> pm.Model:
        trials = [d.trials for d in data]
        successes = [d.successes for d in data]
        with pm.Model() as model:
            p = pm.Beta("p", alpha=self.priors.alpha, beta=self.priors.beta, shape=2)
            obs = pm.Binomial("y", n=trials, p=p, shape=2, observed=successes)
            reluplift = pm.Deterministic("reluplift_b", p[1] / p[0] - 1)
        return model
    
#%%
# check prior predictive
# Prior 함수에 높은 Alpha 값을 할당하면, 신뢰구간이 더 좁아진다. (유의성 확보를 위해 더 많은 표본 필요)

weak_prior = ConversionModelTwoVariant(BetaPrior(alpha=100, beta=100))
strong_prior = ConversionModelTwoVariant(BetaPrior(alpha=10000, beta=10000))

with weak_prior.create_model(data=[BinomialData(1, 1), BinomialData(1, 1)]):
    weak_prior_predictive = pm.sample_prior_predictive(samples=10000, return_inferencedata=False)
    
with strong_prior.create_model(data=[BinomialData(1, 1), BinomialData(1, 1)]):
    strong_prior_predictive = pm.sample_prior_predictive(samples=10000, return_inferencedata=False)
    
fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
az.plot_posterior(weak_prior_predictive["reluplift_b"], ax=axs[0], **plotting_defaults)
axs[0].set_title(f"B vs. A Rel Uplift Prior Predictive, {weak_prior.priors}", fontsize=10)
axs[0].axvline(x=0, color="red")
az.plot_posterior(strong_prior_predictive["reluplift_b"], ax=axs[1], **plotting_defaults)
axs[1].set_title(f"B vs. A Rel Uplift Prior Predictive, {strong_prior.priors}", fontsize=10)
axs[1].axvline(x=0, color="red");

#%%
# 실험 데이터 생성 방법
# 하나의 데이터셋은 A,B 모두 동일한 전환율 값을 설정 (true conversion rate인데 이 값은 A, B 전체의 평균으로 설정)

# 모델의 인풋은 generate_binomial_data(["A", "B"], [0.23, 0.23])의 결과임. 기억할 것.

def generate_binomial_data(
    variants: List[str], true_rates: List[str], samples_per_variant: int = 100000
) -> pd.DataFrame:
    data = {}
    for variant, p in zip(variants, true_rates):
        data[variant] = bernoulli.rvs(p, size=samples_per_variant)
    agg = (
        pd.DataFrame(data)
        .aggregate(["count", "sum"])
        .rename(index={"count": "trials", "sum": "successes"})
    )
    return agg

# Example generated data
generate_binomial_data(["A", "B"], [0.23, 0.23])


# 즉 우리는 A / B의 경우, 각 시도에 따른 확률 값을 알아야 한다.
# A, B의 노출 횟수는 동일해야함. 동일하지 않으면 붓스트랩 사용해서 확률 구해.

#%%
# 플롯 그리는 함수 만들기
def run_scenario_twovariant(
    variants: List[str],
    true_rates: List[float],
    samples_per_variant: int,
    weak_prior: BetaPrior,
    strong_prior: BetaPrior,
) -> None:
    generated = generate_binomial_data(variants, true_rates, samples_per_variant)
    data = [BinomialData(**generated[v].to_dict()) for v in variants]
    with ConversionModelTwoVariant(priors=weak_prior).create_model(data):
        trace_weak = pm.sample(draws=5000)
    with ConversionModelTwoVariant(priors=strong_prior).create_model(data):
        trace_strong = pm.sample(draws=5000)

    true_rel_uplift = true_rates[1] / true_rates[0] - 1

    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    az.plot_posterior(trace_weak.posterior["reluplift_b"], ax=axs[0], **plotting_defaults)
    axs[0].set_title(f"True Rel Uplift = {true_rel_uplift:.1%}, {weak_prior}", fontsize=10)
    axs[0].axvline(x=0, color="red")
    az.plot_posterior(trace_strong.posterior["reluplift_b"], ax=axs[1], **plotting_defaults)
    axs[1].set_title(f"True Rel Uplift = {true_rel_uplift:.1%}, {strong_prior}", fontsize=10)
    axs[1].axvline(x=0, color="red")
    fig.suptitle("B vs. A Rel Uplift")
    return trace_weak, trace_strong

#%%
# A,B의 전환율이 동일하다고 가정했을때. (각각 23%)

trace_weak, trace_strong = run_scenario_twovariant(
    variants=["A", "B"],
    true_rates=[0.23, 0.23],
    samples_per_variant=100000,
    weak_prior=BetaPrior(alpha=100, beta=100),
    strong_prior=BetaPrior(alpha=10000, beta=10000),
)

# 플롯에서 신뢰구간에서 0이 포함되었기 때문에, 적용을 고려하지 않는다.

#%%
# A,B의 전환율이 서로 다르다고 가정했을때 (각 21, 23%)
run_scenario_twovariant(
    variants=["A", "B"],
    true_rates=[0.21, 0.23],
    samples_per_variant=100000,
    weak_prior=BetaPrior(alpha=100, beta=100),
    strong_prior=BetaPrior(alpha=10000, beta=10000),
)

# 플롯에서 0이 포함되지 않는 경우, 확실히 개선이 적용되었다고 판단, 적용을 고려하는게 맞다.

#%%

# 이진분류는 유지하고 (클릭-논클릭 : 베르누이), A/B/C 테스트

class ConversionModel:
    def __init__(self, priors: BetaPrior):
        self.priors = priors

    def create_model(self, data: List[BinomialData], comparison_method) -> pm.Model:
        num_variants = len(data)
        trials = [d.trials for d in data]
        successes = [d.successes for d in data]
        with pm.Model() as model:
            p = pm.Beta("p", alpha=self.priors.alpha, beta=self.priors.beta, shape=num_variants)
            y = pm.Binomial("y", n=trials, p=p, observed=successes, shape=num_variants)
            reluplift = []
            for i in range(num_variants):
                if comparison_method == "compare_to_control":
                    comparison = p[0]
                elif comparison_method == "best_of_rest":
                    others = [p[j] for j in range(num_variants) if j != i]
                    if len(others) > 1:
                        comparison = pm.math.maximum(*others)
                    else:
                        comparison = others[0]
                else:
                    raise ValueError(f"comparison method {comparison_method} not recognised.")
                reluplift.append(pm.Deterministic(f"reluplift_{i}", p[i] / comparison - 1))
        return model
def run_scenario_bernoulli(
    variants: List[str],
    true_rates: List[float],
    samples_per_variant: int,
    priors: BetaPrior,
    comparison_method: str,
) -> az.InferenceData:
    generated = generate_binomial_data(variants, true_rates, samples_per_variant)
    data = [BinomialData(**generated[v].to_dict()) for v in variants]
    with ConversionModel(priors).create_model(data=data, comparison_method=comparison_method):
        trace = pm.sample(draws=5000)

    n_plots = len(variants)
    fig, axs = plt.subplots(nrows=n_plots, ncols=1, figsize=(3 * n_plots, 7), sharex=True)
    for i, variant in enumerate(variants):
        if i == 0 and comparison_method == "compare_to_control":
            axs[i].set_yticks([])
        else:
            az.plot_posterior(trace.posterior[f"reluplift_{i}"], ax=axs[i], **plotting_defaults)
        axs[i].set_title(f"Rel Uplift {variant}, True Rate = {true_rates[i]:.2%}", fontsize=10)
        axs[i].axvline(x=0, color="red")
    fig.suptitle(f"Method {comparison_method}, {priors}")

    return trace


_ = run_scenario_bernoulli(
    variants=["A", "B", "C"],
    true_rates=[0.21, 0.23, 0.228],
    samples_per_variant=100000,
    priors=BetaPrior(alpha=5000, beta=5000),
    comparison_method="best_of_rest",
)


#%%


