# 1. Intro

We are trying to develop an arbitrage strategy on between a spot token and it's perpetual future. For now we will attempt to develop an arb strategy for PERP-ETH/ETH pair on the FTX exchange. 

We note S the spread between the spot and perpetual futures contracts. S is likely mean reverting. We will verify this after.

## 1.1) Assuming no funding
The profit of a trade is equal to:

$P = Amount \left(- 4 Fee - S_{close} + S_{open}\right)$

Thus a trade is profitable if: 

$- 4 Fee - S_{close} + S_{open} = 0$

The series being mean-reverting, we can assume having S = 0 at some point time. Thus after simplification (i.e no capital cost, assuming perfect execution), we can assume the trade is profitable if:

$S_{open} > 4 Fee$

## 1.2) Assuming funding

If we introduce funding we now need to take into account the funding fee happening every hour on the FTX exchange.


# 2. Strategies
## 2.1) First strategy

The first strategy relies on closing the trade before the funding no matter what.
We calculate expected trade profit based on the assumption we will close before funding occurs. 

$P = (S_{open} - S_{close} - 4 * Fee) * Amount$

We note E[P] the statistical expected value of the profit

$E[P] = (S_{open} - E[S_{close}] - 4 * Fee) * Amount$

S is mean reverting. We can model it with an OU stochastic process.\


$S_t - S_{(t-1)} = \mu * (1 - exp(-\theta)) + S_{(t-1)} * (exp(-\theta) - 1) + \epsilon(t)$

where: \
$S_t$: spread at time t \
$S_{(t-1)}$: spread at time t-1 \
$\mu$: mean (we will get it by fitting historical values) \
$\theta$: mean reversion speed (we will get it by fitting historical values) \
$\epsilon$: residual

(OU Process Details: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)

We'll attempt to find $\mu$ and $\theta$ later by fitting historical values. 

We also have: \
$E[S_t] = \mu + (S_{open} - \mu) * exp(-\theta * t)$

where: \
t: time since after we open the trade \
$S_t$ the spread at time t \
$S_{open}$ the spread when opening the trade \
$S_{close}$ the spread when closing the trade \
$\mu$: mean (we will get it by fitting historical values) \
$\theta$: mean reversion speed (we will get it by fitting historical values) \

\
\
So we have: \
$E[P] = (S_{open} + E[S_{close}] - 4 * Fee) * Amount$ \
$E[P] = (S_{open} + \mu + (S_{open} - \mu) * exp(-\theta * (T_f - t)) - 4 * Fee) * Amount$

where: \
t: time since after we open the trade \
$S_t$ the spread at time t \
$S_{open}$ the spread when opening the trade \
$S_{close}$ the spread when closing the trade \
$\mu$: mean (we will get it by fitting historical values) \
$\theta$: mean reversion speed (we will get it by fitting historical values) \
$T_f$ is the time of next funding

Conclusion: \
We open if: $S_{t} > (\mu * (1 + exp(-\theta * (T_f - t))) + 4 * Fee) / (1 - exp(-\theta * (T_f - t)))$ \
We close if: $S_{t} < 0$ or if t ~ $T_f$
\
\
We could assume $\mu = 0$, but it's not really the case for the FTX as the calculations after will show. The futures tend to hover above the spot price.



---
## 2.2) Strategy 2 
\
Generalizing we have: 
E[P]_N is the expected profit if we close the trade after N funding windows:
p_n is the probability that we close the trade in the n_th funding window. 

Thus:
$E[P] = p_1 * E[P]_1 + p_2 * E[P]_2 + .... p_n * E[P]_N$

where: 
- $P$ is the profit of a trade \
- $k=[1,N]$ are indices corresponding to funding windows\
- $p_k$ is the probability of closing trade during N-th funding window \
- $E[P]_k$ is the expected profit of trade closed during the N-th funding window


### 2.2.1) Computing $E[P]_k$ (Expected value of profit if trade close in k-th funding window)
$E[P]_k$ is computed similarly to E[P] in the previous sections. \
$E[P]_k = S_{open} * (1 - exp(-eta * ((k - 1) * \Delta T_{Funding} + T_{Funding}- t)) - 4 Fee - (k-1) * E[Funding]_k$ 


### 2.2.2) Computing $E[Funding]_k$ (Expected value of funding in k-th funding window)

- Compute expected value of funding from live indicators \
The formula used for computing funding was a bit unclear to me: https://help.ftx.com/hc/en-us/articles/360027946571-Funding. 
The one hour twap is based on 1s orderbook snapshots
Apparently the price mentioned is market_price = median(last, best_bid, best_ask)
Additionally, the price is computed from the median(last, best bid, best offer) on 1s every snapshots. Currently i'm using ohlcv as a first approach which makes this hard to implement.

- Compute expected value of funding from candles + model based on historic data \




### 2.2.3) Computing $p_k$ (Probability of closing trade in k-th funding window)
I haven't had the time to look into how compute p_n exactly but probably either of these methods. p_n depends on the trade close conditions.

We use historical values to compute the probability distribution of time to return to the mean. For example assuming normal distribution of time to return to the mean (to verify), we can find the probability of reversion to the mean (= probability of closing) for every funding window.

${\Delta T}$ is the random variable expressing mean reversion times.

Mean Reversion Times computed from the historical values are assumed normally distributed $N(\mu_{\Delta T} , \sigma_{\Delta T})$

$p_1 = p(\Delta T < T_F - t)$ the probability that the series reverts to the mean before $T_F$ \
$p_2 = p(T_F - t < \Delta T < 2 T_F - t)$ the probability that the series reverts to the mean before $2T_F$ after $T_F$

$p_1 = p(\Delta T < T_F - t) = (1/2) [1 + erf((T_F - t - \mu_{\Delta T})/ (\sqrt2\sigma_{\Delta T}))$ \
$p_2 = p(\Delta T < 2 T_F - t) - p_1 = (1/2) * [erf((2T_F - t - \mu_{\Delta T})/ (\sqrt2\sigma_{\Delta T})) - erf((T_F - t - \mu_{\Delta T})/ (\sqrt2\sigma_{\Delta T}))]$ \
etc.


NOTE: \
1 - erf is the error function. Can be calculated with math.erf. Details at: https://en.wikipedia.org/wiki/Normal_distribution\

2 - Maybe expanding to that much terms is not necessary if the p_n * E[P]_N (expected profit for closing during Nth window) terms are negligible, this needs numerical verification. 

3 - We can also compute from model directly given the model is fit. For example, i think in the case of the OU process $p_k$ and $p_{(k-1)}$ are related geometrically. I haven't had the time to look into it but probably something $p_k = 1/\theta * p_{(k-1)}$. Then knowing $\sum p_k = 1$ with k=[1..N], we can compute $p_k$. 


# 3. Application to arb on PERP-ETH / ETH

## 3.1. Evaluation for the first strategy

Parameters: \
exchange = ftx \
spot = ETH-USDT \
future = PERP-ETH \
start date = 01/09/2020 \
end date = 14/09/2020 \
fee = 0.0007 (assuming taker, https://help.ftx.com/hc/en-us/articles/360024479432-Fees) 

With these parameters we find \
$E[S_t] = 0.13 + (S_0 - 0.13) * e^{(-0.248 * t)}$

Meaning: \
$\theta = 0.248$ \
$\overline{\mu} = 0.13$

On the first strategy, the open condition was: \
$S_{t} > (\mu * (1 + exp(-\theta * (T_f - t))) + 4 * Fee) / (1 - exp(-\theta * (T_f - t)))$

$S_{t} > (0.13 * (1 + exp(-0.248 * (T_f - t)) + 4 * 0.00007) / (1 - exp(-0.248 * (T_f - t)))$

## 3.2. Evaluation for the second strategy

We find: 
Average mean reversion time: $\mu_{\Delta T} = 06m:48s$ \
Standard deviation mean reversion time: $\sigma_{\Delta T} = 34m:38s$

In previous section, we found: \
$p_1 = (1/2) [1 + erf((T_F - t - \mu_{\Delta T})/ (\sqrt2\sigma_{\Delta T}))$ \
$p_2 = (1/2) [erf((2T_F - t - \mu_{\Delta T})/ (\sqrt2\sigma_{\Delta T})) - erf((T_F - t - \mu_{\Delta T})/ (\sqrt2\sigma_{\Delta T}))]$  \
$p_n$ ... etc.

#### TODO
- compute p_1 for example


# 4. Improvements/TODO

## 4.1 Improvements on strategies (part 2)

* There are certainly better models that the ornstein uhlenbeck process. Things like ARMA, GARCH, jump-diffusion processes. Maybe other ML models like random forest, etc. Differentiate low volatility, high volatility regimes. Continuously verify and refit the model with recent data.

* Exit before S = 0. I assumed that S = 0 would be the most optimal exit but i'm not sure. 

* Recompute the expected profit of a trade with maker orders. Here the idea is that we would open from a lower spread, hoping that the spread grows a bit and fills our orders and then reverts to the mean. On average, the revenue from the spread would be lower but fees would be lower too. 

* Currently we reduced the problem S_open > 0. Check case of S_open < 0. The  equations will be very similar.

## 4.2 Improvements on application (part 3)

* $\mu_{\Delta T}$ and $\mu_{\Delta T}$ should be made more accurate, for example finding avg, std of return to the mean given a certain number. 

* Instead of using candle data we would have to use L2, L3 data. That would enable computing expected values for funding and 


And much more

