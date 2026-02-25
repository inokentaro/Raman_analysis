# Raman_analysis
走査型ラマン顕微鏡で得られたデータからイメージング画像を描画する手法について解説する。
ここでは、Image scan modeについては扱わず、繰り返しLine scan modeを適用して走査する場合についてのみ触れる。 
なお、登場するコードはあくまでもサンプルコードであることに留意されたい。
基本的なラマンの知識については、
- 教科書
- [ラマン散乱について少しだけ詳しい説明](https://kawasaki.web.nitech.ac.jp/jp/lesson/PolRaman.pdf)

などを参照すること。

## データの前処理
### エネルギーからラマンシフトへ変換
Controlから.txtで出力されたデータは、以下のように空白区切りで出力される。
```
エネルギー　カウント数（1列目)　カウント数（2列目）・・・
```
このとき、エネルギーはラマンシフトに換算しておく必要がある。
```python
  def Raman_shift(x):
    return (1/532 - 1/x)*1e7
```

### 平滑化処理
得られたスペクトルはノイズが混入しているため、必要に応じて平滑化処理を施す。
平滑化処理によってピークがブロードになることに注意。
様々な方法が存在するが（例えば移動平均）、pythonではsavgol_filterが便利。
```python
from scipy.signal import savgol_filter
y = raw_data 
y_smooth = savgol_filter(y, window_length, polyorder)
```

### Cosmic Ray の除去
得られたスペクトルは宇宙線が混入しているため、何かしらの方法で除去しておく必要がある。
一般に、標準偏差は外れ値に弱いので、ここでは外れ値に対してロバストな手法として、[中央絶対偏差(Median Absolute Deviation)](https://note.com/maru_mountain/n/n7407f861abaa)を紹介する。
MADは、「データの中央値とのズレ」の中央値、すなわち​
```math
{\rm MAD} = \mathrm{median}(|x_i - \mathrm{median}(x)|)​
```
によって定義される。
データが正規分布に従う場合、
```math
標準偏差＝{\rm MAD} \times 1.4826​
```
となる。従って、（外れ値にロバストな）MADからデータの標準偏差を推定することができる。​
Cosmic ray を除去する場合、例えば「生データと理想的なデータ（平滑化済みデータ）とのズレ」と標準偏差を比較すればよい。
このとき、サンプルのピークと Cosimic ray を区別することは原理的に不可能なので、閾値の設定には必ず人間の目視確認を必要とする。

```python
import numpy as np
from scipy.signal import savgol_filter

def remove_cosmic(y, window=5, poly=3, k=7):
    y_smooth = savgol_filter(y, window_length=window, polyorder=poly)
    r = y - y_smooth
    
    ### Median Absolute Deviation ###
    mad = np.median(np.abs(r - np.median(r)))
    sigma = 1.4826 * mad
    cosmic = np.abs(r) > k * sigma
    ###

    y_clean = y.copy()
    y_clean[cosmic] = np.nan
    
    return y_clean
```

## フィッテイング
ここでは、得られたスペクトルから様々な情報を引き出す手法として、フィッテイング解析を紹介する。
レーザーによって励起されたフォノンは、他のフォノンや電子との散乱によって減衰していくために有限の寿命を持つ。
このとき、
```math
A(t)\propto e^{-t/\tau}e^{-i \omega t}
```
をフーリエ変換すると、
```math
\tilde{A}(\omega) \propto \frac{\tau}{1-i(\omega-\omega_0)\tau}
```
であるから、スペクトル強度は
```math
I(\omega) \propto |\tilde{A}(\omega)|^2 \propto \frac{\tau^2}{1 + (\omega-\omega_0)^2\tau^2}
```
となる。したがって、一般にラマンスペクトルはガウス分布ではなくローレンツ分布に従う。
得られたスペクトルを次の式でフィッテイングすることによって、それぞれのピークのラマンシフトや寿命といった情報を得ることができる。
```math
I(\omega) = {\rm B.G.}+\sum_i \frac{A_i \gamma_i^2}{(\omega - \omega_i)^2 + \gamma_i^2}
```
ここで、B.G. はバックグラウンドである。
おおむね、バックグラウンドは一定もしくは線形であるとしてよい。
```math
{\rm B.G.} = a_0 + a_i \omega
```

以下のサンプルコードでは、`fit_nlorentz_linbg`によって、複数のピークとバックグラウンドを含むラマンスペクトル`y`を`model_nlorentz_linbg`に従って最小二乗法でフィッテイングし、そのパラメータを出力する。
ここで、`x0_list = [peal_loc1, peal_loc2, ...]`には、目視で決めたスペクトル位置をリスト形式で入れる。
このとき、`peak_loc`から±`dx`の範囲で探索する条件を課すことで、的外れなフィッテイングを回避している。

```python
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import least_squares

def lorentz(x, A, x0, gamma):
    return A * (gamma**2) / ((x - x0)**2 + gamma**2)

def model_nlorentz_linbg(x, p, npeak):
    """
    return the fitting function  
    npeak: number of peak
    p: set of parameters of Lorentzian 
    p = [A_1,x0_1,g_1,A_2,x0_2,g_2, ... ,c0,c1]
    """

    y = np.zeros_like(x, dtype=float)

    for i in range(npeak):
        A, x0, g = p[3*i:3*i+3]
        y += lorentz(x, A, x0, g)
    
    c0, c1 = p[3*npeak:3*npeak+2]
    
    return y + (c0 + c1*x)

def fit_nlorentz_linbg(x, y, x0_list, dx=5.0, g_init=5.0, gmin=0.5, gmax=50.0):
    """
    Fitting part
    x0_list: location of the each peak
    dx: the searching window of the peak location
    g_init: initial parameter of gamma
    g_min:
    g_max:
    """
    npeak = len(x0_list)

    # initial value
    c0 = np.nanmedian(y)
    A0 = np.nanmax(y) - c0
    p0 = []
    for x0 in x0_list:
        p0 += [0.7*A0, x0, g_init]
    p0 += [c0, 0.0]
    p0 = np.array(p0, float)

    # boundary conditions of parameters
    lb, ub = [], []
    for x0 in x0_list:
        lb += [0.0, x0-dx, gmin]
        ub += [np.inf, x0+dx, gmax]
    lb += [-np.inf, -np.inf]
    ub += [ np.inf,  np.inf]

    def resid(p):
        # residual error
        return model_nlorentz_linbg(x, p, npeak) - y

    # Fitting
    res = least_squares(resid, p0, bounds=(lb, ub), loss="soft_l1")

    return res.x # set of parameters
```

## 相分率の評価
イメージングの工程は以下の通りである。
1. それぞれの相の基準となるスペクトルを得る
2. Single spectrumの相分率を計算する
3. 2を各サイトごとに行い画像として出力する

### 基準スペクトル
まず、それぞれの相（例えば低温相と高温相）の基準となるスペクトルを用意する。
露光時間を増やしてS/N比が高いスペクトルが望ましい。
多くの場合、スペクトルにはアーティファクトが混入するため、ここでは基準スペクトルをフィッテイングによって求めることにする。
生データに対してCosmic rayを除去したあと、フィッテイングによってバックグラウンドとサンプル由来の信号を分離する。
このとき、注目するピークの周辺のみのデータを用いれば十分である。
また、フィッテイングによって基準スペクトルを用意するため、必ずしも生データに平滑化処理を行う必要はない。

### 相分率の計算
与えられたsingle spectrumに対して、以下の式を用いて相分率Φを求める。
```math
I(\omega)={\rm B.G.}+A\left[\phi I_1(\omega) + (1-\phi)I_2(\omega) \right]
```
このとき、バックグラウンドのダブルカウントを防ぐために、基準スペクトルとしてはバックグラウンドを除いたスペクトルを採用する。
ここで、パラメータAはサンプル由来の信号強度とみなすことができ、画像化する際にデフォーカス領域の判別に用いる。
以下のサンプルコードでは、基準スペクトル`yref_HT, yref_LT`を用いて、 single spectrum`ym`の相分率`f`とパラメータ`A`を求める。

```python
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear

def estimate_volume_fraction(xm, ym, yref_HT, yref_LT):
    """
    xm, ym : raw data
    yref_** : ref. spectrum (after BG subtracting)
    use_nnls : True if c0,c1>=0 
    """
    # interpolate ref. spectrum
    Ih = np.interp(xm, xf, yref_HT)
    Il = np.interp(xm, xf, yref_LT)

    # removing np. NaN
    ok = np.isfinite(xm) & np.isfinite(ym) & np.isfinite(Ih) & np.isfinite(Il)
    x = xm[ok]
    y = ym[ok]
    Ih = Ih[ok]
    Il = Il[ok]

    # y = c0*Ih + c1*Il + b0 + b1*x
    X = np.column_stack([Ih, Il, np.ones_like(x), x])

    # boundary conditions: c0,c1 >= 0 ; b0,b1 free
    lb = np.array([0.0, 0.0, -np.inf, -np.inf], dtype=float)
    ub = np.array([np.inf, np.inf,  np.inf,  np.inf], dtype=float)

    # solve constrained linear least squares
    res = lsq_linear(X, y, bounds=(lb, ub), method="trf", lsmr_tol="auto")
    beta = res.x
    c0, c1, b0, b1 = beta

    # calculate volume fraction of LT phase
    A = c0 + c1
    f = np.nan if A == 0 else (c1 / A)

    # fitted curve
    y_fit = X @ beta
    residual = y - y_fit

    return {
        "f": f, "A": A,
        "c0": c0, "c1": c1, "b0": b0, "b1": b1,
        "x": x, "y": y, "y_fit": y_fit, "Ih": Ih, "Il": Il, "residual" : residual
    }
```

### 画像化
以上の手続きにより、1点における相分率が計算できる。
走査領域に対して、各ピクセルごとに相分率を表示することで、二次元ラマンイメージが得られる。
デフォーカス領域を除去したい場合は、パラメータAがある閾値より小さい場所をマスク処理すればよい。
このとき、閾値の設定は手で入れることになるため、生データと相分率計算で得られたスペクトルを重ねて、どのくらいのAであれば相共存が目視でも判断可能か確認しておく必要がある。
