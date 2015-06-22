####やること
-  ツイートに対する感情分類器を作成
- ナイーブベイズ (Naive Bayes) の考え方と実装
- ツールは scikit-learn を使う

####背景
- 個人のユーザ情報は誰でもリアルタイムに取得できる(Twitterなど)
- それらの情報に対して感情分析を行えば，周囲の反応を把握可能
- 感情分析とは書き手が取り扱っているテーマについて，その感情を調べること
	- 意見マイニング(opinion mining)とも呼ばれる

###1. ツイートデータを取得
- Niek Sanders氏により作成されたコーパスを利用
	- ツイートに対して手作業でラベル付けされている
	- 合計5,000 tweets (英語は3500 tweets)
	- 対応するラベル
		- positive
		- negative
		- neutral，irrelevant(意見なし)
<br>
###2. ナイーブベイズ分類器について
- 最も洗練された機械学習アルゴリズムのひとつ
- ナイーブという名前だが，分類器の性能は決してナイーブ(単純)ではない
- 出力と関係ない特徴量にロバスト(無関係な特徴量は上手いこと無視)
- 学習も予測も高速
- 「ナイーブ」という名前が付けられた理由
	- ベイズ定理を用いるために一つの仮定が必要
	- その仮定とは，全ての特徴量は互いに独立であるというもの
	- 実際の世界では，その仮定が成立するのは稀
	- しかし仮定が成立しない場合でも，優れた性能を示すことが多い
	
###2.1. ベイズ定理
- ナイーブベイズで行うこと
	- どの特徴量が証拠となるかを追跡

今回例として利用する変数

変数 | 取りうる値   | 意味
----|-------------|-----
C   |"pos", "neg" | ツイートが属するクラス(「ポジティブ」か「ネガティブ」)
F1  |非負整数      |ツイートで「awesome」という単語が用いられた回数
F2  |非負整数      |ツイートで「crazy」という単語が用いられた回数


入力Xが与えられた時に出力Yが得られる確率 [tex:P(Y|X)] は，ベイズの定理より，

> [tex:\displaystyle P(Y|X) = \frac{P(Y) P(X|Y)}{P(X)}]

で，この確率が最大となるYを求めるときに [tex:P(X)] は定数なので，

> [tex: P(Y|X) \propto P(Y) P(X|Y) ]

これを今回の例で考えると，

> [tex:\displaystyle P(C|F_1, F_2) = \frac{P(C) P(F_1, F_2|C)}{P(F_1, F_2)} ]

> [tex:\displaystyle 事後確率 = \frac{事前確率・尤度}{証拠} ]

- 事後確率: [tex: P(C|F_1, F_2)]
	- 入力が与えられた場合に，そのデータがクラス C に属する確率
- 事前確率: [tex: P(C)]
	- データ情報がない場合に，そのデータがクラス C に属する確率
- 証拠: [tex: P(F_1, F_2)]
	- 特徴量が F1 と F2 をとる確率
	- この値は訓練データにおいて，該当する特徴量が全体に占める割合
- 尤度: [tex: P(F_1, F_2|C)]
	- もしあるデータがクラスCに属する場合，特徴量が F1 と F2 である確率がいくらか



確率理論から，

> [tex: P(F_1, F_2|C) = P(F_1|C)P(F_2|C, F_1)]

？？？？？

これでは，ある難しい問題 [tex: P(F_1, F_2|C) ] を別の難しい問題 [tex:
P(F_2|C, F_1) ] に変換しただけ．．．

ここで物事を**ナイーブ**に考える．「F1 とF2 が互いに独立である」と仮定すると，[tex:
P(F_2|C, F_1) ] を [tex:P(F_2|C)] と書くことができる．これにより上の式は，

> [tex: P(F_1, F_2|C) = P(F_1|C)P(F_2|C) ]

よって最終的に，

> [tex:\displaystyle P(C|F_1, F_2) = \frac{P(C) P(F_1|C) P(F_2|C)}{P(F_1, F_2)} ]

となる．
上記の仮定は都合的なもので理論的には正しくないにもかかわらず，現実のアプリケーションでは非常に優れた結果になることが多い．

<br>
###2.2. ナイーブベイズを用いて分類を行う

新しいツイートが与えられた時，次の確率を計算

> [tex:\displaystyle P(C="pos"|F_1, F_2) = \frac{P(C="pos") P(F_1|C="pos") P(F_2|C="pos")}{P(F_1, F_2)} ]
> [tex:\displaystyle P(C="neg"|F_1, F_2) = \frac{P(C="neg") P(F_1|C="neg") P(F_2|C="neg")}{P(F_1, F_2)} ]

- これより確率の高いクラス (Cbest) を選ぶ必要がある
- 分母は無視することができる
- 数式については次のように書くことができる


####簡単な例でナイーブベイズを追う

- ツイートの内容は「awesome」と「crazy」という二つの単語だけしか使われないこととする
- ツイートに対してポジティブ・ネガティブのラベルがすでに手作業で付けられている
- 以下の7ツイートがデータとして与えられていたとする

Tweet|Class
-----|-----
awesome | positive
awesome | positive
awesome crazy | positive
crazy | positive
crazy | negative
crazy | negative

表より，

> [tex:\displaystyle P(C="pos") = \frac{4}{6} ]

> [tex:\displaystyle P(C="neg") = \frac{2}{6} ]

> [tex:\displaystyle P(F_1 = 1|C="pos") = \frac{posで"awesome"を含む数}{posのツイート数} = \frac{3}{4} = 0.75 ]

> [tex:\displaystyle P(F_1 = 0|C="pos") = 1 - 0.75 = 0.25 ]

> [tex:\displaystyle P(F_1 = 1|C="neg") = \frac{0}{2} = 0 ]

> [tex:\displaystyle P(F_1 = 0|C="neg") = 1 ]

> [tex:\displaystyle P(F_2 = 1|C="pos") = 0.5 ]

> [tex:\displaystyle P(F_2 = 0|C="pos") = 0.5 ]

> [tex:\displaystyle P(F_2 = 1|C="pos") = 0 ]

> [tex:\displaystyle P(F_2 = 0|C="neg") = 1 ]


これらの情報から未知の tweet に対して最大となるCを推定する．

- 未知の tweet「awesome」**→ positive**
> [tex:\displaystyle P(C="pos"|F_1 = 1, F_2 = 0) = \frac{0.67 \cdot 0.75 \cdot 0.5 }{0.44} = 0.57 ]
> [tex:\displaystyle P(C="neg"|F_1 = 1, F_2 = 0) = \frac{0.33 \cdot 0 \cdot 0 }{0.44} = 0 ]


- 未知の tweet「crazy」**→ negative**
> [tex:\displaystyle P(C="pos"|F_1 = 0, F_2 = 1) = \frac{0.67 \cdot 0.25 \cdot 0.5 }{0.33} = 0.25 ]
> [tex:\displaystyle P(C="neg"|F_1 = 0, F_2 = 1) = \frac{0.33 \cdot 1 \cdot 1 }{0.33} = 1 ]


- 未知の tweet「awesome crazy」**→ positive**
> [tex:\displaystyle P(C="pos"|F_1 = 1, F_2 = 1) = \frac{0.67 \cdot 0.75 \cdot 0.5 }{0.22} = 0.76 ]
> [tex:\displaystyle P(C="neg"|F_1 = 1, F_2 = 1) = \frac{0.33 \cdot 0 \cdot 1 }{0.22} = 0 ]



- 未知のtweet「text」**→ 未知語への対応が必要**
> [tex:\displaystyle P(C="pos"|F_1 = 0, F_2 = 0) = \frac{0.67 \cdot 0.25 \cdot 0.5 }{0} = ?? ]
> [tex:\displaystyle P(C="neg"|F_1 = 0, F_2 = 0) = \frac{0.33 \cdot 1 \cdot 0 }{0} = ?? ]


<br>
###2.3. 新出単語への対応
- 最も簡単なラプラス・スムージングは

> [tex:\displaystyle P(F_1 = 1|C="pos") = \frac{3}{4} = 0.75]

の式において，分子に+1をすると同時に，分母に+(Cの数)をすることで確率を保ちつつスムージングを行う．

> [tex:\displaystyle P(F_1 = 1|C="pos") = \frac{3+1}{4+2} = 0.67]

<br>
###2.4. アンダーフローへの対応
- 確率の対数をとる
    - 対数をとっても確率の大小関係は変わらない

> [tex:\displaystyle \log{P(C) P(F_1|C) P(F_2|C)} = \log{P(C)}  + \log{P(F_1|C)} + \log{P(F_2|C)} ]

以前考えた Cbest について考えると，

> [tex: Cbest = argmax \log{P(C = c) P(F_1|C =c) P(F_2|C =c)} ]
> [tex: Cbest = argmax( \log{P(C = c)}  + \log{P(F_1|C =c)} + \log{P(F_2|C =c)}) ]

任意の数 (k個) の特徴量を用いた場合の式は以下になる．

> [tex: Cbest = argmax( \log{P(C = c)}  +\sum_{k} \log{P(F_k|C =c)}) ]

<br>
###3.1. scikit-learn を用いたナイーブベイズ分類器の作成

[このあたり](https://github.com/aweiand/TwitterSentiment/tree/master/GetTwitterCorpus)の感情ラベル付きのツイートデータをいただいてくる．
各行が，ポジネガラベル+","+ツイート内容
で構成されている csvfile を用意したとする．
X をツイートのリスト，Y をそのラベルのリストとして，np.array に突っ込むと，以下のようになる．

```
import numpy as np
X =[]
Y =[]
for line in open("full-corpus-2col-2class.csv"):
    item = line.strip().split(",")
    if len(item) > 1:
         X.append(','.join(item[1:]))
         if item[0] == 'positive':
             label = True
         else:
             label = False
         Y.append(label)

npX = np.array(X)
npY = np.array(Y)

```

TfidfVectorizer を使い，生のテキストデータを TF-IDF の特徴量に変える．
TF-IDF の特徴量とラベルを合わせて分類器を学習させる．Pipeline というクラスは，ベクトル化を行う機能と分類器を合わせて持っている．はじめの特徴量は単語の3gramとする．

```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit
from utils import plot_pr


def create_ngram_model():
    tfidf_ngrams = TfidfVectorizer(ngram_range=(1, 3),analyzer="word", binary=False)
    clf = MultinomialNB()
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
    return pipeline


```

また今回はデータ量が少ないので交差検定を用いる．
ここでは KFold を使わず，その代わりに ShuffleSplit を使う．KFold はデータ集合を頭から順に均等に分割するが，ShuffleSplit はデータを混ぜてくれる．
以下に，必要なことを全て行う train_model() という関数を作る．
なお，[すでに公開されているコード](https://github.com/tomzaragoza/learning-ml-python/tree/master/ch06)を参考にした．


```
def train_model(clf_factory, X, Y, name="NB ngram", plot=False):
    cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)
    train_errors = []
    test_errors = []
    scores = []
    pr_scores = []
    precisions, recalls, thresholds = [], [], []
    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]
        clf = clf_factory()
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)
        scores.append(test_score)
        proba = clf.predict_proba(X_test)
        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, 1])
        precision, recall, pr_thresholds = precision_recall_curve(y_test, proba[:, 1])
        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)
    scores_to_sort = pr_scores
    median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
    if plot:
        plot_pr(pr_scores[median], name, "01", precisions[median], recalls[median], label=name)
        summary = (np.mean(scores), np.std(scores), np.mean(pr_scores), np.std(pr_scores))
        print "%.3f\t%.3f\t%.3f\t%.3f\t" % summary
        
    
train_model(create_ngram_model, npX, npY, name="pos vs neg", plot=True)
    
```

実行結果として
`0.734	0.047	0.893	0.023`
と以下の図を得た． Accuracy が 0.734 で，0.893 となっているのが Precision Recall curve 曲線の下側の青い面積 (AUC=Average Precision) である．単純な素性でありながら，73% 当てることができている(データはpos negが1:1くらい)．

[f:id:shinkanouchi:20150615103446p:plain:w300]

<br>
###3.2. 分類器のパラメータ調整
以下のパラメータを調節することができる．

-  TfidfVectorizer
    - NGram
        - 1gram (1,1)
        - 2gram (1,2)
        - 3gram (1,3)
    -  min_df
        - 1 or 2 
    - TF-IDFにおけるIDFの影響を検証するため，use_idfとsmooth_idfを試す
    - ストップワードを用いるかどうか
    - 単語の頻度(sublinear_tf)について対数を用いるかどうか
    - 記録する対象を単語の出現回数(頻度)か単語の出現の有無にするか

- MultinomialNB
    - スムージングについて検証
    - ラプラス・スムージング: 1
    - Lidstone スムージング: 0.01，0.05，0.1，0.5
    - スムージングなし: 0

- 正確にやろうと思えば，全ての組み合わせで訓練を行う必要がある
    - 3・2・2・2・2・2・2・6 = 1152 通りのパラメータの組み合わせ
- GridSearchCV
    - 最適なパラメータ値を設定する専用のクラス
    - 推定器と候補となるパラメータを辞書型で受け取る

```
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score


def create_ngram_model(params=None):
    tfidf_ngrams = TfidfVectorizer(ngram_range=(1, 3),
                                   analyzer="word", binary=False)
    clf = MultinomialNB()
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
    if params:
        pipeline.set_params(**params)
    return pipeline


def grid_search_model(clf_factory, X, Y):
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)
    param_grid = dict(vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                      vect__min_df=[1, 2],
                      vect__stop_words=[None, "english"],
                      vect__smooth_idf=[False, True],
                      vect__use_idf=[False, True],
                      vect__sublinear_tf=[False, True],
                      vect__binary=[False, True],
                      clf__alpha=[0, 0.01, 0.05, 0.1, 0.5, 1],
                      )
    grid_search = GridSearchCV(clf_factory(),
                               param_grid=param_grid,
                               cv=cv,
                               score_func=f1_score,
                               verbose=10)
    grid_search.fit(X, Y)
    clf = grid_search.best_estimator_
    print clf
    return clf


def train_model(clf_factory, X, Y, name="NB ngram", plot=False):
    cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)
    train_errors = []
    test_errors = []
    scores = []
    pr_scores = []
    precisions, recalls, thresholds = [], [], []
    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]
        clf = clf_factory
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)
        scores.append(test_score)
        proba = clf.predict_proba(X_test)
        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, 1])
        precision, recall, pr_thresholds = precision_recall_curve(y_test, proba[:, 1])
        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)
    summary = (np.mean(scores), np.std(scores),
               np.mean(pr_scores), np.std(pr_scores))
    print "%.3f\t%.3f\t%.3f\t%.3f\t" % summary


def get_best_model():
    best_params = dict(vect__ngram_range=(1, 2),
                       vect__min_df=1,
                       vect__stop_words=None,
                       vect__smooth_idf=False,
                       vect__use_idf=False,
                       vect__sublinear_tf=True,
                       vect__binary=False,
                       clf__alpha=0,
                       )
    best_clf = create_ngram_model(best_params)
    return best_clf


best_clf = grid_search_model(create_ngram_model, npX, npY)
train_model(get_best_model(), npX, npY, name="pos vs neg")
train_model(best_clf, npX, npY, name="pos vs neg2")
```


> [Parallel(n_jobs=1)]: Done 11520 out of 11520 | elapsed:  9.3min finished
> Pipeline(steps=[('vect', TfidfVectorizer(analyzer='word', binary=False, decode_error=u'strict',
>        dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
>        lowercase=True, max_df=1.0, max_features=None, min_df=2,
>        ngram_range=(1, 1), norm=u'l2', preprocessor=None, smooth_idf=True,...ue,
>        vocabulary=None)), ('clf', MultinomialNB(alpha=0.05, class_prior=None, fit_prior=True))])

そしてそのパラメータで精度を測定すると，`0.838	0.017	0.890	0.023`となり，
パラメータを最適化することで，Accuracy が10.4ポイント上昇している．

<br>
###3.3. ツイートを前処理する
- 文字列を全部小文字化

> tweet = tweet.lower()

- 顔文字を，辞書で制御

> emo_repl = {":)": " good ", ";)": " good ",  ":(": " bad ",  ":S": " bad "}

- 略語を正規表現で統一

> re_repl = { r"\br\b": "are", r"\bu\b": "you", r"\bhaha\b": "ha" }


###3.4. 品詞を考える
- 今までは Bag of Words しか考えていなかった
- 直感的には
    - 感情が含まれないツイートには「名詞」の割合が多く
    - 感情を含んだツイートには「形容詞」や「動詞」が多く含まれそう
- nltk.pos_tag() を使うことで品詞を考慮

###3.5. SentiWordNet を活用する
- 多くの英単語について「pos」or「neg」の度合いがスコア付けされている
- 13MB のファイル

###3.6.まとめ
- ツイートに対する感情分類器を作成した
- ナイーブベイズについて学んだ
- scikit-learnを使った
- 素性を増やしてチューニングすることの大切さがわかった
