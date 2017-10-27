# AlphaGo Zero

# 前言
1997年5月 IBM 的 "Deep Blue" 超級電腦擊敗了西洋棋世界冠軍卡斯巴羅夫。此刻電腦西洋棋正式宣佈『任務達成』。
傳統桌遊中就只剩圍棋是電腦無法勝任的，那時很多專家以為電腦可能要好幾十年後才能打敗人類。
大約20年後，Google 的 DeepMind 團隊推出了所謂的AlphaGo 電腦圍棋程式，在2016的三月打4-1敗了韓國圍棋大師李世乭。
但DeepMind團隊覺得這樣還不夠好，近一步的研發出 AlphaGo Master，在網路上 50-0 打敗了各國的圍棋專家、五月又 3-0 擊敗世界冠軍柯潔。

事實上 AlphaGo Master 的框架已經跟一開始的 AlphaGo 差很多，但 DeepMind 那時還未發佈相關的技術論文。

這個月 DeepMind 終於發佈了新論文，介紹 AlphaGo 的最終版本 AlphaGo Zero。 AlphaGo Zero 不但比 AlphaGo Master 強很多（100局中有89局獲勝），更驚訝的是它完全沒有 用任何人類的圍棋知識！ 意思是，訓練資料中沒有使用人類的圍棋棋譜或戰術！

原來，已是棋王的AlphaGo欲脫俗成仙，條件是先移除人類的沾污。

---

# 背景
根據賽局的分類，像是圍棋或西洋棋的棋都屬於零和、完美資訊、序列、離散、非合作博弈。 
也就是說，有兩個互相競爭的玩家，會輪流從有限集合選下棋動作，一方獲勝就是令方輸，兩方都看得到整個棋盤，沒有任何隱藏因素會影響你下棋位址對棋盤的改動，而且棋盤狀態能完全決定獲勝與否。

理論上，這類的遊戲的完美解可以用所謂的 min-max search。min-max search不能實際運用因此就有 alpha-beta search + iterative pruning。想法就是減少搜尋書需要被探索的地方。

2006 Remi Coultan https://hal.inria.fr/inria-00116992/document 對電腦圍棋搜尋樹太龐大的問題提出了一半的解答，Monte-carlo Tree-search（MCTS）。

2016 AlphaGo 也是利用MCTS，但也引用早期解stochastic 遊戲的用大量的self-play訓練一個模型的概念，和近年的深度學習。這三東西就使得電腦圍棋突破打敗了人類。

2017的AlphaGo Zero一樣就是MCTS + self-play RL + deep neural network，但整個架構乾淨許多。接下來我們來看看 AlphaGo Zero的架構。

---

# 重點
持續自己和自己下棋（self-Play）。每局中的每一輪，用類神經網路主導的MCTS搜尋選這輪的棋布。類神經網路持續的做訓練，使得它輸出的動作機率（move/action probilities）要近似前面下棋中的MCTS動作機率，還有輸出的狀態價值（value estimates）近似下棋中遇到的狀態是否最後導致獲勝或失敗。

----

* Monte-Carlo 樹搜尋
樹裡有『節點』和『邊』，節點為狀態（棋盤狀態），邊為動作。某節點的邊就可以對應獨一的狀態動作對（state-action pair）。

樹搜尋的目的是從根節點往下長出一棵搜尋樹。當一個節點被探索時，它可以說是被『展開』，他的邊會初始化某些統計，也把這些邊指向的節點加入樹。樹的葉節點就是所有未被展開的

----

* Steps:
    * **選擇新節點**: 從根節點用開始走一條樹徑，直到你遇到新的節點（葉節點）。走路的方式就是在節點用某個規則選該走得邊，則選新節點這步等於：選邊，走到下一個節點，選邊，走到下一個節點 ... 遇到新的節點
        * 走路時的『某個規則』為UCT (Upper Confidence Bound (UCB) for Trees)：
            * 選動作（邊）$a = \mathrm{argmax}_a Q_{(s_t, a)} + U_{(s_t, a)}$
                * Q 為邊上的一個統計值，是這邊底下的子樹的平均價值。各個邊的 Q 會在搜尋的過程中被跟新，初始化時是 0。
                * U為所謂的「信賴上界」，代表我們對目前邊上Q值的信賴。
$U_{(s, a)} = c_{puct} P_{(s, a)} \frac{\sqrt{\sum_b N_{(s, b)}} }{1 + N_{(s, a)}}$
這裡N為這邊之前被選過幾次，及為走訪次數。走越多的邊信賴上線就越小。我們可以把U想成是一個各邊有的加分條件，讓比較沒走訪過得邊有加分條件，這樣比較容易被選到。
        * Virtual loss to discourage redundant exploration
            * During the traversal, the node statistics are temporarily updated with a large negative value N, W as if this path leads to large number of lost games. This *virtual loss* is removed in the backup.
    * **用類神經做評估**: 走到新的節點 $s_L$ 後就要拿給類神經網路做評估。類神經網路回傳的是這狀態的各個動作的動作機率以及這狀態 $\mathbf{p}$ 的價值估計$v$。
        * **邊值初始化（樹的擴充）**: 用$\mathbf{p}$初始化 $s_L$ 所有的邊
            * For all actions $a$ possible, initialize at edge ($s_L, a$) the values, N = 0, W, Q, and P 
                * N=0: 走訪次數
                * W=0: total state-action value in sub-tree
                * Q=0: mean state-action value in sub-tree
                * P=$p_a$ + [$\mathrm{Dir}$ if $s_L$ is root]: stored *prior probability* of taking action $a$ at state $s_L$
                    * add Dirichlet loss to root node to encourage exploration
        * **更新樹邊的統計（反向傳播）**: 
            * 用$v$更新根節點到$s_L$路徑各邊 ($s_t$, $a_t$) 的統計值 
                * $W_{(s_t, a_t)} = W_{(s_t, a_t)} + v$
                * $N_{(s_t, a_t)} = N_{(s_t, a_t)} + 1$
                * $Q_{(s_t, a_t)} = \frac{W_{(s_t, a_t})}{N_({s_t, a_t})}$

---

* 選下一步棋: MCTS做到一定程度後（論文是固定1600個迴圈），就可以用算出來的統計值下這一步的棋。假設這一盤現在的狀態為 $s_0$，也就是 MCTS 的根節點那：
    * 在 $s_0$ 的狀態下，選棋步a的機率分佈為 $\pi(a | s_0 ) = \frac{ {N_{( s_0 , a )}}^{\frac{1}{\tau}} }{ \sum_{b \in A_{s_0}}\  {{N_{ (s_0 , b)} }}^{\frac{1}{\tau}}  }$ 
        * $\gamma$ 為一個『溫度』常數。溫度月低，這機率分佈會集中在N最高的a。
    * 論到下一步棋時，新的搜尋開始不需要從無開始，可以沿用對應到下的棋步的子樹。

---

* 訓練類神經網路:
    * 過去自己打自己的每一局的每一布可以當作一筆訓練資料 $(s_t, \pi_t, z_t)$
        * $s_t$: 當時盤狀
        * $\pi_t$: 當時MCTS後的動作機率值
        * $z_t$: 1或-1，如果這步的玩家也是最後這局的贏家則1，否則 -1
    * 抽幾筆 $(s_t, \pi_t, z_t)$ 當作一個訓練batch。類神經網路是同時估計策略和價值，策略的部份就是靠 $\pi_t$ 訓練，價值的部份就是靠 $z_t$ 訓練。
        * Loss function: mean-squared error 和 cross-entropy loss 的合（論文是選擇平等加權）。
For single example from time-step $t$, $(s_t , \pi_t , z_t )$, $\mathcal{L} = (z_t-v)^2 + (-\mathbf{\pi_t}^{\intercal}\mathrm{log}\mathbf{p}) + c\|\theta\|^2$
$(\mathbf{p}, v) = f_\theta(s_t)$, $c$ 為 L2-reguglarization的常數

---

## 策略迭代（Policy Iteration）

![](http://incompleteideas.net/sutton/book/ebook/figtmp18.png)
<sup>來源：http://incompleteideas.net/sutton/book/ebook/node46.html</sup>

普通的策略迭代演算法如上圖片所示，利用目前的策略函數去算出一個新的價值函數（策略評估），再用用新的價值函數導出新的策略函數（策略改進），這樣交互作用使得兩函數趨近於最佳。

AlphaGo Zero 自己對自己的學習演算法可以視為一種近似策略迭代的演算法，這裡用MCTS的計算和類神經網路的學習來做一種間接的策略評估和策略改進。
![](https://i.imgur.com/yluUf60.png)

## References
* [DeepMind AlphaGo Zero 網誌](https://deepmind.com/blog/alphago-zero-learning-scratch/)
"AlphaGo Zero: Learning from scratch"
DeepMind Blog Post

* [AlphaGo Zero paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ) 
Silver, D. et al. *Mastering the game of Go without
human knowledge* Nature. (2017)
* [原AlphaGo paper](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
Silver, D. et al. *Mastering the game of Go with deep neural networks and tree
search* Nature. (2016)
* [Paper discussing using MCTS in Go](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/grand-challenge.pdf)
Gelly, S. et al. *The Grand Challenge of Computer Go:
Monte Carlo Tree Search and Extensions* Communications of the ACM. (2012)
* [MCTS survey](https://www.researchgate.net/publication/235985858_A_Survey_of_Monte_Carlo_Tree_Search_Methods)
Browne, C. et al. *A Survey of Monte Carlo Tree Search Methods* IEEE Transactions on Computational Intelligence and AI in Games. (2012)