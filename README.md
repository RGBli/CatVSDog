# DogsVsCats
---
Kaggle 竞赛题猫狗大战 ，用于 PyTorch 入门

</br>

### 运行步骤
1）训练
``` shell
python3 train.py
```

2）测试
```shell
python3 test.py
```

3）查看 tensorboard
```shell
cd CatVSDog
tensorboard --logdir=log
```

</br>

### 分类效果
使用代码中的超参数在 Kaggle 的训练集训练出的猫狗分类模型在测试集中达到 97.2% 的预测准确率，如果进一步调参相信会有更好的表现。

数据集链接：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

</br>


### 参考文章
https://github.com/xbliuHNU/DogsVsCats
https://github.com/espectre/Kaggle-Dogs_vs_Cats_PyTorch