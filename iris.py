{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN2e6NKDh/CjsbNxznIr1tq",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rrejinrejinr/iris-dataset/blob/main/iris.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "2zknW33_AJcY"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.datasets import load_iris\n",
        "%matplotlib inline\n",
        "import seaborn as sns\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "iris=load_iris()\n",
        "x=iris.data\n",
        "y=iris.target\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)\n",
        "y_test"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cC-b01q_B5zm",
        "outputId": "5b4d955a-b6da-419e-bfa0-8af57238ad51"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1, 2, 0, 2, 0, 2, 1, 2, 1, 2, 1, 1, 0, 2, 1, 2, 2, 2, 1, 1, 1, 2,\n",
              "       1, 0, 1, 1, 0, 2, 2, 2])"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.svm import SVC\n",
        "model=SVC()\n",
        "model.fit(X_train,y_train)\n",
        "predict=model.predict(X_test)\n",
        "from sklearn.metrics import accuracy_score\n",
        "print(accuracy_score(y_test,predict)*100)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9r7wsGw6Y0Jd",
        "outputId": "f466fd66-a9f6-4f88-8a61-c6806c4ef440"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "90.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "model1=LogisticRegression(max_iter=100)\n",
        "model1.fit(X_train,y_train)\n",
        "predict1=model1.predict(X_test)\n",
        "predict1\n",
        "from sklearn.metrics import accuracy_score\n",
        "print(accuracy_score(y_test,predict)*100)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1AecbeBUE78s",
        "outputId": "9535e9a5-3c0d-4bef-85a4-61a1b4e9cc28"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "90.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "model2=DecisionTreeClassifier()\n",
        "model2.fit(X_train,y_train)\n",
        "predict3=model1.predict(X_test)\n",
        "from sklearn.metrics import accuracy_score\n",
        "print(accuracy_score(y_test,predict3)*100)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eP3sbmr3FG7C",
        "outputId": "c26d65c3-e227-4b89-d7f5-cf1add44eeef"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "93.33333333333333\n"
          ]
        }
      ]
    }
  ]
}
