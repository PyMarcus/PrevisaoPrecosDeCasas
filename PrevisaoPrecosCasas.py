import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

boston = load_boston()
df = pd.DataFrame(boston.data)


class Crawler:
    """
    classe responsável por conter os métodos que buscarão os dados
    """
    def __init__(self, url):
        self.url = url
        self.bs = ''

    def parseHtml(self):
        try:
            req = urlopen(self.url)
        except HTTPError:
            print("HTTP ERROR, página, possivelmente, indisponível")
        except URLError:
            print("URL INCORRETA")
        else:
            self.bs = BeautifulSoup(req.read(), 'html.parser')
            return self.bs

    def escreve_arquivo(self):
        with open('./dataset.csv', 'w') as f:
            arquivo = f.writelines(self.bs)


class DataSet:
    def data(self):
        global  boston, df
        print("SHAPE DO DATASET:")
        print(boston.data.shape)
        print("DESCRIÇÃO DO DATASET:")
        print(boston.DESCR)
        df.columns = boston.feature_names # altera-se o nome das colunas

        # array com preços das casas
        print(f"PREÇOS {boston.target}")

        # incluir precos no dataframe
        df['PRECO'] = boston.target


class Modelo:
    def modelo(self):
        """
        variaveis independetes ou explanatórias(características da casa)
        variável dependente (preço)
        :return:
        """
        x = df.drop('PRECO', axis=1) # exclui a coluna preço
        y = df.PRECO

        plt.scatter(df.RM, y) # numero de quartos e vetor de preço
        plt.xlabel("Média do número de quartos por casas")
        plt.ylabel("Preço da casa")
        plt.title("Relação entre número de quartos e preços")
        plt.show() # ascendência positiva, aumento de quartos gera aumento do preço

        linear_regre = LinearRegression()
        linear_regre.fit(x, y)


        # comparação entre preço original e previsto
        plt.scatter(df.PRECO, linear_regre.predict(x))
        plt.xlabel("PREÇO ORIGINAL")
        plt.ylabel("PREÇO PREVISTO")
        plt.title("PREÇO ORIGINAL x PREÇO PREVISTO")
        plt.show()


        #preco previsto
        print("PREÇOS PREVISTOS\n", linear_regre.predict(x))


if __name__ == '__main__':
    url = 'http://lib.stat.cmu.edu/datasets/boston'
    #crawler = Crawler(url)
    #crawler.parseHtml()
    #crawler.escreve_arquivo()
    dataSet = DataSet()
    dataSet.data()
    modelo = Modelo()
    modelo.modelo()
