Wstęp.
======

Analiza nastrojów jest trudnym przedmiotem uczenia się maszyn. Ludzie wyrażają swoje emocje w języku, który jest często zasłonięty przez sarkazm, wieloznaczność i grę słów, co może być bardzo mylące dla ludzi i komputerów. Ten projekt jest przykładem analizy nastrojów dla recenzji filmowych i ma na celu przykładowe użycie modelów BagOfWords oraz Word2Vec do analizy nastrojów.
- Część 1. Basic Natural Language Processing: przykład użycia modelu BagOfWords do analizy nastrojów.
- Część 2. Deep Learning for Text Understanding: przykład trenowania i użycia modelu Word2Vec oraz wykorzystanie wynikajacych wektorów słow do analizy nastrojów.



Narzędzia.
==========

Projekt ten jest realizowany w języku programowania ``Python 3.5.2``, przy użyciu podstawowych technik NLP na przykładzie biblioteki ``nltk`` oraz algorytmów klasyfikacji na przykładzie biblioteki ``scikit-learn``, takich jak:
- Random Forest
- Naive Bayes Gaussian
- Naive Bayes Multinomial
- Naive Bayes Bernoulli
- k-Nearest Neighbors
Aby użyc modelu BagOfWords również korzystamy się z biblioteki ``scikit-learn``. Aby użyć modelu Word2Vec używamy bibliotekę ``gensim``. W celu przeszkolenia modelu Word2Vec w rozsądnym czasie, będziemy musieli zainstalować paczkę ``cython``. Word2Vec będzie działać bez ``cython``, ale będzie to trwało dużo więcej czasu. Do czyszczenia tekstu od tagów HTML, została użyta biblioteka ``Beutiful Soup``. Po zakończeniu wszystkich obliczeń, w celu zapisywania wyników do pliku, używamy pakietu ``pandas``. Jako pomocniczą bibliotekę do pracy z tablicami użyto paczkę ``numpy``.



Dane.
=====

Został użyty zbiór 25000 recenzji filmów do trenowania i 25000 w celu przetestowania. Dane zostały wzięte z następującej publikacji: http://ai.stanford.edu/~amaas/data/sentiment/



Teoria.
=======

``bag-of-words.py``
Model BagOfWords jest uproszczeniem reprezentacji tekstu, wykorzystywanym w przetwarzaniu języka naturalnego i wyszukiwaniu informacji (Information Retrieval). Tekst jest reprezentowany jako zbiór słów, nie zważając na gramatykę, a nawet kolejność słów ale utrzymując ich wielość.

``word-2-vec-average-vectors.py``
Google Word2Vec jest metodą nauki, która koncentruje się na rozumieniu słów. Word2Vec próbuje zrozumieć sens i semantyczne relacje między słowami. Działa w sposób, podobny do głębokich metod, takich jak sieci neuronowe. Word2Vec nie wymaga etykiety w celu tworzenia znaczących wyników. Jest to przydatne, ponieważ większość danych w rzeczywistym świecie nie ma etykiety. Słowa o podobnym znaczeniu zapisują się w zbiór, dzieki czemu można odtworzyć niektóre relacje słowne, takie jak np. analogii czy synonimy.



Uruchomienie projektu.
======================

Najpierw musimy stworzyć wirtualne środowisko (``virtualenvwrapper``) i zainstalować wszystkie zależności z pliku ``requirements.txt``:
```
$ mkvirtualenv -p /usr/bin/python3 virualenv_name
$ workon virualenv_name
$ pip install -r requirements.txt
```

Aby uruchomić model BagOfWords musimy wpisać:
```
$ python /path/to/project/bag-of.words.py
```

Aby uruchomić model Word2Vec musimy wpisać:
```
$ python /path/to/project/word-2-vec-average-vectors.py
```


Opis eksperymentów.
===================

``bag-of-words.py``
- czytanie danych do trenowania i testowania
- czyszczenie danych do trenowania i testowania
    - usunięcie znaczników HTML (``BeautifulSoup``)
    - usunięcie znaków interpunkcyjnych oraz cyfr (``Regularne wyrażenia``)
    - konwertacja na małe litery
    - podział na poszczególne słowa (nazywane "atomizacja" w ``NLP`` żargonie)
    - usunięcie stop-słów (``NLTK``)
- stworzenie modelu BagOfWords (przy użyciu ``CountVectorizer`` z ``scikit-learn``)
- trenowanie klasyfikatorów
- dokonanie prognozy
- obliczanie dokładności prognozy i zapisywanie wyników


``word-2-vec-average-vectors.py``
- czytanie danych dla treningu modelu Word2Vec
- czyszczenie danych dla treningu modelu Word2Vec
    - usunięcie znaczników HTML (``BeautifulSoup``)
    - podział tekstu na lista zdań (``NLTK``)
    - usunięcie znaków interpunkcyjnych oraz cyfr (``Regularne wyrażenia``)
    - konwertacja na małe litery
    - podział na poszczególne słowa (nazywane "atomizacja" w ``NLP`` żargonie)
- spłaszczenie Lista recenzji gdzie każda recenzja jest lista zdań, gdzie każde zdanie jest listą słów do listy zdań gdzie każde zdanie jest listą słów
- stworzenie, trenowanie i zapisanie modelu Word2Vec
- czytanie danych do trenowania i testowania
- czyszczenie danych do trenowania i testowania
    - usunięcie znaczników HTML (``BeautifulSoup``)
    - usunięcie znaków interpunkcyjnych oraz cyfr (``Regularne wyrażenia``)
    - konwertacja na małe litery
    - podział na poszczególne słowa (nazywane "atomizacja" w ``NLP`` żargonie)
    - usunięcie stop-słów (``NLTK``)
- stworzenie wektorów słów z danych testowych i treningowych (przy użyciu modelu Word2Vec)
- trenowanie klasyfikatorów przy użyciu utworzonego w poprzednim kroku treningowych wektoru słów
- dokonanie prognozy przy użyciu testowych wektorów słów
- obliczanie dokładności prognozy i zapisywanie wyników



Pliki.
======
``classifiers/*`` - klasyfikatory
``README-PL.md`` - opis projektu w języku polskim
``README.md`` - opis projektu w języku angielskim
``AclImdb_v1.tar.gz`` - zbiór danych z recenzjami
``bag-of-words-summary.txt`` - wyniki algorytmu BagOfWords
``bag-of-words.py`` - algorytm BagOfWords
``config.py`` - konfiguracja projektu (ścieżki do zbioru danych z recenzjami)
``parsers.py`` - parsery tekstowe
``requirements.txt`` - zależności
``utils.py`` - funkcje pomocnicze
``word-2-vec-average-vectors-summary.txt`` - wyniki algorytmu Word2Vec
``word-2-vec-average-vectors.py`` - algorytm Word2Vec



Wyniki.
=======

Wyniki konkretnego klasyfikatora dla BagOfWords są przechowywane w pliku:
``bag-of-words-*-model.csv``

Podsumowanie wyników dla BagOfWords są przechowywane w pliku:
``bag-of-words-summary.txt``

Wyniki konkretnego klasyfikatora dla Word2Vec są przechowywane w pliku:
``word-2-vec-average-vectors-*-model.csv``

Podsumowanie wyników dla Word2Vec są przechowywane w pliku:
``word-2-vec-average-vectors-summary.txt``



Interpretacja wyników.
======================

Jak widzimy BagOfWords daje lepsze wyniki niż Word2Vec. Głównym powodem jest uśrednienie wektorów i wykorzystanie centroidy, prze co gubi się kolejność słów, dzięki czemu model Word2Vec jest bardzo podobnym do koncepcji modelu BagOfWords. Fakt, że wyniki są podobne (w zasięgu błąd standardowy) sprawia, że ​obydwa sposoby w praktyce są równoważne.

Szkolenie Word2Vec na dużo więcej tekstu powinno znacznie poprawić wydajność. Wyniki Google są oparte na wektorach słownych, które nauczyły się z ponad miliarda słów. Dodatkowo, Word2Vec zapewnia funkcje załadowania z góry przeszkolonego modelu, który jest wyjściem oryginalnego narzędzia Google C, więc jest to również możliwe, aby trenować model w C, a następnie zaimportować je do Pythona.



Materiały.
==========

https://www.kaggle.com/c/word2vec-nlp-tutorial
