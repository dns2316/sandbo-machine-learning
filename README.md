
tutorial from http://mlbootcamp.ru/article/tutorial/


```python
import numpy as np
import pandas as pd
# from pd.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
data = pd.read_csv(url, header=None, na_values='?')
```

    Using matplotlib backend: TkAgg



```python
data.shape
```




    (690, 16)




```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>30.83</td>
      <td>0.000</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.25</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>202.0</td>
      <td>0</td>
      <td>+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>3.04</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
      <td>f</td>
      <td>g</td>
      <td>43.0</td>
      <td>560</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>1.50</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>280.0</td>
      <td>824</td>
      <td>+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>3.75</td>
      <td>t</td>
      <td>t</td>
      <td>5</td>
      <td>t</td>
      <td>g</td>
      <td>100.0</td>
      <td>3</td>
      <td>+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.71</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>s</td>
      <td>120.0</td>
      <td>0</td>
      <td>+</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>685</th>
      <td>b</td>
      <td>21.08</td>
      <td>10.085</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>1.25</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>260.0</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>686</th>
      <td>a</td>
      <td>22.67</td>
      <td>0.750</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>2.00</td>
      <td>f</td>
      <td>t</td>
      <td>2</td>
      <td>t</td>
      <td>g</td>
      <td>200.0</td>
      <td>394</td>
      <td>-</td>
    </tr>
    <tr>
      <th>687</th>
      <td>a</td>
      <td>25.25</td>
      <td>13.500</td>
      <td>y</td>
      <td>p</td>
      <td>ff</td>
      <td>ff</td>
      <td>2.00</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>t</td>
      <td>g</td>
      <td>200.0</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>688</th>
      <td>b</td>
      <td>17.92</td>
      <td>0.205</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.04</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>280.0</td>
      <td>750</td>
      <td>-</td>
    </tr>
    <tr>
      <th>689</th>
      <td>b</td>
      <td>35.00</td>
      <td>3.375</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>h</td>
      <td>8.29</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>0.0</td>
      <td>0</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A1</th>
      <th>A2</th>
      <th>A3</th>
      <th>A4</th>
      <th>A5</th>
      <th>A6</th>
      <th>A7</th>
      <th>A8</th>
      <th>A9</th>
      <th>A10</th>
      <th>A11</th>
      <th>A12</th>
      <th>A13</th>
      <th>A14</th>
      <th>A15</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>30.83</td>
      <td>0.000</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.25</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>202.0</td>
      <td>0</td>
      <td>+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>3.04</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
      <td>f</td>
      <td>g</td>
      <td>43.0</td>
      <td>560</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>1.50</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>280.0</td>
      <td>824</td>
      <td>+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>3.75</td>
      <td>t</td>
      <td>t</td>
      <td>5</td>
      <td>t</td>
      <td>g</td>
      <td>100.0</td>
      <td>3</td>
      <td>+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.71</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>s</td>
      <td>120.0</td>
      <td>0</td>
      <td>+</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['A5'][687]
```




    'p'




```python
data.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A2</th>
      <th>A3</th>
      <th>A8</th>
      <th>A11</th>
      <th>A14</th>
      <th>A15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>678.000000</td>
      <td>690.000000</td>
      <td>690.000000</td>
      <td>690.00000</td>
      <td>677.000000</td>
      <td>690.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>31.568171</td>
      <td>4.758725</td>
      <td>2.223406</td>
      <td>2.40000</td>
      <td>184.014771</td>
      <td>1017.385507</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.957862</td>
      <td>4.978163</td>
      <td>3.346513</td>
      <td>4.86294</td>
      <td>173.806768</td>
      <td>5210.102598</td>
    </tr>
    <tr>
      <th>min</th>
      <td>13.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>22.602500</td>
      <td>1.000000</td>
      <td>0.165000</td>
      <td>0.00000</td>
      <td>75.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.460000</td>
      <td>2.750000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>160.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>38.230000</td>
      <td>7.207500</td>
      <td>2.625000</td>
      <td>3.00000</td>
      <td>276.000000</td>
      <td>395.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.250000</td>
      <td>28.000000</td>
      <td>28.500000</td>
      <td>67.00000</td>
      <td>2000.000000</td>
      <td>100000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print(categorical_columns)
print(numerical_columns)
```

    ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13', 'class']
    ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']



```python
data[categorical_columns].describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A1</th>
      <th>A4</th>
      <th>A5</th>
      <th>A6</th>
      <th>A7</th>
      <th>A9</th>
      <th>A10</th>
      <th>A12</th>
      <th>A13</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>678</td>
      <td>684</td>
      <td>684</td>
      <td>681</td>
      <td>681</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>b</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>t</td>
      <td>f</td>
      <td>f</td>
      <td>g</td>
      <td>-</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>468</td>
      <td>519</td>
      <td>519</td>
      <td>137</td>
      <td>399</td>
      <td>361</td>
      <td>395</td>
      <td>374</td>
      <td>625</td>
      <td>383</td>
    </tr>
  </tbody>
</table>
</div>




```python
for c in categorical_columns:
    print(data[c].unique())
```

    ['b' 'a' nan]
    ['u' 'y' nan 'l']
    ['g' 'p' nan 'gg']
    ['w' 'q' 'm' 'r' 'cc' 'k' 'c' 'd' 'x' 'i' 'e' 'aa' 'ff' 'j' nan]
    ['v' 'h' 'bb' 'ff' 'j' 'z' nan 'o' 'dd' 'n']
    ['t' 'f']
    ['t' 'f']
    ['f' 't']
    ['g' 's' 'p']
    ['+' '-']



```python
pd.plotting.scatter_matrix(data, alpha=0.05, figsize=(10, 10));data.corr()
```


```python
data.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>7</th>
      <th>10</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.202317</td>
      <td>0.395751</td>
      <td>0.185912</td>
      <td>-0.079812</td>
      <td>0.018553</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.202317</td>
      <td>1.000000</td>
      <td>0.298902</td>
      <td>0.271207</td>
      <td>-0.224242</td>
      <td>0.123121</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.395751</td>
      <td>0.298902</td>
      <td>1.000000</td>
      <td>0.322330</td>
      <td>-0.077163</td>
      <td>0.051345</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.185912</td>
      <td>0.271207</td>
      <td>0.322330</td>
      <td>1.000000</td>
      <td>-0.120096</td>
      <td>0.063692</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.079812</td>
      <td>-0.224242</td>
      <td>-0.077163</td>
      <td>-0.120096</td>
      <td>1.000000</td>
      <td>0.066853</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.018553</td>
      <td>0.123121</td>
      <td>0.051345</td>
      <td>0.063692</td>
      <td>0.066853</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
col1 = "A2"
col2 = "A11"

plt.figure(figsize=(10, 6))

plt.scatter(data[col1][data['class'] == '+'],
            data[col2][data['class'] == '+'],
            alpha=0.75,
            color='red',
            label='+')

plt.scatter(data[col1][data['class'] == '-'],
            data[col2][data['class'] == '-'],
            alpha=0.75,
            color='blue',
            label='-')

plt.xlabel(col1)
plt.ylabel(col2)
plt.legend(loc='best')
```




    <matplotlib.legend.Legend at 0x7f302966ca90>




```python
data.count(axis=0) # el without empty
```




    A1       678
    A2       678
    A3       690
    A4       684
    A5       684
    A6       681
    A7       681
    A8       690
    A9       690
    A10      690
    A11      690
    A12      690
    A13      690
    A14      677
    A15      690
    class    690
    dtype: int64




```python
data = data.fillna(data.median(axis=0), axis=0) # fill empty el
```


```python
data.count(axis=0)
```




    A1       678
    A2       690
    A3       690
    A4       684
    A5       684
    A6       681
    A7       681
    A8       690
    A9       690
    A10      690
    A11      690
    A12      690
    A13      690
    A14      690
    A15      690
    class    690
    dtype: int64




```python
data['A1'].describe()
```




    count     678
    unique      2
    top         b
    freq      468
    Name: A1, dtype: object




```python
data['A1'] = data['A1'].fillna('b') # fill empty el top value
```


```python
data_describe = data.describe(include=[object]) # fill all table empty el top value
for c in categorical_columns:
    data[c] = data[c].fillna(data_describe[c]['top'])
```


```python
data.describe(include=[object])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A1</th>
      <th>A4</th>
      <th>A5</th>
      <th>A6</th>
      <th>A7</th>
      <th>A9</th>
      <th>A10</th>
      <th>A12</th>
      <th>A13</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>b</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>t</td>
      <td>f</td>
      <td>f</td>
      <td>g</td>
      <td>-</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>480</td>
      <td>525</td>
      <td>525</td>
      <td>146</td>
      <td>408</td>
      <td>361</td>
      <td>395</td>
      <td>374</td>
      <td>625</td>
      <td>383</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A2</th>
      <th>A3</th>
      <th>A8</th>
      <th>A11</th>
      <th>A14</th>
      <th>A15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>690.000000</td>
      <td>690.000000</td>
      <td>690.000000</td>
      <td>690.00000</td>
      <td>690.000000</td>
      <td>690.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>31.514116</td>
      <td>4.758725</td>
      <td>2.223406</td>
      <td>2.40000</td>
      <td>183.562319</td>
      <td>1017.385507</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.860245</td>
      <td>4.978163</td>
      <td>3.346513</td>
      <td>4.86294</td>
      <td>172.190278</td>
      <td>5210.102598</td>
    </tr>
    <tr>
      <th>min</th>
      <td>13.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>22.670000</td>
      <td>1.000000</td>
      <td>0.165000</td>
      <td>0.00000</td>
      <td>80.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.460000</td>
      <td>2.750000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>160.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>37.707500</td>
      <td>7.207500</td>
      <td>2.625000</td>
      <td>3.00000</td>
      <td>272.000000</td>
      <td>395.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.250000</td>
      <td>28.000000</td>
      <td>28.500000</td>
      <td>67.00000</td>
      <td>2000.000000</td>
      <td>100000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print(binary_columns, nonbinary_columns)
```

    ['A1', 'A9', 'A10', 'A12', 'class'] ['A4', 'A5', 'A6', 'A7', 'A13']



```python
data.at[data['A1'] == 'b', 'A1'] = 0
data.at[data['A1'] == 'a', 'A1'] = 1
data['A1'].describe() # change True/False to 0/1
```




    count     690
    unique      2
    top         0
    freq      480
    Name: A1, dtype: int64




```python
for c in binary_columns[1:]:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1
```


```python
data[binary_columns].describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A1</th>
      <th>A9</th>
      <th>A10</th>
      <th>A12</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>480</td>
      <td>361</td>
      <td>395</td>
      <td>374</td>
      <td>383</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['A4'].unique()
```




    array(['u', 'y', 'l'], dtype=object)




```python
data_nonbinary = pd.get_dummies(data[nonbinary_columns])
print(data_nonbinary.columns)
```

    Index(['A4_l', 'A4_u', 'A4_y', 'A5_g', 'A5_gg', 'A5_p', 'A6_aa', 'A6_c',
           'A6_cc', 'A6_d', 'A6_e', 'A6_ff', 'A6_i', 'A6_j', 'A6_k', 'A6_m',
           'A6_q', 'A6_r', 'A6_w', 'A6_x', 'A7_bb', 'A7_dd', 'A7_ff', 'A7_h',
           'A7_j', 'A7_n', 'A7_o', 'A7_v', 'A7_z', 'A13_g', 'A13_p', 'A13_s'],
          dtype='object')



```python
data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
data_numerical.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A2</th>
      <th>A3</th>
      <th>A8</th>
      <th>A11</th>
      <th>A14</th>
      <th>A15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.900000e+02</td>
      <td>6.900000e+02</td>
      <td>6.900000e+02</td>
      <td>6.900000e+02</td>
      <td>6.900000e+02</td>
      <td>6.900000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-2.497197e-15</td>
      <td>1.956567e-16</td>
      <td>4.942906e-16</td>
      <td>1.029772e-17</td>
      <td>3.861645e-17</td>
      <td>-2.059544e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.497787e+00</td>
      <td>-9.559198e-01</td>
      <td>-6.643947e-01</td>
      <td>-4.935286e-01</td>
      <td>-1.066043e+00</td>
      <td>-1.952717e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.456942e-01</td>
      <td>-7.550425e-01</td>
      <td>-6.150897e-01</td>
      <td>-4.935286e-01</td>
      <td>-6.014412e-01</td>
      <td>-1.952717e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-2.575087e-01</td>
      <td>-4.035072e-01</td>
      <td>-3.655762e-01</td>
      <td>-4.935286e-01</td>
      <td>-1.368388e-01</td>
      <td>-1.943120e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.221970e-01</td>
      <td>4.919034e-01</td>
      <td>1.200038e-01</td>
      <td>1.233822e-01</td>
      <td>5.136044e-01</td>
      <td>-1.193615e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.109180e+00</td>
      <td>4.668645e+00</td>
      <td>7.851932e+00</td>
      <td>1.328414e+01</td>
      <td>1.054901e+01</td>
      <td>1.899821e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
print(data.shape)
print(data.columns)
```

    (690, 43)
    Index(['A2', 'A3', 'A8', 'A11', 'A14', 'A15', 'A1', 'A9', 'A10', 'A12',
           'class', 'A4_l', 'A4_u', 'A4_y', 'A5_g', 'A5_gg', 'A5_p', 'A6_aa',
           'A6_c', 'A6_cc', 'A6_d', 'A6_e', 'A6_ff', 'A6_i', 'A6_j', 'A6_k',
           'A6_m', 'A6_q', 'A6_r', 'A6_w', 'A6_x', 'A7_bb', 'A7_dd', 'A7_ff',
           'A7_h', 'A7_j', 'A7_n', 'A7_o', 'A7_v', 'A7_z', 'A13_g', 'A13_p',
           'A13_s'],
          dtype='object')



```python
X = data.drop(('class'), axis=1)  # drop column 'class'.
y = data['class']
feature_names = X.columns
print(feature_names)
```

    Index(['A2', 'A3', 'A8', 'A11', 'A14', 'A15', 'A1', 'A9', 'A10', 'A12', 'A4_l',
           'A4_u', 'A4_y', 'A5_g', 'A5_gg', 'A5_p', 'A6_aa', 'A6_c', 'A6_cc',
           'A6_d', 'A6_e', 'A6_ff', 'A6_i', 'A6_j', 'A6_k', 'A6_m', 'A6_q', 'A6_r',
           'A6_w', 'A6_x', 'A7_bb', 'A7_dd', 'A7_ff', 'A7_h', 'A7_j', 'A7_n',
           'A7_o', 'A7_v', 'A7_z', 'A13_g', 'A13_p', 'A13_s'],
          dtype='object')



```python
print(X.shape)
print(y.shape)
N, d = X.shape
```

    (690, 42)
    (690,)


make machine learning algorithm


```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)

N_train, _ = X_train.shape
N_test,  _ = X_test.shape
print(N_train, N_test)
```

    483 207



```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')




```python
y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

err_train = np.mean(y_train != y_train_predict)
err_test  = np.mean(y_test  != y_test_predict)
print(err_train, err_test) #error 16%!
```

    0.151138716356 0.164251207729



```python
from sklearn.grid_search import GridSearchCV
n_neighbors_array = [1, 3, 5, 7, 10, 15]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})
grid.fit(X_train, y_train)

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
print(best_cv_err, best_n_neighbors)
```

    0.20703933747412007 7


    /usr/local/lib/python3.5/dist-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)



```python
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)

err_train = np.mean(y_train != knn.predict(X_train))
err_test  = np.mean(y_test  != knn.predict(X_test))
print(err_train, err_test)
```

    0.151138716356 0.164251207729



```python
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print(err_train, err_test) # error 13%
```

    0.144927536232 0.130434782609



```python
from sklearn.grid_search import GridSearchCV
C_array = np.logspace(-3, 3, num=7)
gamma_array = np.logspace(-5, 2, num=8)
svc = SVC(kernel='rbf')
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
grid.fit(X_train, y_train)
print('CV error    = ', 1 - grid.best_score_)
print('best C      = ', grid.best_estimator_.C)
print('best gamma  = ', grid.best_estimator_.gamma)
```

    CV error    =  0.13871635610766042
    best C      =  1.0
    best gamma  =  0.01



```python
svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print(err_train, err_test) # error 11%!
```

    0.134575569358 0.111111111111



```python
from sklearn.grid_search import GridSearchCV
C_array = np.logspace(-3, 3, num=7)
svc = SVC(kernel='linear')
grid = GridSearchCV(svc, param_grid={'C': C_array})
grid.fit(X_train, y_train)
print('CV error    = ', 1 - grid.best_score_)
print('best C      = ', grid.best_estimator_.C)
```

    CV error    =  0.15113871635610765
    best C      =  0.1



```python
svc = SVC(kernel='linear', C=grid.best_estimator_.C)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print(err_train, err_test)
```

    0.151138716356 0.125603864734



```python
from sklearn.grid_search import GridSearchCV
C_array = np.logspace(-5, 2, num=8)
gamma_array = np.logspace(-5, 2, num=8)
degree_array = [2, 3, 4]
svc = SVC(kernel='poly')
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array, 'degree': degree_array})
grid.fit(X_train, y_train)
print('CV error    = ', 1 - grid.best_score_)
print('best C      = ', grid.best_estimator_.C)
print('best gamma  = ', grid.best_estimator_.gamma)
print('best degree = ', grid.best_estimator_.degree)
```

    CV error    =  0.13871635610766042
    best C      =  0.0001
    best gamma  =  10.0
    best degree =  2



```python
svc = SVC(kernel='poly', C=grid.best_estimator_.C, 
          gamma=grid.best_estimator_.gamma, degree=grid.best_estimator_.degree)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print(err_train, err_test)
```

    0.0973084886128 0.12077294686



```python
from sklearn import ensemble
# top algoritm RandomForest
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

err_train = np.mean(y_train != rf.predict(X_train))
err_test  = np.mean(y_test  != rf.predict(X_test))
print(err_train, err_test) # error 10%!
```

    0.0 0.101449275362



```python
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
```

    Feature importances:
     1. feature 'A9   ' (0.2269)
     2. feature 'A8   ' (0.1020)
     3. feature 'A11  ' (0.0816)
     4. feature 'A15  ' (0.0813)
     5. feature 'A3   ' (0.0791)
     6. feature 'A14  ' (0.0730)
     7. feature 'A2   ' (0.0672)
     8. feature 'A10  ' (0.0649)
     9. feature 'A6_x ' (0.0155)
    10. feature 'A7_h ' (0.0141)
    11. feature 'A12  ' (0.0139)
    12. feature 'A1   ' (0.0133)
    13. feature 'A7_v ' (0.0122)
    14. feature 'A6_q ' (0.0116)
    15. feature 'A6_k ' (0.0110)
    16. feature 'A5_p ' (0.0104)
    17. feature 'A13_g' (0.0100)
    18. feature 'A6_w ' (0.0099)
    19. feature 'A6_ff' (0.0093)
    20. feature 'A5_g ' (0.0087)
    21. feature 'A6_c ' (0.0079)
    22. feature 'A4_u ' (0.0075)
    23. feature 'A4_y ' (0.0071)
    24. feature 'A7_bb' (0.0071)
    25. feature 'A13_s' (0.0070)
    26. feature 'A6_cc' (0.0067)
    27. feature 'A6_i ' (0.0059)
    28. feature 'A7_ff' (0.0058)
    29. feature 'A6_aa' (0.0047)
    30. feature 'A6_m ' (0.0037)
    31. feature 'A6_e ' (0.0035)
    32. feature 'A13_p' (0.0035)
    33. feature 'A6_d ' (0.0031)
    34. feature 'A7_j ' (0.0022)
    35. feature 'A7_n ' (0.0021)
    36. feature 'A4_l ' (0.0016)
    37. feature 'A7_dd' (0.0012)
    38. feature 'A7_z ' (0.0012)
    39. feature 'A5_gg' (0.0011)
    40. feature 'A6_j ' (0.0007)
    41. feature 'A6_r ' (0.0004)
    42. feature 'A7_o ' (0.0000)



```python
d_first = 20
plt.figure(figsize=(8, 8))
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first]);
```


```python
best_features = indices[:8]
best_features_names = feature_names[best_features]
print(best_features_names)
```

    Index(['A9', 'A8', 'A11', 'A15', 'A3', 'A14', 'A2', 'A10'], dtype='object')



```python
from sklearn import ensemble
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train, y_train)

err_train = np.mean(y_train != gbt.predict(X_train))
err_test = np.mean(y_test != gbt.predict(X_test))
print(err_train, err_test) # error 10%!
```

    0.0248447204969 0.101449275362

