### Appendix. 01 이미지와 Numpy

> OpenCV에서 이미지나 동영상을 읽어들이는 함수 `cv2.imread()` 는 Numpy 배열을 반환한다. 따라서 OpenCV를 파이썬 언어로 프로그래밍한다는 것은 Numpy 배열을 다룬다는 의미이다.

<br>

### 1. Image Type

- OpenCV로 읽어들인 500X500 픽셀 이미지 정보를 담은 Numpy 배열의 속성 정보

```python
img_file = 'img/figures.jpg' # 이미지 불러오기
img = cv2.imread(img_file) # 이미지 변수 할당

type(img)
```

- 결과

```shell
numpy.ndarray
```

<br>

### 2. Image Dimension / Shape / Size / Dtype

- OpenCV는 기본적으로 이미지를 3차원 배열, 즉 행 X 열 X 채널로 표현
- 행과 열은 이미지의 크기, 즉 높이와 폭만큼의 길이를 갖고 채널은 컬러인 경우 파랑, 초록, 빨강 3개의 길이를 갖는다.

```python
img.ndim
# 결과 : 3
```

- 일반적인 이미지를 읽었을 때 3차원 배열은 `높이 X 폭 X 3` 형태

<br>

```python
img.shape
# 결과 : (579, 916, 3)
img.size
# 결과 : 1591092
```

- img.size는 전체 요소의 개수로 각 차원의 길이를 곱한 값과 같음

<br>

```python
img.dtype
# 결과 : dtype('uint8')
img.itemsize
# 결과 : 1
```

- 이미지 픽셀 데이터는 음수이거나 소수점을 갖는 경우가 없고, 값의 크기도 최대 255이므로 부호 없는 8비트, 즉 `uint8` 을 데이터 타입으로 사용한다.
- `img.itemsize` 결과 값은 각 요소의 크기가 1 바이트인 것을 나타낸다.

<br>

### 3. Numpy 배열 생성

#### 3.1 값으로 생성 : `array()`

- 예시 01 - `array()` dtype 미지정시 자동으로 결정

```python
np.array([[1,2,3,4], [5,6,7,8]])
# 결과 
# array([[1, 2, 3, 4],
#        [5, 6, 7, 8]])
```

- 예시 02 - `array()` - dtype 지정

```python
np.array([[1,2,3,4], [5,6,7,8]], dtype = np.float32)
# 결과
# array([[1., 2., 3., 4.],
#       [5., 6., 7., 8.]], dtype=float32)
```

<br>

#### 3.2 초기 값으로 생성 : `empty()`, `zeros()`, `ones()`, `full()`

- 예시 01 - `empty()`

```python
a = np.empty((3,3))
```

```shell
array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
       [0.00000000e+000, 0.00000000e+000, 8.85365637e-321],
       [1.63857973e+248, 3.54556998e+246, 2.28402781e+242]])
```

   - 위의 결과에서 알 수 있듯이, 2행 3열 배열이 만들어졌지만 초기 값은 제각각이다.

   - 어떤 값으로 초기화를 하고 싶을 경우, `fill()`을 사용하면 좋다.

     			- `fill()`

     ```python
     a.fill(255)
     ```

     ```shell
     array([[255., 255., 255.],
            [255., 255., 255.],
            [255., 255., 255.]])
     ```

<br>

- 예시 02 - `zeros() ` : 영행렬 dtype 미지정

```python
b = np.zeros((2,3))
```

```python
array([[0., 0., 0.],
       [0., 0., 0.]])
```

- 위의 영행렬은 dtype을 지정하지 않았다. float64 형태로 출력되었다.

```python
b = np.zeros((2,3), np.int8)
```

```
array([[0, 0, 0],
       [0, 0, 0]], dtype=int8)
```

- dtype을 int8 형태로 지정해주었다.
- 원하는 dtype을 지정해 배열을 생성할 수 있음

<br>

- 예시 03 - `ones()` 

```python
b = np.ones((2,3))
```

```shell
array([[1., 1., 1.],
       [1., 1., 1.]])
```

- 예시 04 - `full()`

```python
e = np.full((2,3,4), 255, dtype = np.uint8)
```

```shell
array([[[255, 255, 255, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255]],

       [[255, 255, 255, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255]]], dtype=uint8)
```

- 위의 배열은 2 X 3 X 4 배열을 255로 초기화한 코드의 결과이다.

<br>

#### 3.3 기존 배열로 생성 : `empty_like()`, `zeros_like()`, `ones_like()`, `full_like()`

- 예시 01 - `empty_like()`

```
img_file = 'img/girl.jpg' # 이미지 불러오기
img = cv2.imread(img_file) # 이미지 변수 할당
img
```

```shell
# 결과
array([[[ 70,  70,  70],
        [ 66,  66,  66],
        [ 64,  64,  64],
        ...,
        
        [200, 200, 200],
        [203, 203, 203],
        [205, 205, 205]]], dtype=uint8)
```

```python
img.shape
# 결과 : (293, 406, 3)
```

```python
a = np.empty_like(img)
```

```shell
array([[[ 80,   1, 232],
        [121, 243,   1],
        [  0,   0, 240],
        ...,
        [  0,   0, 128],
        [  0,   0,   0],
        [128,   0,   0]],

       [[  0, 128,   0],
        [  0,   0, 153],
        [  0,   0,   0],
        ...,
        [  0,   0,   0],
        [  0,   0,   0],
        [  0,   0,   0]]], dtype=uint8)
```

- 기존의 `img`의 shape와 dtype과 동일한 값을 갖지만, 초기화된 값만 다르다.

```python
a.shape
# 결과 : (293, 406, 3)
```

<br>

- 예시 02 - `zeros_like()`

```python
b = np.zeros_like(img)
```

```shell
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], dtype=uint8)
```

```python
b.shape
# 결과 : (293, 406, 3)
```

- `empty_like()`와 마찬가지로 동일한 shape, dtype을 갖지만 초기화된 값만 다르다. `ones_like()`, `full_like()` 모두 동일하다.

<br>

#### 3.4 순차적인 값으로 생성 : `arange()`

- 예시 01 - `arange()`

```python
a = np.arange(5)
# 결과 : array([0, 1, 2, 3, 4])
```

- `numpy.arange(start=0, stop [, step = 1, dtype = float64])`
  - `start` : 시작 값
  - `stop` : 종료 값, 범위에 포함하는 수는 stop -1 까지
  - `step` : 증가 값
  - `dtype` : 데이터 타입

<br>

- 예시 02 - `arange()`

```python
a = np.arange(5.0)
# 결과 : array([0., 1., 2., 3., 4.])
```

<br>

- 예시 03 - `arange()`

```python
a = np.arange(3, 9, 2)
# 결과 : array([3, 5, 7])
```

<br>

#### 3.5 난수로 생성 : `random.rand()`, `random.randn()`

- 예시 01 - `random.rand()`

```python
np.random.rand(2,3)
# 결과
# array([[0.49216677, 0.83138897, 0.83845003],
#       [0.72060499, 0.16273978, 0.06740348]])
```

- 난수로 채워진 2행 3열의 배열을 생성

<br>

### 4. 차원 변경

- `reshape(newshape)` 

  - 예시

  ```python
  a = np.arange(6)
  # 결과 : array([0, 1, 2, 3, 4, 5])
  
  a.reshape(2,3)
  # 결과
  # array([[0, 1, 2],
  #        [3, 4, 5]])
  ```

  - 처음에 생성한 배열은 1차원 배열이지만, `reshape()` 을 통해서 2행 3열로 바꾸었다.

  <br>

  - 예시

  ```python
  a = np.arange(10)
  # 결과 : array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  ```

  ```python
  a.reshape(2,-1)
  # 결과
  # array([[0, 1, 2, 3, 4],
  #       [5, 6, 7, 8, 9]])
  ```

  - `(2, -1)` 은 2행 -1 열을 생성하겠다는 의미인데, 2행에 맞춰서 자동으로 열을 생성하겠다는 의밐ㅁㅂ

- `numpy.ravel(ndarray)` : 1차원 배열로 차원 변경

  - `ndarray` ; 변경할 원본 배열

- `ndarray.T` : 전치배열 Transpose

<br>

### 5. 병합과 분리

#### 5.1 2개 이상의 NumPy 배열 병합

- 배열들을 이여 붙여서 크기를 키우는 방법 `vstack()`, `hstack()`

  - `numpy.vstack()` 수직 병합

  ```python
  a = np.arange(4).reshape(2,2)
  b = np.arange(10, 14).reshape(2,2)
  np.vstack((a,b))
  ```

  ```shell
  array([[ 0,  1],
         [ 2,  3],
         [10, 11],
         [12, 13]])
  ```

  - 수직 병합한 결과, 4행 2열의 배열이 생성되었다.
  - 동일한 결과를 `concatenate()` 를 통해서도 생성가능하다.

  ```python
  np.concatenate((a,b), 0)
  # 축 번호로 0을 지정한 결과(행 방향으로)
  ```

  <br>

  - `numpy.hstack()` 수평 병합

  ```python
  np.hstack((a,b))
  ```

  ```
  array([[ 0,  1, 10, 11],
         [ 2,  3, 12, 13]])
  ```

  - 수평으로 병합한 결과, 2행 4열의 배열이 생성되었다.
  - 동일한 결과를 `concatenate()` 를 통해서도 생성가능하다.

  ```python
  np.concatenate((a,b), 1)
  # 축 번호를 1로 지정(열 방향으로)
  ```

<br>

- 새로운 차원을 만들어 서로서로 끼워넣는 방법 `np.stack()`
  - `np.stack()` 에서 축 번호를 지정하지 않으면 `0번`을 의미하고, `-1`은 마지막 축 번호를 의미함
- 예시 01 `np.stack(arrays, axis = 0)`

```python
a = np.arange(12).reshape(4,3)
b = np.arange(10, 130, 10).reshape(4,3)
```

```python
a
# 결과
# array([[ 0,  1,  2],
#       [ 3,  4,  5],
#       [ 6,  7,  8],
#       [ 9, 10, 11]])
```

```python
b
# 결과
# array([[ 10,  20,  30],
#       [ 40,  50,  60],
#       [ 70,  80,  90],
#       [100, 110, 120]])
```

```python
c = np.stack((a,b), 0)
c
# 결과
# array([[[  0,   1,   2],
#        [  3,   4,   5],
#        [  6,   7,   8],
#        [  9,  10,  11]],

#       [[ 10,  20,  30],
#        [ 40,  50,  60],
#        [ 70,  80,  90],
#        [100, 110, 120]]])
```

```python
c.shape
# 결과 : (2, 4, 3)
```

​	`np.stack(arrays, axis=0)` 을 통해서, 새로운 차원을 만들어 `배열 a` 와 `배열 b`를 병합했다. (`0` 이 default)

<br>

- 예시 02 - `np.stack(arrays, axis = 1)`

```python
d = np.stack((a,b), axis = 1)
```

```shell
array([[[  0,   1,   2],
        [ 10,  20,  30]],

       [[  3,   4,   5],
        [ 40,  50,  60]],

       [[  6,   7,   8],
        [ 70,  80,  90]],

       [[  9,  10,  11],
        [100, 110, 120]]])
```

```python
d.shape
# 결과 : (4, 2, 3)
```



- 예시 03 - `np.stack(arrays, axis = -1)` / `np.stack(arrays, axis = 2)`

```python
e = np.stack((a,b), -1)
```

```shell
array([[[  0,  10],
        [  1,  20],
        [  2,  30]],

       [[  3,  40],
        [  4,  50],
        [  5,  60]],

       [[  6,  70],
        [  7,  80],
        [  8,  90]],

       [[  9, 100],
        [ 10, 110],
        [ 11, 120]]])
```

```python
e.shape
# 결과 : (4, 3, 2)
```

- `np.stack(axis = )`
  - `axis 는 0 / 1/ 2 / -1` 의 값을 가질 수 있는데, 새로운 축은 0번 축, 1번 축, 2번 축에 추가된다는 의미이다.

<br>

#### 5.2 배열 분리

- `np.vsplit(array, indice)` : 배열 수평 분리

  ```python
  b = np.arange(12).reshape(4,3)
  np.vsplit(b,2)
  ```

  ```shell
  [array([[0, 1, 2],
          [3, 4, 5]]), array([[ 6,  7,  8],
          [ 9, 10, 11]])]
  ```

  - `np.vsplit(b,2)` 은 4행 2열의 배열에서 4행을 2로 나누어 2행씩 갖게 하라는 의미

  ```python
  np.split(b, 2, 0)
  ```

  - `np.split(b, 2, 0)` 은 `vsplit()` 과 동일한 결과를 가져온다. axis 방향이 `0` 이기 때문

- `np.hsplit(array, indice)` : 배열 수직 분리

  ```python
  a = np.arange(12)
  np.hsplit(a, 3 )
  ```

  ```shell
  [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]
  ```

  <br>

  ```python
  np.hsplit(a, [3,6])
  ```

  ```markdown
  [array([0, 1, 2]), array([3, 4, 5]), array([ 6,  7,  8,  9, 10, 11])]
  ```

  - `[3, 6]` 은 [인덱스로 0:3], [3:6], [6:] 과 같은 의미이다.
  - `np.split()` 를 활용할 경우, axis 방향을 `1` 로 설정해주면 된다.

- `np.split(array, indice, axis = 0)` : array 배열을 axis 축으로 분리

  ```python
  np.split(b, [0], 1)
  ```

  ```shell
  array([], shape=(4, 0), dtype=int32), array([[ 0,  1,  2],
          [ 3,  4,  5],
          [ 6,  7,  8],
          [ 9, 10, 11]])]
  ```

  - `np.split(b, [0], 1)` 
  - `[0]` 은 [:0], [0:] 을 의미한다.
    - [:0] 은 어떤 value 도 추출되지 않는다.
  
  ```python
np.split(b, [1], 1)
  ```
  
  ```shell
  array([[0],
          [3],
          [6],
          [9]]), array([[ 1,  2],
          [ 4,  5],
          [ 7,  8],
        [10, 11]])]
  ```
  
  - `np.split(b, [1], 1)` 
    - `[1]` 은 [:1], [1:] 을 의미한다.

<br>

### 6. 검색

- `np.where(condition, t, f)`

  ```python
  a = np.arange(10, 20)
  
  np.where(a > 15)
  # 결과
  # (array([6, 7, 8, 9], dtype=int64),)
  
  np.where(a> 15, 1, 0)
  # 결과
  # array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
  ```

  <br>

- `np.nonzero(array)` : array 에서 요소 중에 0이 아닌 요소의 인덱스들을 반환

```python
z = np.array([0, 1, 2, 0, 1, 2])

np.nonzero(z)
# 결과 : (array([1, 2, 4, 5], dtype=int64),)
```

<br>

- `np.all(array, )` : array의 모든 요소가 True 인지 검색

```python
t = np.array([True, True, True])
np.all(t)
# 결과 : True
```

```python
t[1] = False # t의 2번째 value를 False로 바꾸기
# t는 array([True, False, True]) 가 됨

np.all(t) 
# 결과 : False
```

<br>

```python
a = np.arange(10)
b = np.arange(10)

np.all(a==b)
# 결과 : True
```

```python
np.where(a==b)
# 결과 : (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int64),)
```

- `np.any(array, )` : array의 어느 요소이든 True 가 있는지 검색

