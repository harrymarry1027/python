#!/usr/bin/env python
# coding: utf-8

# # 판다스 

# ### 파이썬에서 데이터 처리를 위해 존재하는 가장 있기 있는 라이브러리 
# 
# - 일반적으로 대부분의 데이터 세트는 2차원 데이터 
# - 행(Row) * 열(Column)로 구성되어 있다(RDBMS)
# - 판다스의 핵심 개체는 DataFrame 
# - DataFrame은 여러개의 행과 열로 이뤄진 2차원 데이터를 담는 데이터 구조체 
#     - index와 Series를 이해하는 것도 중요하다 
#     - Index는 개별 데이터를 고유하게 식별하는 Key의 값이다.
#     - Series와 DataFrame의 가장 큰 차이는 Series는 칼럼이 하나뿐이 데이터 구조체 이고, DataFrame은 여러개인 데이터 구조체 라는점
#     

# ## 판다스 시작 - 파일을 DataFrame으로 로딩

# In[6]:


import pandas as pd
import numpy as np


# - read_csv(): CSV(칼럼을 ','로 구분한 파일 포맷) 파일 포맷 변환을 위한 API
# 
# - read_table() : 필드 구문 문자가(Delimeter)가 콤마(','), 탭('\t') 이냐 차이, 디폴트(기본값) 필드 구분 문자는 탭 문자
# 
# - read_csv(filepath_or_buffer, sep=","...)함수에서 가장 중요한 인자는 filepath
# 
# - filepath에는 로드하는 데이터 파일의 경로를 포함한 파일명을 입력하면 된다. 
# 
# 1) 구분자 : sep='I'/ '/t'
# - 기본적으로 csv파일을 쉼표로 데이터 값이 구분되기 때문에 따로 구분자를 설정할 필요가 없다 
# - 하지만 콤마(,)가 아닌 다른것으로 구분자가 설정되어 있을 경우 데이터를 그냥 불러오게 되면 에러가 생김
#  
#  pd.read_csv('파일명', sep = '|')
#  
# 2) 인코딩 : encoding = 'utf-8'/'cp949'
# - 내가 불러오고자 하는 파일의 encoding과 python encoding의 설정이 맞지 않으면 에러가 발생
# - (UnicodeDecodeError) 이러한 경우 endcoding='utf-8' 혹은 encoding='cp949'로 설정해준다 
# 
#  pd.read_csv('파일명', endcoding='utf-8')
#  
# 3) 특정 줄은 제외하고 파일 불러오기 : skiprows
# - skiprows에 내가 스킵하고 싶은 행의 개수를 적어주면 된다. skiprows=15를 적어주면 15번째 행까지 스킵해보면 16번째 행부터 출력 됨을 알수 있다.
# 
#  pd.read_csv('파일명', endcoding='utf-8', skiporws=15)
#  
# 4) 특정열(columns)을 index로 지정 : index_col=1(인덱스번호)/'칼럼명
# 
#  pd.read_csv('파일명', index_col =1)
#  
#  pd.read_csv('파일명', index_col = '칼럼명')
#  
# 5) 칼럼명 변경 : names=[]
# 
#  pd.read_csv('파일명', names=['칼럼명1','칼럼명2'])
#  
# 
# 6) 특정값을 NaN으로 취급 : na_values
#  pd.read_csv('파일명', na_values = ['Q']로 써준다

# In[7]:


df = pd.read_csv('titanic_train.csv')


# In[8]:


df


# In[9]:


df.head(3)


# In[10]:


df.shape # df는 891개의 로우와 12개의 칼럼으로 이뤄졌습니다 


# In[11]:


df.info()


# ### 타이타닉 분석
# - 전체데이터는 891개 row이고, Column수는 12개 이다 
# - 칼럼별 데이터 타입은 2개의 칼럼이 float64, 5개의 컬럼이 int64, 5개의 컬럼이 object타입
# - non-null(Null값이아님)인지 나타냄
# 

# In[12]:


df.describe()


# - describe()는 오직 숫자형(int, float등) 칼럼의 분포도만 조사하며 자동으로 object타입의 칼럼은 출력에서 제외 
# 
# - count: not Null인 데이트 건수
# - mean : 평균값
# - std : 표준편차
# - min : 최솟값
# - max : 최댓값
# 
# - describe() 해당 숫자 칼럼이 숫자형 카테고리 칼럼인지를 판달할수 있게 도와준다 
#     - 카테고리 칼럼은 특정 범주에 속하는 값을 코드화한 칼럼 
#     - 가령 survived 
# tip : 데이터의 분포도를 아는 것은 머신러닝 알고리즘의 성능을 향상시키는 중요한 요소 
# 가령 회귀 에서 결정값이 정규 분포를 이루지 않고 특정 값으로 왜곡돼 있는 경우, 또는 데이터 값에 이상치가 많을 경우 예측 성능이 저하됨
# 

# In[13]:


df['Pclass'].value_counts()


# In[14]:


df_Pclas = df['Pclass'] # DataFrame의 []연산자 내부에 컬럼명을 입력하면 해당 칼럼에 해당하는 Series객체를 반환


# In[15]:


df_Pclass.head() # Series는 index와 단 하나의 칼럼으로 구성된 데이터 세트 


# - value_counts() : 칼럼 값별 데이터 건수를 반환하므로 구유 칼럼 값을 식별자로 사용할수 있다. 
# - 인덱스는 또한 숫자형 뿐만 아니라 문자열도 가능, 단 모든 인덱스는 고유성이 보장 되어야 한다 
# - Null값을 무시하고 결괏값을 내놓기 쉽다는 점을 유의해야 한다 
#   -> 이에 dropna 인자로 Null 값을 포함하여 개별 데이터 값의 건수를 계산할지를 판단해야 한다 
#   

# In[94]:


df['Embarked'].value_counts()


# In[95]:


df['Embarked'].value_counts(dropna=False) # 널값을 포함하여 value_counts를 적용하고 자 하면 False, 아니면, True-> 이는 디폴트 값이므로


# ## 넘파이 ndarray, 리스트, 딕셔너리를 DataFrame으로 변환하기 

# In[96]:


import numpy as np


# In[97]:


col_name1=['col1']
list1 = [1,2,3]
array1 = np.array(list1)
array1.shape


# ### 리스트를 이용해 DataFrame 생성

# In[98]:


df_list1=pd.DataFrame(list1, columns=col_name1)


# In[99]:


df_list1


# ### 넘파이 ndarray를 이용해 DataFrame 생성

# In[100]:


df_array1 = pd.DataFrame(array1, columns=col_name1)


# In[101]:


df_array1 


# - 1차원 형태의 데이터 기반으로 DataFrame을 생성하므로 칼럼명이 한개만 필요하다는 사실에 주의 

# In[102]:


col_name2=['col1','col2','col3']

list2 = [[1,2,3],[11,12,13]]

array2 = np.array(list2)


# In[103]:


df_list2 = pd.DataFrame(list2, columns=col_name2)
df_list2


# In[104]:


df_array2 = pd.DataFrame(array2, columns=col_name2)
df_array2


# ### DataFrame을 넘파이 ndarray, 리스트, 딕셔너리로 변환하기 

# 많은 머신러닝 패키지가 기본 데이터 형으로 넘파이 ndarray를 사용한다 
# 데이터 핸들링은 DataFrame을 이용하더라도 머신러닝 패키지의 입력 인장 등에 적용하기 위해 다시 넘파이로 변환하는 경우가 종종생김
# 

# #### DataFrame을 ndarray로 변환

# In[105]:


# DataFrame을 ndarray로 변환
array3 = df_list2.values
array3


# In[106]:


type(array3)


# In[107]:


array3.shape


# #### DataFrame을 리스트로 변환

# In[ ]:


list3 = df_list2.values.tolist()
list3


# In[ ]:


type(list3)


# #### DataeFrame을 딕셔너리로 변환

# In[ ]:


dict3 = df_list2.to_dict('list')


# In[ ]:


dict3


# In[ ]:


type(dict3)


# ### Data Frame의 칼럼 데이트 세트 생성과 수정
# - 칼럼 데이터 세트 생성과 수정 역시 []연산자를 이용해 쉽게 할수 있다.

# In[108]:


df1=df.copy()


# In[109]:


df1['Age_0']=0


# In[110]:


df1


# In[ ]:


df1['Age_by_10'] = df1['Age']*10


# In[ ]:


df1.head(2)


# In[ ]:


df1['Family_no'] = df1['SibSp']+df1['Parch']+1


# In[ ]:


df1.head(2)


# In[ ]:


df1['Age_by_10'] = df['Age_by_10'] +100
df.head(4)


# ### DataFrame 데이터 삭제
# 
# - 데이터의 삭제 : drop() 메서드를 사용
# 
# 
# DataFrame.drop(labels=None, axis=0, index=None, Columns=None, level=None, inplace=False, errors='raise')
# 
# - 가중 중요한 파라미터는 labels, axis, inplace
#     - axis : 값에 따라서 특정 칼럼 또는 특정 행을 드롭한다. axis =0 로우 방향축, axis=1 칼럼 방향축
#     - axis=0 을 으로 설정하고 로우 레벨로 삭제를 하는 경우는 이상치 데이터를 삭제하는 경우에 주로 사용한다 
# 

# In[ ]:


drop_df1= df1.drop('Age_0', axis=1)


# In[ ]:


drop_df1


# In[ ]:


df1


#  inplace=False, inplace 파라미터는 기재하지 않으면 자동으로 False가 된다, inplace는 디폴트 값이 False이므로 True로 해줘얗ㄴ다

# In[ ]:


drop_result = df1.drop(['Age_0','Age_by_10','Family_no'], axis=1, inplace=True)


# In[ ]:


df1


# row을 삭제 할때는 
# df.drop([1,2,3], axis=0, inplace=True)

# ### Index 객체
# 
# - index는 오직 식별용으로만 사용된다 

# In[ ]:


indexes =df.index


# In[ ]:


indexes


# In[ ]:


indexes.values  # index 객체를 실제 값 array로 변환


# In[ ]:


print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])


# In[ ]:


series_fair = df['Fare']
print(series_fair.max())
print(series_fair.min())
print(series_fair.sum())


# In[ ]:


print(series_fair+3)


# In[ ]:


reset_df1 = df1.reset_index(inplace=False)


# In[ ]:


reset_df1.head(3) # 새로운 인덱스를 추가함 , 인덱스가 연속된 int 숫자형 데이터가 아닐 경우에 다시 이를 연속 int 숫자형 데이터로 만들때 사용


# In[ ]:


value_count =  df1['Pclass'].value_counts()


# In[ ]:


value_count


# In[ ]:


type(value_count)


# In[ ]:


new_value_counts = value_count.reset_index(inplace=False)
new_value_counts


# In[ ]:


type(new_value_counts)


# ### 데이터 셀렉션 및 필터링 
# 
# - 판다스의 셀렉션 및 필터링은 iloc[], loc[] 연산자를 통해 동일한 작업을 수행한다 

# #### DataFrame의 [  ] 연산자 
# - 넘파이의 [  ] 연산자는 행의 위치, 열의 위치, 슬라이싱 범위 등을 지정해 데이터를 가져올수 있었다
# - DataFrame 바로 뒤에 있는 '[ ]'안에 들어갈수 있는 것은 컬럼명 문제(또는 칼럼명의 리스트 객체), 또는 인덱스로 가능한 표현식
# - DataFrame의 [  ] 내에 숫자 값을 입력할 경우 오류가 발생
# 
# - 판다스의 인덱스 형태로 변환 가능한 표현식은 [ ] 입력 가능

# In[ ]:


df1[0:2] # 인덱스의 0,1행을 가져옴, 하지만 이 방법은 좋지 않다 


# In[ ]:


df[df['Pclass']==3].head(3)


#  혼돈을 방지하기 위한 좋은 가이드
#  - dataFrame 바로 뒤의[]연산자는 넘파이의 []나 Series의 []와 다르다
#  - DataFrame 바로 뒤의[] 내 입력값은 칼럼명(또는 칼럼 리스트)을 지정해 칼럼 지정 연산에 사용하거나 불린 인덱스 용도로만 사용해야 한다 
#  - DataFrame[:2]와 같은 슬라이싱 연산으로 데이터를 추출하는 방법은 사용하지 않는게 좋다 

# #### DataFrame iloc[   ] 연산자
# - 로우나 칼럼을 지정하여 데이터를 선택할 수 있는 인덱싱 방식으로 iloc[]와 loc[]를 제공한다 
# - iloc[ ] :  위치(location)기반 인덱싱 방식으로 동작
#              위치기반 인덱싱은 행과 열위치를, 0을 출발점으로 하는 세로축, 가로축 좌표 정숫값으로 지정하는 방식
#              iloc는 위치기반 인덱싱만 허용하기 때문에 행과 열의 좌표 위치에 해당하는 값으로 정숫값, 또는 정수형의 슬라이싱, 팬시 리스트값을 입력해야 한다 
# - loc[] : 명칭(label)기반 인덱싱으로 동작 한다 
#           명칭 기반 인덱싱은 데이터 프레임의 인덱스 값으로 행 위치를, 칼럼의 명칭으로 열 위치를 지정하는 방식
# - 슬라이싱과 팬시 인덱싱은 제공하나 명확한 위치기반 인덱싱이 사용되어야 하는 제약으로 인해 불린 인덱싱은 제공하지 않는다 

# In[16]:


date = {'Name':['Chulmin','Eunkyung','Jinwoong','Soobeom'],
        'Year':[2011,2016,2015,2015],
        'Gender':['Male','Female','Male','Male']}
date_df = pd.DataFrame(date, index=['one','two','three','four'])


# In[17]:


date_df


# In[18]:


date_df.iloc[0,0]


# In[19]:


date_df.iloc[1,2]


# In[20]:


date_df.iloc[0,'Name']  # ValueError 오류, 정수값이 아닌 칼럼 명칭을 입력하여 오류가 발생함


# In[21]:


date_df.iloc[1:2, [0,1]] # 첫번째 행과, 첫번째열에서 두번째열 


# In[ ]:


date_df.iloc[0:2, 0:3] # 첫번째행과, 두번째행, 첫번째에서 세번째 열까지


# In[ ]:


date_df.iloc[:] # 전체 DataFrame 반환


# In[ ]:


date_df.iloc[:,-1] # 가장 마지막 열데이터를 가져오는데 자주 사용, 타깃값을 가져온다 


# In[ ]:


date_df.iloc[:, :-1] # 가장 마지막 열 데이터를 제외하고 나머지 모든 데이터 , 피처값을 가져옴


# #### DataFrame loc[  ] 연산자
# - loc[ ] : 명칭(LaBel)기반으로 데이터를 추출
# - loc[인덱스값, 칼럼명] 과 같은 형식으로 데이터를 추출한다 
# - loc[]는 명칭 기반이므로 열 위치에 '칼럼명'이 들어가는 것은 직관적으로 이해가 된다 
# - 정수값을 사용하지 않는다 

# In[22]:


date_df.loc['one','Name']


# In[23]:


date_df.loc[0,'Name'] # key error가 난다 


# In[24]:


date_df.loc['one':'two','Name']


# In[25]:


date_df.loc['three','Name']


# In[27]:


date_df.loc['one':'two',['Name','Year']]


# In[28]:


date_df.loc['one':'two','Name':'Gender'] # 인덱스 값 one 부터 three까지 행의 Name부터 Gender 칼럼까지의 DataFrame 반환


# In[29]:


date_df.loc[:]


# In[30]:


date_df.loc[date_df.Year>=2014]


# #### 불린 인덱싱
# 
# - 매우 편리한 데이터 필터링 방식 
# - [  ], loc[]에서 공통으로 지원
# 

# In[32]:


df = pd.read_csv('titanic_train.csv')
df_boolean = df[df['Age']>60]


# In[33]:


df_boolean


# In[37]:


df[df['Age']>60][['Name','Age']].head(3)


# In[39]:


df.loc[df['Age']>60,['Name',"Age"]].head(3) # loc 사용한것


# In[40]:


df[(df['Age']>60)&(df['Pclass']==1)&(df['Sex']=='female')]


# In[42]:


cond1 = df['Age']>60
cond2 = df['Pclass']==1
cond3 = df['Sex']=='female'
df[cond1&cond2&cond3]


# ### 정렬, Aggreation 함수, GroupBy적용

# #### DataFrame, Series의 정렬 -sort_values()
# 
# - sort_values : 정렬을 위해서 사용하는 메서드
# - SQL의 order by 키워드와 매우 유사하다 
# - sort_values()의 주요 입력 파라미터는 : by, ascending, inplace
# - by로 특정 칼럼을 입력하면 해당 칼럼으로 정렬을 수행한다 
# - ascending = True로 설정하면 오름차순으로 정렬
# - ascending = False, 이것이 기본, 설정하면 sort_values()를 호출한 DataFrame 그대로 정렬된 결과로 변환 
# 

# In[44]:


df_sorted = df.sort_values(by=['Name'])
df_sorted


# In[45]:


df_sorted = df.sort_values(by=['Name','Pclass'], ascending=False)


# In[46]:


df_sorted.head(3)


# #### Aggregation함수 적용

# In[47]:


df.count()


# In[49]:


df[["Age","Fare"]].mean()


# #### groupby() 적용
# 
# - DataFramedp groupby()를 호출하면 DataFrameGroupBy라는 또 다른 형태의 DataFrame을 반환, 

# In[50]:


df_groupby = df.groupby(by='Pclass')


# In[51]:


df_groupby


# In[53]:


df_groupby = df.groupby('Pclass').count()


# In[54]:


df_groupby


# In[57]:


df_groupby = df.groupby('Pclass')[['PassengerId','Survived']].count()


# In[58]:


df_groupby


# In[59]:


## 한번에 하고 싶다 하면 agg()내에 인자로 입력해서 사용할수 있다 


# In[63]:


df.groupby('Pclass')['Age'].agg([max,min])# agg()내에 입력값으로 딕셔너리 형태로 aggregation이 적용된 칼럼들과 aggration함수를 입력


# ### 결손 데이터 처리하기
# 
# - 판다스는 결손 데이터를 처리하는 편리한 API를 제공
# - 결손 데이터는 칼럼에 값이 없는, 즉 NULL인 경우를 의미, 이를 넘파이의 NaN으로 표시 한다 
# - 기본적으로 머신러닝 알고리즘은 이 NaN값을 처리하지 않으므로 이 값을 다른값으로 대체 
# 

# #### isna()로 결손 데이터 여부 확인
# - isna() : 데이터가 NaN인지 아닌지를 알려준다 
# - isna()를 수행하면 모든 칼럼의 값이 NaN인지 아닌지를 True나 False로 알려준다 

# In[68]:


df.isna().head(3)


# In[70]:


df.isna().sum() # sum()을 호출시 True는 내부적으로 숫자1, False는 숫자0으로 변환되므로 결손 데이터의 개수를 구할수 있다


# #### fillna()로 결손 데이터 대체 하기 
# - fillna: 결손 데이터를 편리하게 다른 값으로 대체 할수 있다
# - 주의 할점 : fillna()를 이용해 반환값을 다시 받거나, inplace=True파라미터를 fillna()에 추가해야 실제 데이터 세트값이 변경된다는 점

# In[71]:


df['Cabin'] = df['Cabin'].fillna('C000')


# In[72]:


df


# In[74]:


df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna('s')
df.isna().sum()


# In[75]:


df


# ### apply lambda 식으로 데이터 가공 
# 
# - 판다스는 apply 함수 lambda 식을 결합해 DataFrame이나 Series의 레코드 별로 데이터를 가공하는 기능을 제공
# 
# **lambda x : x **2

# In[77]:


a=[1,2,3]
square = map(lambda x: x**2, a)
list(square)


# In[78]:


df['Child_Adult']= df['Age'].apply(lambda x:'Child'if x<=15 else 'Adult')
df[['Age','Child_Adult']].head(8)


# In[79]:


df['Age_cat'] = df['Age'].apply(lambda x:'Child' if x<=15 else ('Adult' if x<=60 else 'Elderly'))


# In[86]:


df[['Age','Age_cat']]


# In[81]:


df['Age_cat'].value_counts()


# In[89]:


def get_category(age):
    cat = ''
    if age <=5: cat = 'Baby'
    elif age <= 12: cat='Child'
    elif age <= 18: cat='Teenager'
    elif age <= 25: cat='Student'
    elif age <= 35: cat='young Adult'
    elif age <= 60: cat='Adult'
    else : cat='Elderly'
        
    return cat


# In[90]:


df['Age_cat']=df['Age'].apply(lambda x:get_category(x))


# In[92]:


df[['Age','Age_cat']].head(10)


# ### 두 개의 DataFrame 합치기

# In[111]:


##가상 abalone 1개 row데이터 생성 및 결합
# one_abalone_df = abalone_df.iloc[[0]]
# pd.concat([abalone_df,one_abalone_df],axis=0)

## 가상 abalone 1개 col 데이터 생성 및 결합
# one_abalone_df = abalone_df.iloc[:,[0]]
# pd.concat([abalone_df,one_abalone_df],axis=1)


# In[ ]:




