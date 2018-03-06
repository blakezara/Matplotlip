
##Analysis:

#1. Average Fares are higher in rural areas.

#2. Urban areas had more drivers, rides and fare overall.

#3. Suburban areas have higher number of riders than rural but less than rural.


```python
# Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Import Files And Merge

ride_data = pd.read_csv("raw_data/ride_data.csv")
city_data = pd.read_csv("raw_data/city_data.csv")

merged = pd.merge(city_data, ride_data, how='outer', on='city')
merged.head()

merged.columns=("City", "Driver Count", "Type", "Date", "Fare", "Ride ID")


merged.head()


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
      <th>City</th>
      <th>Driver Count</th>
      <th>Type</th>
      <th>Date</th>
      <th>Fare</th>
      <th>Ride ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-19 04:27:52</td>
      <td>5.51</td>
      <td>6246006544795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-04-17 06:59:50</td>
      <td>5.54</td>
      <td>7466473222333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-05-04 15:06:07</td>
      <td>30.54</td>
      <td>2140501382736</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-01-25 20:44:56</td>
      <td>12.08</td>
      <td>1896987891309</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-09 18:19:47</td>
      <td>17.91</td>
      <td>8784212854829</td>
    </tr>
  </tbody>
</table>
</div>




```python
#groupby
merged_data = merged.groupby(["City","Type","Driver Count"])

#average fare
new_data = merged_data["Fare"].mean()
new_data = pd.DataFrame(new_data)

#total rides
new_data["Total Rides"] = merged_data["Ride ID"].nunique()
new_data.reset_index(inplace=True)

#reorganize
new_data.rename(columns={"Fare":"Average Fare"},inplace=True)

new_data.head()
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
      <th>City</th>
      <th>Type</th>
      <th>Driver Count</th>
      <th>Average Fare</th>
      <th>Total Rides</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alvarezhaven</td>
      <td>Urban</td>
      <td>21</td>
      <td>23.928710</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alyssaberg</td>
      <td>Urban</td>
      <td>67</td>
      <td>20.609615</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anitamouth</td>
      <td>Suburban</td>
      <td>16</td>
      <td>37.315556</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Antoniomouth</td>
      <td>Urban</td>
      <td>21</td>
      <td>23.625000</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aprilchester</td>
      <td>Urban</td>
      <td>49</td>
      <td>21.981579</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
size = np.arange(0, 1000, 10)

figure= sns.lmplot(x='Total Rides', y='Average Fare', hue='Type', scatter_kws={"s": size, 'alpha':0.50,}, data=new_data, fit_reg=False)

plt.ylim(15, 45)
plt.xlim(0, 40)
plt.xlabel('Total Number of Rides Per City')
plt.ylabel('Average Fare')

plt.title('Pyber Rideshare Data')

plt.show()
```


![png](output_4_0.png)



```python
#Percent of Total Fares by City Type
type_data = merged.groupby("Type")['Type', 'Fare', 'Ride ID', 'Driver Count']

fare = type_data.sum()["Fare"]                           
fare

labels = fare.index
explode = [.3 , 0, 0]

plt.pie(fare, startangle = 140, explode = explode, labels = labels, 
        autopct = "%1.1f%%", 
        shadow = True, 
        wedgeprops = {'linewidth': .2, 'edgecolor': 'black'})

plt.title("Percentage of Total Fares by City Type")
plt.show()
```


![png](output_5_0.png)



```python
#Percentage of Total Rides by City Type

rides= type_data.count()["Ride ID"]
rides

labels = rides.index
explode = [.3 , 0, 0]

plt.pie(rides, startangle = 140, explode = explode, labels = labels, autopct = "%1.1f%%", shadow = True, wedgeprops = {'linewidth': .5, 'edgecolor': 'black'})

plt.title("Percentage of Total Rides by City Type")

plt.show()
```


![png](output_6_0.png)



```python
#Percentage of Total Drivers by City Type

drivers= type_data.sum()["Driver Count"]
drivers

labels = rides.index
explode = [.3 , 0, 0]

plt.pie(drivers, startangle = 140, explode = explode, labels = labels, autopct = "%1.1f%%", shadow = True, wedgeprops = {'linewidth': .5, 'edgecolor': 'black'})

plt.title("Percentage of Total Drivers by City Type")
plt.show()
```


![png](output_7_0.png)

