import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def pluskey(r):
    v=0
    if(r==1):
        v=v+0.2
    elif(r==2):
        v=v+0.4
    elif(r==3):
        v=v+0.6
    elif(r==4):
        v=v+0.8
    elif(r==5):
        v=v+1
    return v

def negkey(r):
    v=0
    if(r==5):
        v=v+0.2
    elif(r==4):
        v=v+0.4
    elif(r==3):
        v=v+0.6
    elif(r==2):
        v=v+0.8
    elif(r==1):
        v=v+1
    return v

    
#questions and response recording
g=int(input("Are you a male or a female(1 for male 0 for female)"))

print("Enter choices for below questions as 1 for strongly disagree, 2 for disagree, 3 for moderate, 4 for agree, 5 for Strongly disagree" )
#Extraversion
e=0
r=int(input("Q.1 Do you feel comforatble around people?"))
e=pluskey(r)
    
r=int(input("Q.2 At parties, do you often talk to different people?"))
e=e+pluskey(r)

r=int(input("Q.3 Do you mind being center of attention"))
e=e+pluskey(r)
      
r=int(input("Q.4 Are you the life of the party?(Party animal)"))
e=e+pluskey(r)   

r=int(input("Q.5 Do you talk a lot"))
e=e+negkey(r)

r=int(input("Q.6 Do you like to draw attention to yourself"))
e=e+negkey(r)

r=int(input("Q.7 Are you quite around strangers"))
e=e+negkey(r)

r=int(input("Q.8 Do you wait for others to lead the way"))
e=e+negkey(r)

E=(e/8)*100   

#Agreeableness
a=0
r=int(input("Q.9 Do you think you sympathise with others easily?"))
a=pluskey(r)
    
r=int(input("Q.10 Do you take out some of your time for others"))
a=a+pluskey(r)

r=int(input("Q.11 You feel others emotions?"))
a=a+pluskey(r)
      
r=int(input("Q.12 Can you make others feel at ease"))
a=a+pluskey(r)   

r=int(input("Q.13 Are you interested in other people's life?"))
a=a+negkey(r)

r=int(input("Q.14 Do you insult others?"))
a=a+negkey(r)

r=int(input("Q.15 Are you interested in other peoples' problem"))
a=a+negkey(r)

r=int(input("Q.16 Are you hard to get to know"))
a=a+negkey(r)

A=(a/8)*100  

#Conscientiousness
c=0
r=int(input("Q.17 Do you follow a regular schedule?"))
c=pluskey(r)
    
r=int(input("Q.18 Are you always prepared ?"))
c=c+pluskey(r)

r=int(input("Q.19 Are you exacting in your work?"))
c=c+pluskey(r)
      
r=int(input("Q.20 Do you play attention to details?"))
c=c+pluskey(r)   

r=int(input("Q.21 Do you waste time?"))
c=c+negkey(r)

r=int(input("Q.22 Often forget to put back things in their proper place"))
c=c+negkey(r)

r=int(input("Q.23 Do you neglect your duties?"))
c=c+negkey(r)

r=int(input("Q.24 Do things in half-way manner"))
c=c+negkey(r)

C=(c/8)*100 

#Openness
o=0
r=int(input("Q.25 Do you enjoy wild flights of fantasy?"))
o=pluskey(r)
    
r=int(input("Q.26 Enjoy thinking about things?"))
o=o+pluskey(r)

r=int(input("Q.27 Believe in importance of art?"))
o=o+pluskey(r)
      
r=int(input("Q.28 Tend to vote for liberal political candidates"))
o=o+pluskey(r)   

r=int(input("Q.29 Avoid philosophical discussions"))
o=o+negkey(r)

r=int(input("Q.30 Do not like poetry?"))
o=o+negkey(r)

r=int(input("Q.31 Rarely look for deeper meaning in things."))
o=o+negkey(r)

r=int(input("Q.32 Have difficulty understanding abstract ideas."))
o=o+negkey(r)

O=(o/8)*100   

#Neuroticism
n=0
r=int(input("Q.33 Dislike yourself?"))
n=pluskey(r)
    
r=int(input("Q.34 Are you filled with doubt about things?"))
n=n+pluskey(r)

r=int(input("Q.35 Are you relaxed most of the time?"))
n=n+pluskey(r)
      
r=int(input("Q.36 Seldom get mad?"))
n=n+pluskey(r)   

r=int(input("Q.37 Get upset eaily"))
n=n+negkey(r)

r=int(input("Q.38 Have frequent mood swings"))
n=n+negkey(r)

r=int(input("Q.39 Remain calm under pressure"))
n=n+negkey(r)

r=int(input("Q.40 Get stressed out easily"))
n=n+negkey(r)

N=(n/8)*100 
    




#training_data and prediction algorithm
training_data=pd.read_csv(r'archive\train.csv')

training_data = training_data.values

type(training_data)
for i in range(len(training_data)):
 	if training_data[i][0]=="Male":
	     training_data[i][0]=1
 	else:
	     training_data[i][0]=0

training_data=pd.DataFrame(training_data)
training_data

X_train = training_data[[0,2,3,4,5,6]].values
X_train
y_train = training_data.iloc[:, -1].values
y_train

testing_data=pd.read_csv(r"archive\test.csv")
X_test = training_data[[0,2,3,4,5,6]].values
X_test
y_test = training_data.iloc[:, -1].values
y_test



DecisionTreeClassifier = DecisionTreeClassifier()
DecisionTreeClassifier.fit(X_train, y_train)
DecisionTreeClassifier.score(X_train, y_train)

y_pred =DecisionTreeClassifier.predict([[g,o,n,c,a,e]])
print(y_pred)

print('SGDClassifierModel Train Score is : ' , DecisionTreeClassifier.score(X_train, y_train))
print('SGDClassifierModel Test Score is : ' , DecisionTreeClassifier.score(X_test, y_test))
print('DecisionTreeClassifierModel Classes are : ' , DecisionTreeClassifier.classes_)
print('DecisionTreeClassifierModel feature importances are : ' , DecisionTreeClassifier.feature_importances_)



