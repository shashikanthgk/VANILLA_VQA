import csv

for i in range(61):
    if (i+1)%5 == 0:
        with open('./logs/{}-log-epoch-{:02d}.txt'.format('train', i+1), 'r') as f:
            df = csv.reader(f, delimiter='\t')
            data1 = list(df) 
        with open('./logs/{}-log-epoch-{:02d}.txt'.format('valid', i+1), 'r') as f:
            df = csv.reader(f, delimiter='\t')
            data2 = list(df)
        print(str(i+1)+" & "+str(round(float(data1[0][2])*100,2))+" & "+str(round(float(data1[0][3])*100,2))+" & "+str(round(float(data2[0][2])*100,2))+" & "+str(round(float(data2[0][3])*100,2))+"\\\\")
        print("\hline")
