import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


'''
#在线获取数据 
from ucimlrepo import fetch_ucirepo 
solar_flare = fetch_ucirepo(id=89)

X = solar_flare.data.features 
y = solar_flare.data.targets
print(X.columns, y.columns)
'''

#从本地加载数据
df1 = pd.read_csv('/home/AIIAcourse/course/231503031/flare_prediction/data/flare.data1', sep = '\s+', header = None, skiprows = 1 )
df2 = pd.read_csv('/home/AIIAcourse/course/231503031/flare_prediction/data/flare.data1', sep = '\s+', header = None, skiprows = 1 )
df = pd.concat([df1, df2], ignore_index=True)
df.columns = ['modified Zurich class', 'largest spot size', 'spot distribution',
       'activity', 'evolution', 'previous 24 hour flare activity',
       'historically-complex', 'became complex on this pass', 'area',
       'area of largest spot', 'common flares', 'moderate flares', 'severe flares']
X = df.iloc[:, :-3]
y = df.iloc[:, -3:]




#样本分布

print((y != 0).any(axis = 1).value_counts())
y1 = y[(y == 0).all(axis = 1)]
X1 = X[(y == 0).all(axis = 1)]
y2 = y[(y != 0).any(axis = 1)]
X2 = X[(y != 0).any(axis = 1)]
X = X2
y = y2

'''
#对y中每一列的数据加权求和，结果为dataframe，权重为0.1,0.3,0.6
weights = [0.1, 0.3, 0.6]
y = (y * weights).sum(axis = 1)
print(X.head(), y.head())
print(X.shape, y.shape)
'''

#用独热编码处理非数值型数据
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

#数据清洗
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

#划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)     
        self.dropout2 = nn.Dropout(0.2)      
        self.fc3 = nn.Linear(64, 32) 
        self.fc4 = nn.Linear(32, output_size)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)            
        return x

def train(epochs, early_stop = False, sample = False, patience_threshold=10):
    print(f'Training on {device}')
    model = NeuralNet(input_size=X_train.shape[1], output_size=y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    num_epochs = epochs
    train_loss_list = []
    val_loss_list = []
    accuracy_list = []
    best_loss = float('inf')
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        train_loss_list.append(loss.item())
        loss.backward() 
        optimizer.step() 

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            model.eval()
            with torch.no_grad():
                y_pred_tensor = model(X_test_tensor)
                y_pred = y_pred_tensor.cpu().numpy()
                y_pred = np.round(y_pred).astype(int)
            
            #计算准确率
            
            right = 0
            for index in X_test.index:
                true_value = y_test.loc[index].values
                predicted_value = y_pred[np.where(X_test.index == index)[0][0]]  
                right += np.count_nonzero(true_value == predicted_value)/len(true_value)
            accuracy = right / len(X_test)
            accuracy_list.append(accuracy)  
            print(f'Accuracy: {accuracy:.4f}')
            '''
            #对输出长度为1的情况:计算准确率
            right = 0
            for index in X_test.index:
                true_value = y_test.loc[index]
                predicted_value = y_pred[np.where(X_test.index == index)[0][0]]  
                right += 1 if true_value == predicted_value else 0
            accuracy = right / len(X_test)
            accuracy_list.append(accuracy)  
            print(f'Accuracy: {accuracy:.4f}')
            '''
        #早停
        if early_stop:
            model.eval()
            with torch.no_grad(): 
                y_pred_tensor = model(X_val_tensor) 
                val_loss = criterion(y_pred_tensor, y_val_tensor)
                val_loss_list.append(val_loss.item())

            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                patience = 0
            else:
                    patience += 1
            
            if patience == patience_threshold:
                print(f'Early stopping at epoch {epoch+1}, loss: {loss.item():.4f}, val_loss: {val_loss.item():.4f}, best_loss: {best_loss:.4f}')
                break
    
    #在test集上抽样
    if sample:
        sample_indices = np.random.choice(X_test.index, size=5, replace=False) 
        for index in sample_indices:
            features = X_test.loc[index]
            true_value = y_test.loc[index].values
            predicted_value = y_pred[np.where(X_test.index == index)[0][0]]  
            print(f'Features: {features}, True Value: {true_value}, Predicted Value: {predicted_value}')


        '''
    #对输出长度为1:抽样测试
    if sample:
        sample_indices = np.random.choice(X_test.index, size=10, replace=False) 
        for index in sample_indices:
            features = X_test.loc[index]
            true_value = y_test.loc[index]
            predicted_value = y_pred[np.where(X_test.index == index)[0][0]]  
            print(f'Features: {features}, True Value: {true_value}, Predicted Value: {predicted_value}')
    '''
        
    #在test集上计算表现
    no_right_num, part_right_num, almost_right_num, right_num = 0, 0, 0, 0
    for index in X_test.index:
        true_value = y_test.loc[index].values
        predicted_value = y_pred[np.where(X_test.index == index)[0][0]] 
        accuracy = np.count_nonzero(true_value == predicted_value)/len(true_value)

        if accuracy == 0:
            no_right_num +=1
        if 0 < accuracy < 0.4:
            part_right_num +=1
        if  0.4 <= accuracy < 0.7:
            almost_right_num +=1
        if accuracy >= 0.7:
            right_num +=1
    
        
    print(f'No right: {no_right_num}, Part right: {part_right_num}, Almost right: {almost_right_num}, Right: {right_num}')   



    return train_loss_list, val_loss_list, accuracy_list, [no_right_num, part_right_num, almost_right_num, right_num]

def try_diff_epochs():

    loss_list_list = []
    accuracy_indexs = []

    epoch_values = np.arange(10, 100, 10)

    for epochs in epoch_values:
        loss_list, val_loss_list, accuracy_list, right_num_list = train(epochs) 
        accuracy_indexs.append(np.sum(0.2*right_num_list[1]+0.3*right_num_list[2]+0.5*right_num_list[3])/(np.sum(right_num_list[i] for i in range(1,4))))  # 保存每个 epoch 的准确率
        loss_list_list.append(loss_list)

    # 绘制不同epoch下accuracy的变化图
    plt.plot(epoch_values, accuracy_indexs)
    plt.xlabel('Epochs_num')
    plt.ylabel('Accuracy index')
    plt.grid(True)
    plt.savefig('/home/AIIAcourse/course/231503031/flare_prediction/pics/Accuracyindex_Epochsnum.png')
    plt.close()
    
    plt.subplot(212)
    # 绘制不同epoch下loss的变化图,每隔一个epoch值画一次
    for i in range(0, len(epoch_values), 2):
        plt.plot(loss_list_list[i], label=f'Epochs: {epoch_values[i]}')
    plt.xlabel('Epochs_num')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/AIIAcourse/course/231503031/flare_prediction/pics/Loss_Epochsnum.png')
    plt.close()

def one_epoch_train(epochs):
    epochs = epochs
    loss_list, val_loss_list, accuracy_list, right_num_list = train(epochs,sample=True)
    
    plt.subplot(311)
    plt.plot(loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs_num')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(312)
    plt.plot(accuracy_list, label='Accuracy')
    plt.xlabel('Epochs_num')
    plt.ylabel('Accuracy')
    x_ticks = np.arange(0, len(accuracy_list), 5)
    plt.xticks(ticks=x_ticks, labels=x_ticks * 10)
    plt.legend()
    plt.grid(True)

    plt.subplot(313)
    plt.bar(['No right', 'Part right', 'Almost right', 'Right'], right_num_list, label='Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Number')
    plt.legend()
    plt.grid(True)
  
    plt.tight_layout()
    plt.show()

def main():
    try_diff_epochs()


if __name__ == '__main__':
    main()











