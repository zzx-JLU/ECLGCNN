import matplotlib.pyplot as plt
import os

from model import *


def subject_dependent_train():
    dataset = DeapDataset('./data')
    train_data, train_labels = dataset[15]  # 取 No.16 受试者的数据来训练模型

    model = ECLGCNN(K=2, T=6, num_cells=30)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_file = './model/dependent_model.pt'
    if os.path.exists(model_file):
        print('model already exists, loading...')
        trained_model = torch.load(model_file)
        print('model loaded.')
    else:
        trained_model = train(model, device, train_data, train_labels, max_step=100000,
                              e=0.1, lr=0.003, alpha=0.0008)
        torch.save(trained_model, model_file)
        print('model saved.')

    acc_list = []
    f_score_list = []
    for data, label in dataset:
        acc, f_score = validate(trained_model, device, data, label)
        acc_list.append(acc)
        f_score_list.append(f_score)

    acc_array = np.array(acc_list)
    valence_acc = acc_array[:, 0]
    arousal_acc = acc_array[:, 1]

    f_score_array = np.array(f_score_list)
    valence_f_score = f_score_array[:, 0]
    arousal_f_score = f_score_array[:, 1]

    print('---------RESULT---------')
    print('valence')
    print(f'  acc: {valence_acc}')
    print(f'  average acc: {valence_acc.mean()}')
    print(f'  f-score: {valence_f_score}')
    print(f'  average f-score: {valence_f_score.mean()}')
    print('arousal')
    print(f'  acc: {arousal_acc}')
    print(f'  average acc: {arousal_acc.mean()}')
    print(f'  f-score: {arousal_f_score}')
    print(f'  average f-score: {arousal_f_score.mean()}')

    fig1 = plt.figure(figsize=(10, 5))
    axes1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    axes1.plot(valence_acc, 'rs-')  # 红色，正方形点，实线
    axes1.plot(valence_f_score, 'bo--')  # 蓝色，圆点，虚线
    axes1.legend(labels=('acc', 'f-score'), loc='lower right')
    axes1.set_title('valence')
    fig1.savefig('./imgs/dependent_valence.png')
    fig1.show()

    fig2 = plt.figure(figsize=(10, 5))
    axes2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    axes2.plot(arousal_acc, 'rs-')  # 红色，正方形点，实线
    axes2.plot(arousal_f_score, 'bo--')  # 蓝色，圆点，虚线
    axes2.legend(labels=('acc', 'f-score'), loc='lower right')
    axes2.set_title('arousal')
    fig2.savefig('./imgs/dependent_arousal.png')
    fig2.show()

    # kfold = ShuffleSplit(n_splits=5, random_state=0)
    #
    # fold_acc = []
    # fold_f_score = []
    #
    # for epoch in range(3):
    #     print(f'===epoch {epoch}===')
    #     acc_list = []
    #     f_score_list = []
    #     for i, (train_index, val_index) in enumerate(kfold.split(train_subject, labels)):
    #         train_data = []
    #         for index in train_index:
    #             train_data.append(train_subject[index])
    #
    #         val_data = []
    #         for index in val_index:
    #             val_data.append(labels[index])
    #
    #         train_label = np.array(labels)[train_index]
    #         val_label = np.array(labels)[val_index]
    #
    #         model = ECLGCNN(K=2, T=6, num_cells=30)
    #         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #         trained_model = train(model, device, train_data, train_label, max_step=100000,
    #                               e=0.1, lr=0.003, alpha=0.0008)
    #         torch.save(trained_model, f'./model/dependent_model_epoch{epoch}_fold{i}.pt')
    #
    #         acc, f_score = validate(trained_model, device, val_data, val_label)
    #         acc_list.append(acc)
    #         f_score_list.append(f_score)
    #         print(f'  fold {i}: acc = {acc}, f_score = {f_score}')
    #     fold_acc.append(np.array(acc_list).mean())
    #     fold_f_score.append(np.array(f_score_list).mean())
    #
    # avg_acc = np.array(fold_acc).mean()
    # avg_f_score = np.array(fold_f_score).mean()


if __name__ == '__main__':
    subject_dependent_train()
