import matplotlib.pyplot as plt
import os

from model import *


def subject_dependent_train():
    dataset = DeapDataset('./data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './model/'

    acc_list = []
    f_score_list = []
    step = 1
    for data in dataset:
        print(f'-------------subject: {step}-------------')

        save_path = save_dir + f'model_{step}.pt'
        if os.path.exists(save_path):
            print('model already exists, loading...')
            trained_model = torch.load(save_path)
            print('model loaded.')
        else:
            model = ECLGCNN(K=2, T=6, num_cells=30)
            trained_model = train(model, device, data, batch_size=40, max_step=100000,
                                  e=0.1, lr=0.003, alpha=0.0008)
            torch.save(trained_model, save_path)
            print('model saved.')

        acc, f_score = validate(trained_model, device, data)
        acc_list.append(acc)
        f_score_list.append(f_score)
        step += 1

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


if __name__ == '__main__':
    subject_dependent_train()
