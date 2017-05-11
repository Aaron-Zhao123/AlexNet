import os
import alexnet_train
import sys

def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(p['cov1'] * 10))
    name += 'cov' + str(int(p['cov2'] * 10))
    name += 'fc' + str(int(p['fc1'] * 100))
    name += 'fc' + str(int(p['fc2'] * 10))
    name += 'fc' + str(int(p['fc3'] * 10))
    return name

acc_list = []
count = 0
retrain = 0
parent_dir = 'assets/'
# lr = 1e-5
lr = 1e-4
crates = {
    'cov1': 0.,
    'cov2': 0.,
    'fc1': 1.70,
    'fc2': 0.,
    'fc3': 0.
}

retrain_cnt = 0
roundrobin = 0
with_biases = False
# Prune
while (crates['cov2'] < 2):
    count = 0
    iter_cnt = 0
    while (iter_cnt < 7):
        if (iter_cnt > 3 and iter_cnt < 5):
            lr = 5e-5
        elif (iter_cnt >= 5):
            lr = 1e-5
        else:
            lr = 1e-4
        param = [
            ('-cRates', crates),
            ('-first_time', False),
            ('-train', False),
            ('-prune', True),
            ('-lr', lr),
            ('-with_biases', with_biases),
            ('-parent_dir', parent_dir),
            ('-lambda1', 1e-5),
            ('-lambda2', 1e-5)
            ]
        # _ = train.main(param)

        # TRAIN
        param = [
            ('-cRates', crates),
            ('-first_time', False),
            ('-train', True),
            ('-prune', False),
            ('-lr', lr),
            ('-with_biases', with_biases),
            ('-parent_dir', parent_dir),
            ('-lambda1', 1e-5),
            ('-lambda2', 1e-5)
            ]
        # _ = train.main(param)

        # TEST
        param = [
            ('-cRates', crates),
            ('-first_time', False),
            ('-train', False),
            ('-prune', False),
            ('-test', True),
            ('-lr', lr),
            ('-with_biases', with_biases),
            ('-parent_dir', parent_dir),
            ('-lambda1', 1e-5),
            ('-lambda2', 1e-5)
            ]
        acc = train.main(param)

        sys.exit()

        if (acc > 0.823 or iter_cnt == 7):
            file_name = compute_file_name(crates)
            crates['fc1'] = crates['fc1'] + 0.05
            # crates['cov2'] = crates['cov2'] + 0.2
            # crates['fc2'] = crates['fc2'] + 0.5
            # crates['fc3'] = crates['fc3'] + 0.1
            # crates['cov2'] = crates['cov2'] + 0.5
            # crates['cov1'] = crates['cov1'] + 0.2
            acc_list.append((crates,acc))
            param = [
                ('-first_time', False),
                ('-train', False),
                ('-prune', False),
                ('-lr', lr),
                ('-with_biases', with_biases),
                ('-parent_dir', parent_dir),
                ('-iter_cnt',iter_cnt),
                ('-cRates',crates),
                ('-save', True),
                ('-lambda1', 1e-5),
                ('-lambda2', 1e-5),
                ('-org_file_name', file_name)
                ]
            _ = train.main(param)
            break
        else:
            iter_cnt = iter_cnt + 1
    print('accuracy summary: {}'.format(acc_list))


print('accuracy summary: {}'.format(acc_list))
