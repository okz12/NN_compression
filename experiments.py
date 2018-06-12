import subprocess
import time

ab_list = []
for alpha in [2500, 25000, 250000]:
    for beta in [0.1, 1, 10, 100]:
        ab_list.append((alpha, beta))

print (ab_list)

start_total = time.time()
for i in range(0,len(ab_list),3):
    start_batch = time.time()
    p1 = subprocess.Popen("python retrain.py --alpha {} --beta {}".format(ab_list[i][0], ab_list[i][1]).split(" "))
    p2 = subprocess.Popen("python retrain.py --alpha {} --beta {}".format(ab_list[i+1][0], ab_list[i+1][1]).split(" "))
    p3 = subprocess.Popen("python retrain.py --alpha {} --beta {}".format(ab_list[i+2][0], ab_list[i+2][1]).split(" "))
    while (p1.poll() is None and p2.poll() is None and p3.poll() is None):
        time.sleep(5)
    print ("Time Taken: {}s".format(time.time() - start_batch))
print ("Total Time Taken: {}s".format(time.time() - start_total))