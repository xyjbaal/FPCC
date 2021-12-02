import os
root_dir  = 'ring_train/'
note =''
for h5_name in os.listdir(root_dir):
	note = note+'./datas/'+root_dir+h5_name+'\n'

f = open('./ring_train.txt','w')
f.write(note)
f.close()
