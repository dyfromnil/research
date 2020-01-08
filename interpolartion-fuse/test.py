import re



a='slajf_sjdf_1_2_5aslkdjf_6sjkf.os'

split=re.split("\D+",a)
del split[0];del split[len(split)-1]

image_A_path='aklsjhf/33_19_10.png'
split_str=re.split("\D+",image_A_path)
del split_str[0];del split_str[len(split_str)-1]

image_B_name=split_str[0]+'_'+split_str[1]+'_'+str(int(split_str[2])+1)+'.png'