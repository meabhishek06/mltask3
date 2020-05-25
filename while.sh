pre=0.98
while [$i -lt 8]
do
 var=$(sudo cat /root/task3/acc.txt )
 echo $var
 if [$var -lt $pre]
 then
  sudo docker exec  d1 python3.7 /train/mnist$i.py
  var=$(sudo cat /root/task3/acc.txt )
  i=$(( $i + 1 ))
 fi
 done
