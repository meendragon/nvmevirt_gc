#!/bin/sh

sudo mkfs.ext4 -F /dev/nvme0n1
#파일시스템을 만든다 - 리눅스에서 가장 많이 쓰이는거 ㅇㅇ 해당 드라이브에 아이노드라든지 수퍼블록같은 메타데이터
sudo mount /dev/nvme0n1 /home/meen/nvmevirt_gc/mnt
sudo chown meen:meen /home/meen/nvmevirt_gc/mnt
