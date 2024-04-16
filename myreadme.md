# git 配置
1. 生成秘钥信息
```
ssh-keygen -t rsa
```
2. 设置github key
'头像 -> setting -> SSH and GPG keys'


# 代理
## 代理设置
export http_proxy=http://192.168.200.26:51837  
export https_proxy=http://192.168.200.26:51837  

## 取消代理
unset https_proxy  
unset http_proxy  

## 设置git 代理
git config --global https.proxy http://192.168.200.26:51837  
git config --global http.proxy http://192.168.200.26:51837  

# 取消git 代理
git config --global --unset http.proxy
git config --global --unset https.proxy