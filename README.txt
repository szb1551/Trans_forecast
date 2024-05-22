numpy latest
matplotlib latest
pytorch 1.12 + cu113

Git 使用教程
git init
git add . 提交全部文件
git add main.py 提交单个文件 
git commit -m "加备注" 
git log 查看提交日志信息
git checkout HEAD  main.py 恢复文件
git clone + http网址    克隆网上文件

git rm -r --cached 删除已经提交的文件

git 删除大文件
从历史记录中删除
git log --pretty=oneline --branches -- your_dir/file
重写所有commit
git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch your_dir/file' -- --all

把文件引用删除
rm -Rf .git/refs/original
rm -Rf .git/logs/
git gc
git prune
到这里.git文件就应该明显变小了

git push --force
把之前上传到网上的库给替换掉


