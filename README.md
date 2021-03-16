# Tree_generation

## git commands

Check/Switch/Delete branch and create new branch:
```
git checkout [branch_name]
git checkout -a
git checkout -b [name_of_your_new_branch]
git branch -d/-D(force delete) [branch_name]  (local)
git push origin --delete [branch_name]  (remote)
```
Push code to your working branch
```
git add [file_name or “.” for all file within current dir]
git commit -m “commit_message”
git push origin [branch_name]

git pull (to get updates from remove)

```

Merge with main (View pull request on github.com and get someone to review the code):
```
git checkout main
git merge [branch_name]
```




