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

Check log and remove last commit:
```
git log --oneline
git reset --soft <HEAD~1>
git reset HEAD^
git status
```
How to solve a merge conflict, see [Resolving a merge conflict using the command line](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-using-the-command-line)

How to reset, revert, and return to previous states in Git, see [previous states](https://opensource.com/article/18/6/git-reset-revert-rebase-commands)

