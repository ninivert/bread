# gh-pages-template

git repo to clone as a branch to create gh-pages

```
git clone https://github.com/ninivert/gh-pages-template.git --branch gh-pages --single-branch gh-pages
cp -r docs/build/html/* gh-pages/
cd gh-pages
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"
git add .
git commit -m "Update documentation" -a || true
```

Adapted from https://github.com/ammaraskar/sphinx-action-test/tree/gh-pages
