##### to get the new exercises run set-template (needed only once) and then "make update"
set-template:
	git remote add template git@github.com:PDM4AR/exercises-template-2023hs-ex01.git

update:
	git pull -X theirs template master --allow-unrelated-histories
