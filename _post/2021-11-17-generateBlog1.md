```
title: GIthub page 블러그 만들기

categorires:
- blog
tags:
- blog
- github page
-jetkll
last_modified-at:2021-11-17
```

# jekyll blog 생성 

[참조](https://jinhoooooou.github.io/making-blog/making-blog-3/)

1. 새로운 폴더에 블로그 생성 
```
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog$ jekyll new HelloBlog
```
대상 폴더에 웹사이트가 생성되었습니다. 

2. 그 디렉토리에서 명령 ``bundle exec jekyll serve``으로 웹사이트 호스팅합니다. 
```
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog$ cd HelloBlog
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog$ bundle exec jekyll serve
```
브라우저에 http://127.0.0.1:4000/ 을 입력하면 기본 화면이 출력됩니다. 

![기본화면](./image/jekyll기본화면.png)

## jekyll  theme  선택 

** minimal-mistakes 테마 사용**

다른 테마들에 대한 참조 사이트 

[사이트1](http://jekyllthemes.org/)

[사이트2](https://jekyllthemes.io/)

1. [minimal-mistakes](https://github.com/mmistakes/minimal-mistakes) 가져오기 

minimal-mistakes repo를 clone하여 가져오기 

```
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog$ git clone https://github.com/mmistakes/minimal-mistakes.git
```
<span style="color:blue"> 'minimal-mistakes'에 복제합니다...<br/>
     remote: Enumerating objects: 18691, done.<br/>
     remote: Total 18691 (delta 0), reused 0 (delta 0), pack-reused 18691<br/>
     오브젝트를 받는 중: 100% (18691/18691), 44.66 MiB | 9.62 MiB/s, 완료.<br/>
     델타를 알아내는 중: 100% (11136/11136), 완료.</span>

2. clone으로 소스다운후 해당 폴더에서 bundle 명령 수행으로 gemfile을 검사하고 필요한 목록을 설치합니다. 
```
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog$ cd minimal-mistakes
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog/minimal-mistakes$ bundle 
```
<span style="color:blue">Fetching gem metadata from https://rubygems.org/..........</span>

3. 설치후 웹호스팅합니다. (이전의 항목 2 참조) 
```
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog/minimal-mistakes$ bundle exec jekyll serve
```
<span style="color:blue"> Configuration file: /home/son/Jekyll_Blog/HelloBlog/minimal-mistakes/_config.yml
            Source: /home/son/Jekyll_Blog/HelloBlog/minimal-mistakes
       Destination: /home/son/Jekyll_Blog/HelloBlog/minimal-mistakes/_site
 Incremental build: disabled. Enable with --incremental
      Generating... 
       Jekyll Feed: Generating feed for posts
                    done in 0.227 seconds.
 Auto-regeneration: enabled for '/home/son/Jekyll_Blog/HelloBlog/minimal-mistakes'
    Server address: http://127.0.0.1:4000
  Server running... press ctrl-c to stop.</span>

![기본화면2](./image/jekyll기본화면2.png)

4. 3에서 생성한 웹호스팅은 로컬 서버에서만 사용할 수 있습니다. github에서 사용할 수 있도록 합니다. 이를 위해 github에 존재하는 repository와 연결합니다. 이 저장소의 이름은 enshs.github.io이므로 minimal-mistakes의 폴더명을 enshs.github.io로 변경합니다. 
```
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog$ mv minimal-mistakes enshs.github.io
```
그 폴더로 이동하여 로컬의 폴더와 github의 저장소 enshs.github.io와 연결합니다. 
```
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog/enshs.github.io$ git remote -v
```
<span style="color:blue">origin	https://github.com/mmistakes/minimal-mistakes.git (fetch)<br/>
origin	https://github.com/mmistakes/minimal-mistakes.git (push)</span>
```
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog/enshs.github.io$ git remote remove origin
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog/enshs.github.io$ git remote add origin https://github.com/enshs/enshs.github.io
(base) son@son-HP-ENVY-Notebook-15-PC:~/Jekyll_Blog/HelloBlog/enshs.github.io$ git push -u origin master
Username for 'https://github.com': enshs
Password for 'https://enshs@github.com':
```
<span style="color:blue">오브젝트 나열하는 중: 16701, 완료.<br/>
오브젝트 개수 세는 중: 100% (16701/16701), 완료.<br/>
Delta compression using up to 4 threads<br/>
오브젝트 압축하는 중: 100% (6429/6429), 완료.<br/>
오브젝트 쓰는 중: 100% (16701/16701), 43.96 MiB | 7.34 MiB/s, 완료.<br/>
Total 16701 (delta 10001), reused 16701 (delta 10001), pack-reused 0<br/>
remote: Resolving deltas: 100% (10001/10001), done.</span>

위 password는 github의 가입시 설정한 비밀번호가 아닌 github 자체에서 발급한 개인용 token입니다. 이는 다음의 절차로 생성하여 사용할 수 있습니다. 

자신의 github의 홈에서 오른쪽 상단에 있는 toggle을 사용하여 다음 과정으로 획득할 수 있습니다. 

$$\text{설정} \rightarrow \text{개발자 설정} \rightarrow \text{Personal acess tokens}$$

![개인용토큰1](./image/github_setting1.png)



