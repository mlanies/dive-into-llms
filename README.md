<p align="center">
<h1 align="center">Серия практических руководств по программированию «Практикум по большим моделям»</h1>
</p>
<p align="center">
  	<a href="https://img.shields.io/badge/version-v0.1.0-blue">
      <img alt="version" src="https://img.shields.io/badge/version-v0.1.0-blue?color=FF8000?color=009922" />
    </a>
  <a >
       <img alt="Status-building" src="https://img.shields.io/badge/Status-building-blue" />
  	</a>
  <a >
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
  	</a>
   	<a href="https://github.com/Lordog/dive-into-llms/stargazers">
       <img alt="stars" src="https://img.shields.io/github/stars/Lordog/dive-into-llms" />
  	</a>
  	<a href="https://github.com/Lordog/dive-into-llms/network/members">
       <img alt="FORK" src="https://img.shields.io/github/forks/Lordog/dive-into-llms?color=FF8000" />
  	</a>
    <a href="https://github.com/Lordog/dive-into-llms/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/Lordog/dive-into-llms?color=0088ff"/>
    </a>
    <br />
</p>


<div align="center">
<p align="center">
  <a href="#-мотивация-проекта">Мотивация проекта</a>/
  <a href="#-содержание-руководств">Содержание руководств</a>/
  <a href="#%EF%B8%8F-список-участников">Список участников</a>
</p>
</div>

## 💡 Updates

2025/06/06  Спасибо всем за внимание и активную обратную связь! Мы обновили это руководство в двух направлениях:

- [x] Запущено отечественное (китайское) бесплатное руководство «Полный цикл разработки больших моделей» (с презентациями, методичками и видео); особая благодарность за поддержку сообществу Huawei Ascend!
- [x] Обновлено содержание исходной серии практических руководств по программированию и добавлены новые темы (математические рассуждения, GUI-агенты, выравнивание больших моделей, стеганография и др.)!

## 🎯 Мотивация проекта

Серия практических руководств по программированию «Практикум по большим моделям» выросла из конспектов курсов Шанхайского университета Цзяотун «Передовые технологии обработки естественного языка» (NIS8021) и «Технологии безопасности искусственного интеллекта» (NIS3353) (преподаватель: [Чжан Чжошэн](https://bcmi.sjtu.edu.cn/home/zhangzs/)) и призвана дать вводный ориентир по программированию для больших моделей. Это руководство имеет благотворительный характер и полностью бесплатно. Через простую практику оно помогает студентам быстро освоить большие модели, чтобы лучше выполнять курсовые проекты или вести научные исследования.

## 📚 Содержание руководств

| Содержание руководства | Краткое описание | Ссылки |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Дообучение и развёртывание | Руководство по дообучению и развёртыванию предобученных моделей: хотите повысить качество предобученной модели на конкретной задаче? Давайте выберем подходящую предобученную модель, дообучим её под задачу и развернём в виде удобного демо! | [[Слайды](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter1/dive-into-llm.pdf)] [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter1/README.md)] [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter1/dive-tuning.md)] |
| Промпт-обучение и цепочка рассуждений | Руководство по вызову API больших моделей и инференсу: «ИИ просит подбодрить его онлайн? Ответы больших моделей на некоторые вопросы порой ошеломляют, но, возможно, модели просто не хватает слова „поддержки“» | [[Слайды](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter2/dive-into-prompting.pdf)] [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter2/README.md)] [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter2/dive-prompting.md)] |
| Редактирование знаний | Методы и инструменты редактирования языковых моделей: хотите управлять тем, как языковая модель «помнит» определённые знания? Давайте выберем подходящий метод редактирования, отредактируем конкретное знание и проверим отредактированную модель! | [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter3/dive_edit_0410.pdf)] [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter3/README.md)]  [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter3/dive_edit.md)] |
| Математические рассуждения | Как научить большую модель математическим рассуждениям? Давайте быстро дистиллируем мини-R1! | [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter4/math.pdf)] [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter4/README.md)]  [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter4/sft_math.md)] |
| Водяные знаки в моделях | Текстовые водяные знаки в языковых моделях: встраивание в генерируемый моделью контент незаметных для человека водяных знаков | [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter5/watermark.pdf)] [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter5/README.md)]  [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter5/watermark.md)] |
| Джейлбрейк-атаки | Хотите добиться большей безопасности — начните с того, чтобы научиться атаковать. Давайте разберёмся, как джейлбрейк-атаки «развязывают язык» большим моделям! | [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter6/dive-Jailbreak.pdf)] [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter6/README.md)] [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter6/dive-jailbreak.md)] |
| Стеганография в больших моделях | «Невидимые чернила»! Хотите, чтобы большая модель, плавно отвечая, при этом незаметно несла информацию, распознаваемую только «своими»? Стеганография в больших моделях расскажет как! | [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter7/stega.pdf)] [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter7/README.md)] [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter7/llm_stega.md)] |
| Мультимодальные модели | Как мультимодальные большие языковые модели, способные более полно моделировать реальный мир, достигают более мощного мультимодального понимания и генерации? Могут ли они помочь в достижении AGI? | [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter8/mllms.pdf)]  [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter8/README.md)] [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter8/mllms.md)] |
| GUI-агент | Хотите есть, не вставая, и освободить руки? Тогда давайте вместе научим AI-агента заказывать вам еду, отвечать на сообщения и сравнивать цены при покупках! | [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter9/GUIagent.pdf)]  [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter9/README.md)] [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter9/GUIagent.md)] |
| Безопасность агентов | Агенты на основе больших моделей отправились в путь к операционной системе будущего. Но осознаёт ли большая модель угрозы и риски в открытых сценариях работы агента? | [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter10/dive-into-safety.pdf)] [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter10/README.md)] [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter10/agent.md)] |
| Безопасное выравнивание RLHF | Руководство по экспериментам RLHF на основе PPO: это руководство «очень опасное» — после прочтения проверьте, не ухмыляется ли ваша большая модель. | [[Слайды](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter11/RLHF.pdf)] [[Руководство](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter11/README.md)] [[Скрипт](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter11/RLHF.md)] |



## 🔥 Новинка: отечественный (китайский) курс «Полный цикл разработки больших моделей»

- **✨ Совместно с Huawei Ascend мы официально запустили бесплатный курс «Полный цикл разработки больших моделей»! Передовые технологии + практика на коде, пошагово осваиваем большие AI-модели ✨**: 

  На основе исходной серии руководств «Практикум по большим моделям» мы совместно с Huawei разработали серию курсов «Полный цикл разработки больших моделей». Эта серия основана на базовом программно-аппаратном обеспечении Ascend и охватывает форматы презентаций, методичек и видео. Курс делится на начальный, средний и продвинутый уровни, ориентирован на разные потребности в практике с большими моделями и призван через практику на коде дать исследователям и разработчикам полное руководство по разработке — от быстрого старта и применения уже поддерживаемых на Ascend моделей до миграции и тонкой настройки совершенно новых моделей, постепенно, от простого к сложному.
  
- **🚀 Перейдите в сообщество Ascend, чтобы изучить серию курсов «Полный цикл разработки больших моделей»**： 
  
  👉 «[Учебная зона по разработке больших моделей](https://www.hiascend.com/edu/growth/lm-development#classification-floor-1)» @ сообщество Ascend 👈 
  
- **✨ Демонстрация содержания курса ✨**

  <!-- <img src="./pics/icon/title.jpg" width="300"/>
  <img src="./pics/icon/cover.png" width="300"/>
  <img src="./pics/icon/team.png" width="300"/>
  <img src="./pics/icon/agent.png" width="300"/> -->

<p align = "center">
  <img src="./pics/icon/title.jpg" width="48%"/>
  <img src="./pics/icon/cover.png" width="48%"/>
  <img src="./pics/icon/team.png" width="48%"/>
  <img src="./pics/icon/agent.png" width="48%"/>
</p>

## 🙏 Отказ от ответственности

Всё содержание этого руководства основано лишь на личном опыте контрибьюторов, данных из интернета и наработках повседневной научной работы. Все приёмы приведены только для справки и не гарантируют стопроцентной правильности. При возникновении любых вопросов приветствуются Issue или PR. Кроме того, использованные в проекте бейджи взяты из интернета; если они нарушают ваши авторские права на изображения, свяжитесь с нами для удаления, спасибо.

## 🤝 Приглашаем к участию

Это руководство на данный момент находится в стадии разработки, и недочёты неизбежны; приветствуются любые PR и обсуждения в issue.

## ❤️ Список участников

Благодарим следующих преподавателей и студентов за поддержку и вклад в этот проект:

**Команда разработки серии руководств «Практикум по большим моделям»**:

- Шанхайский университет Цзяотун: [Чжан Чжошэн](https://bcmi.sjtu.edu.cn/home/zhangzs/), [Юань Тунсинь](https://github.com/Lordog), [Ма Синьбэй](https://scholar.google.com/citations?user=LpUi3EgAAAAJ&hl=zh-CN&oi=ao), [Хэ Чживэй](https://zwhe99.github.io), [Ду Вэй](https://scholar.google.com/citations?user=tFYUBLkAAAAJ&hl=en), [Чжао Хаодун](https://dongdongzhaoup.github.io/), [У Цзунжу](https://zrw00.github.io/), [У Чжэн](https://wuzheng02.github.io/), [Дун Линчжун](https://github.com/LZ-Dong), [Чжан Юйлун](https://aslan-yulong.github.io/)

- Национальный университет Сингапура: [Фэй Хао](http://haofei.vip/)

**Команда разработки серии руководств «Полный цикл разработки больших моделей»:**

- Шанхайский университет Цзяотун: [Чжан Чжошэн](https://bcmi.sjtu.edu.cn/home/zhangzs/), [Лю Гуншэнь](https://infosec.sjtu.edu.cn/DirectoryDetail.aspx?id=75), [Чэнь Синъюй](https://scholar.google.com/citations?user=d-dNtjrMJ5YC&hl=en), [Чэн Пэнчжоу](https://scholar.google.com/citations?user=qxnwzDUAAAAJ&hl=en), [Дун Линчжун](https://github.com/LZ-Dong), [Хэ Чживэй](https://zwhe99.github.io), [Цзюй Тяньцзе](https://scholar.google.com/citations?user=f8PPcnoAAAAJ&hl=en), [Ма Синьбэй](https://scholar.google.com/citations?user=LpUi3EgAAAAJ&hl=zh-CN&oi=ao), [У Чжэн](https://scholar.google.com/citations?hl=zh-CN&user=qBM1UbUAAAAJ&view_op=list_works&gmla=AIfU4H6PG9JyjRub6BYIIZ4isQE7MBAM3Eoec6OJfX4z_8-pOE8bI1Wgdo3XL5qOZWR3U-h-lIP2q0zXt5gzyFKMSg7MNnBBWLv5d1IVG30UANczTP0), [У Цзунжу](https://zrw00.github.io/), [Янь Цзыхэ](https://scholar.google.com/citations?user=O2YfSHoAAAAJ&hl=zh-CN), [Яо Яо](https://scholar.google.com/citations?user=tLMP3IkAAAAJ), [Юань Тунсинь](https://github.com/Lordog), [Чжао Хаодун](https://dongdongzhaoup.github.io/);

- Сообщество Huawei Ascend: ZOMI, Се Цянь, Чэн Лимин, Лоу Лихуа, Цзяо Цзэюй

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Lordog/dive-into-llms&type=Date)](https://star-history.com/#Lordog/dive-into-llms&Date)
