# 三个可以直接读的例子

这些是看过结果后为说明评分选的例子，不是独立测试或总体比例。完整小批次均在experiments中。

## 编辑：粘贴也可能比原生正确

案例：artifact-book-0 / reorder_pages

当前请求：

In our latest book, reverse the order of only the first three pages. Keep all eight page strings exactly unchanged. Return the complete JSON book.

指标依据是实际冻结成果加本次修改要求；下面保持原文。

### N：完整修改失败

~~~~text
```json
{
  "pages": [
    "Inside, he warmed the gull by the fire, watching it regain its strength.",
    "Keeper Tom opened the door and gently scooped up the tired gull.",
    "The gull followed the beam, circling closer to the lighthouse.",
    "The friends rested beneath a bright blue umbrella.",
    "From the lighthouse, Keeper Tom watched the gull's erratic flight.",
    "He grabbed his lantern and stepped out onto the tower's balcony.",
    "With a soft glow, he shone the light towards the confused bird.",
    "A small gull flew aimlessly, lost in the night sky."
  ]
}
```
~~~~

### P：完整修改通过

~~~~text
```json
{
  "pages": [
    "From the lighthouse, Keeper Tom watched the gull's erratic flight.",
    "A small gull flew aimlessly, lost in the night sky.",
    "The friends rested beneath a bright blue umbrella.",
    "He grabbed his lantern and stepped out onto the tower's balcony.",
    "With a soft glow, he shone the light towards the confused bird.",
    "The gull followed the beam, circling closer to the lighthouse.",
    "Keeper Tom opened the door and gently scooped up the tired gull.",
    "Inside, he warmed the gull by the fire, watching it regain its strength."
  ]
}
```
~~~~

### Hint：完整修改通过

~~~~text
```json
{
  "pages": [
    "From the lighthouse, Keeper Tom watched the gull's erratic flight.",
    "A small gull flew aimlessly, lost in the night sky.",
    "The friends rested beneath a bright blue umbrella.",
    "He grabbed his lantern and stepped out onto the tower's balcony.",
    "With a soft glow, he shone the light towards the confused bird.",
    "The gull followed the beam, circling closer to the lighthouse.",
    "Keeper Tom opened the door and gently scooped up the tired gull.",
    "Inside, he warmed the gull by the fire, watching it regain its strength."
  ]
}
```
~~~~

### D：完整修改通过

~~~~text
```json
{
  "pages": [
    "From the lighthouse, Keeper Tom watched the gull's erratic flight.",
    "A small gull flew aimlessly, lost in the night sky.",
    "The friends rested beneath a bright blue umbrella.",
    "He grabbed his lantern and stepped out onto the tower's balcony.",
    "With a soft glow, he shone the light towards the confused bird.",
    "The gull followed the beam, circling closer to the lighthouse.",
    "Keeper Tom opened the door and gently scooped up the tired gull.",
    "Inside, he warmed the gull by the fire, watching it regain its strength."
  ]
}
```
~~~~

## 策略：双版本变成单版本

历史约定：All your responses in the upcoming conversation must have two different responses. Your two different responses should be separated by 6 asterisk symbols: ******. Keep all your responses under 200 words.

当前请求：How do vaccines work and why are they important?

前三个实际任务回复都通过类别checker；这条最终请求的具体例子如下。

### AOnline

~~~~text
Vaccines work by training your immune system to recognize and fight specific diseases. They contain weakened or dead pathogens, or parts of them, which stimulate your immune system to produce antibodies. This way, if you encounter the actual disease, your body is already prepared to fight it off.

Vaccines are crucial because they prevent the spread of diseases, protect vulnerable populations, and reduce the burden on healthcare systems. By getting vaccinated, you not only protect yourself but also help create herd immunity, which protects those who cannot be vaccinated due to age or health conditions.

******

Vaccines work by introducing a harmless form of a virus or bacteria into your body, which triggers your immune system to create antibodies. These antibodies help your body recognize and fight off the actual disease if you are exposed. They are important because they prevent the spread of diseases, protect vulnerable populations, and reduce the strain on healthcare resources. By getting vaccinated, you contribute to the overall health and safety of your community.
~~~~

### BReplay

~~~~text
Vaccines work by training your immune system to recognize and fight specific diseases. They contain weakened or dead pathogens, or parts of them, which stimulate your immune system to produce antibodies. This way, if you encounter the actual disease, your body is already prepared to fight it off.

Vaccines are crucial because they prevent the spread of diseases, protect vulnerable populations, and reduce the burden on healthcare systems. By getting vaccinated, you not only protect yourself but also help create herd immunity, which protects those who cannot be vaccinated due to age or health conditions.

******

Vaccines work by introducing a harmless form of a virus or bacteria into your body, which triggers your immune system to produce antibodies. These antibodies help your body recognize and fight off the actual disease if you are exposed. Vaccines are important because they prevent the spread of diseases, protect vulnerable populations, and reduce the strain on healthcare resources. By getting vaccinated, you contribute to the overall health and safety of your community.
~~~~

### CPasted

~~~~text
Vaccines work by training your immune system to recognize and fight off specific diseases. They contain weakened or dead pathogens, which stimulate your immune system to produce antibodies without causing the full-blown disease. This way, if you encounter the actual pathogen later, your immune system is already prepared to fight it off.

Vaccines are crucial because they protect individuals and communities by preventing the spread of diseases. They reduce the burden on healthcare systems and save lives. By getting vaccinated, you not only protect yourself but also help create herd immunity, which protects those who cannot be vaccinated, such as infants and people with certain medical conditions.
~~~~

### CSpecificRule

~~~~text
Vaccines work by introducing a weakened or inactivated form of a virus or bacteria into the body. This triggers an immune response, which helps the body recognize and fight off the actual virus if exposed in the future. They are crucial because they prevent the spread of diseases and protect individuals who might not be able to get vaccinated due to health reasons. ******

Vaccines work by priming the immune system to recognize and fight specific pathogens. They are important because they not only protect the individual but also contribute to herd immunity, reducing the overall spread of diseases in a community.
~~~~

后续同两历史的新请求B/C都9/10，所以这个单点不能写成稳定大批现象。

## 风格：新四历史方向不一致

|新历史|完整文本C−B缩略语密度|
|---|---:|
|confirm-00-casual|-0.30177|
|confirm-01-casual|+0.00000|
|confirm-02-casual|+0.09724|
|confirm-03-casual|+1.84346|

新批平均仍为正，但只有两条为正。必须同时看分历史结果和实际命中总量，不能只看平均值。
