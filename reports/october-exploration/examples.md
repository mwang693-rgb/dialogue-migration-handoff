# Examples

These examples were picked after inspection to make the scoring easier to follow. Complete selected batches are in `experiments/`.

## An edit where pasting helped

`artifact-book-0 / reorder_pages`

In our latest book, reverse the order of only the first three pages. Keep all eight page strings exactly unchanged. Return the complete JSON book.

The expected answer was built from the saved artifact and this edit request. Replies below are unchanged.

### B: role replay: fail

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

### C: pasted: pass

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

### C + continuation hint: pass

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

### C + history-difference vector: pass

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

## Two responses became one

History rule: All your responses in the upcoming conversation must have two different responses. Your two different responses should be separated by 6 asterisk symbols: ******. Keep all your responses under 200 words.

Follow-up: How do vaccines work and why are they important?

All three practice replies passed the category checker.

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

The later questions from these same histories gave B/C 9/10 each. This one example does not establish a general loss.

## The four new style histories

|History|C−B contraction density|
|---|---:|
|confirm-00-casual|-0.30177|
|confirm-01-casual|+0.00000|
|confirm-02-casual|+0.09724|
|confirm-03-casual|+1.84346|

The average remains positive, but only two histories are positive. The per-history results and raw occurrence counts tell a different story from the mean alone.
