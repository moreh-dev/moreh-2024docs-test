---
icon: terminal
tags: [tutorial, llama2]
order: 40
---

# 4. Checking Training Results

Upon running the **`train_llama2.py`** script as described earlier, the resulting model will be saved in the **`llama2_summarization`** directory. This model is a pure PyTorch parameter file and is fully compatible with regular GPU servers, not just the MoAI Platform.

You can test the trained model using the **`inference_llama2.py`** script located under the **`tutorial`** directory of the pre-downloaded GitHub repository.

For testing, articles related to English Premier League (EPL) match results have been used.

```python
# tutorial/inference_llama2.py
...
input_text = """[SUMMARIZE] (CNN)Arsenal kept their slim hopes of winning this season's English Premier League title alive by beating relegation threatened Burnley 1-0 at Turf Moor. A first half goal from Welsh international Aaron Ramsey was enough to separate the two sides and secure Arsenal's hold on second place. More importantly it took the north London club to within four points of first placed Chelsea, with the two clubs to play next week. But Chelsea have two games in hand and play lowly Queens Park Rangers on Sunday, a team who are themselves struggling against relegation. Good form . Arsenal have been in superb form since the start of the year, transforming what looked to be another mediocre season struggling to secure fourth place -- and with it Champions League qualification -- into one where they at least have a shot at winning the title. After going ahead, Arsenal rarely looked in any danger of conceding, showing more of the midfield pragmatism epitomized by the likes of Francis Coquelin, who also played a crucial role in the goal. "He has been absolutely consistent in the quality of his defensive work," Arsenal coach Arsene Wenger told Sky Sports after the game when asked about Coquelin's contribution to Arsenal's current run. They have won eight games in a row since introducing the previously overlooked young Frenchman into a more defensive midfield position. "He was a player who was with us for seven years, from 17, he's now just 24," Wenger explained. "Sometimes you have to be patient. I am very happy for him because he has shown great mental strength." Now all eyes will be on next week's clash between Arsenal and Chelsea which will likely decide the title. "They have the games in hand," said Wenger, playing down his club's title aspirations. "But we'll keep going and that's why the win was so important for us today." Relegation dogfight . Meanwhile it was a good day for teams at the bottom of the league. Aston Villa continued their good form since appointing coach Tim Sherwood with a 1-0 victory over Tottenham, who fired Sherwood last season. Belgian international Christian Benteke scored the only goal of the game, his eighth in six matches, to secure a vital three points to give the Midlands club breathing space. Another Midlands club looking over their shoulder is West Brom, who conceded an injury time goal to lose 3-2 against bottom club Leicester City. But it was an awful day for Sunderland's former Dutch international coach Dick Advocaat, who saw his team lose 4-1 at home against form team Crystal Palace. Democratic Republic of Congo international Yannick Bolasie scored Crystal Palace's first ever hat trick in the Premier League to secure an easy victory. [/SUMMAIRZE]"""
```
Run the train script.

```bash
~/quickstart$ python tutorial/inference_llama2.py
```

From the output, you'll notice that Llama2 has appropriately summarized the contents of the prompt.

```
Llama2: [SUMMARIZE] (CNN)Arsenal kept their slim hopes of winning this season's English Premier League title alive by beating relegation threatened Burnley 1-0 at Turf Moor. A first half goal from Welsh international Aaron Ramsey was enough to separate the two sides and secure Arsenal's hold on second place. More importantly it took the north London club to within four points of first placed Chelsea, with the two clubs to play next week. But Chelsea have two games in hand and play lowly Queens Park Rangers on Sunday, a team who are themselves struggling against relegation. Good form . Arsenal have been in superb form since the start of the year, transforming what looked to be another mediocre season struggling to secure fourth place -- and with it Champions League qualification -- into one where they at least have a shot at winning the title. After going ahead, Arsenal rarely looked in any danger of conceding, showing more of the midfield pragmatism epitomized by the likes of Francis Coquelin, who also played a crucial role in the goal. "He has been absolutely consistent in the quality of his defensive work," Arsenal coach Arsene Wenger told Sky Sports after the game when asked about Coquelin's contribution to Arsenal's current run. They have won eight games in a row since introducing the previously overlooked young Frenchman into a more defensive midfield position. "He was a player who was with us for seven years, from 17, he's now just 24," Wenger explained. "Sometimes you have to be patient. I am very happy for him because he has shown great mental strength." Now all eyes will be on next week's clash between Arsenal and Chelsea which will likely decide the title. "They have the games in hand," said Wenger, playing down his club's title aspirations. "But we'll keep going and that's why the win was so important for us today." Relegation dogfight . Meanwhile it was a good day for teams at the bottom of the league. Aston Villa continued their good form since appointing coach Tim Sherwood with a 1-0 victory over Tottenham, who fired Sherwood last season. Belgian international Christian Benteke scored the only goal of the game, his eighth in six matches, to secure a vital three points to give the Midlands club breathing space. Another Midlands club looking over their shoulder is West Brom, who conceded an injury time goal to lose 3-2 against bottom club Leicester City. But it was an awful day for Sunderland's former Dutch international coach Dick Advocaat, who saw his team lose 4-1 at home against form team Crystal Palace. Democratic Republic of Congo international Yannick Bolasie scored Crystal Palace's first ever hat trick in the Premier League to secure an easy victory. [/SUMMAIRZE]
Arsenal beat Burnley 1-0 in the English Premier League.
Aaron Ramsey scores the only goal of the game.
Arsenal remain in second place.
Chelsea can extend their lead to seven points.
```
