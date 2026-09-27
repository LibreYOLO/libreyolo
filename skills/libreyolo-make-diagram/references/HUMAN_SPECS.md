# Human Specs

**This file is owned by Xuban (the human). Agents may modify it only with explicit authorization for the specific change.**

The human's words are recorded verbatim. Typos, informality and profanity are intentional. Agent analysis and implementation rules belong in other files. Xuban requested this record in the same conversation on 5 September 2026.

Source: Xuban's Codex conversation about the LibreYOLO architecture diagram, 5 September 2026. Entries follow conversation order. Headings are editorial labels, not additional requirements.

## 1. Initial diagram request

> 2026-09-05, Xuban, initial request

Hey do you know that famous Yolo8 chart PNG that's been circulating and that people have loved? I would like you to experiment with whether we could take LibreYolo docs. You can clone that repo if you want somewhere.

I had this idea that maybe we could do something like this: every model at the bottom of the docs has this chart, right? The chart shows the blocks and you can click and so on. Also if you take a screenshot, it should still have every block, like the famous Yolo8 PNG chart that is around the web, the one that everybody references even in papers.

How hard do you think that would be? Also we could brand it with LibreYolo, meaning we can put the logo and the text and make it beautiful. If you could, work on that maybe start with something like Yolo9 or whatever model you think is easy. It wouldn't have to be like a rendered PNG but I'm thinking more of an interactive thing. It still has to show everything. Even if it's interactive it has to show all the information like the PNG does.

## 2. Block internals and the first attempt

> 2026-09-05, Xuban, feedback on the first poster

looks a bit ai sloppy-i don't you think? something is kind of wrong . can you make it look a little bit more like the yolo 8 chart partciluarly for hte "inside the blocks" part please

## 3. Text separators, colors and humanizer

> 2026-09-05, Xuban, feedback after the block diagrams were redrawn

Yes and now let's further not make it feel like AI by removing the points between, for example, "rep NCS plan" and "P3." There's a dot. That dot is very AI. Humans don't write like that.  Yeah this is so much better already. Maybe play a bit with the colors to look more like the Yolo8 diagram. But this is already so, so close to what I want. If you see any AI patterns remove them. For example take the humanizer skill and read the contents.

## 4. Unicode arrows and drawn arrows

> 2026-09-05, Xuban, final text correction

Little change: I don't like those long arrows. I'm talking about the Unicode arrows not the diagram arrows. Diagram arrows are fine.

## 5. Approval, documenting the skill, and model trials

> 2026-09-05, Xuban, response to the fourth poster revision

OH FUCKING YEAH??????? OMFG this is so so sos o freacking freacklishly beautiful omfg. Could you look at this conversation, document everything I said regarding what I like and what I don't like, then document the rules that you applied? What I need is a libreyolo-make diagram skill that's going to go on libreyolo. Should it go on libreyolo or should it go on the website? I think it should go on libreyolo.

We do this skill and then we practice using it with a few models. You launch sub-agents with a few models. I will try to see what happens. I think that this is the fucking feature honestly. In my opinion this could be interactive. Should this be interactive or a PNG? What do you think? Should this be an interactive thing on the website or a PNG?

## 6. Symbolic family diagrams and resolved variants

> 2026-09-05, Xuban, follow-up while the skill was being written

Should we, for family models where the size is just a variable, maybe create two versions of the diagram:
- The version with variables, where you insert size variables
- One version that's like the model, without the variants, just variables for the sizes of the variants, and then another version for each one of the variants, actually with the actual number from that variant burned in?
Do you think that makes sense, because the original Yolo8 diagram has all the sizes in it? For me it was a pain to try to understand how it looks in a specific size because you were having to replace the width and the depth and stuff like this. For me if we have both, that would be great.But you let me know, with your god-given taste, what we should do.

## 7. Closely spaced parallel arrows

> 2026-09-05, Xuban, feedback during the model trials

And there's a new edge case where some models have up to four arrows. I would say that we need to try different combinations because when you have four arrows so close, it looks very bad. We need to handle that situation. Like the situation where there are a lot of parallel arrows, super thin parallel arrows, because it looks really bad. When you zoom out it looks like one line. When you zoom in it glitches and suddenly looks like four thin lines. So definitely Definitely has to be addressed.

## 8. A palette closer to LibreYOLO's website

> 2026-09-05, Xuban, color feedback during the model trials

Just, regarding the colors, I know you're in the middle of working so once you have finished your tasks, I cannot stop noticing that they don't really make a lot of sense at the moment, which is fine. I don't know if we could make some code color or reduce the amount of colors because if you take a look at libreyolo.com, it has a certain color scheme: mostly blue, white, black, not much variation.

If you could find a color palette you could draw a lot of colors from and think about some rule, like the block that appears the most has to have this color and the second block this other color. Think about the color palette. That basically maybe makes it look a bit better because it's true that there's something weird about the colors, right? Not about what things have what color, but just the colors per se. Perhaps I'm wrong.

## 9. Rejection of the near-monochrome trial

> 2026-09-05, Xuban, response to the pale-blue/gray trial

it looks ugly as hell now hahaha if we need to pick i prefer hte old version huh. omfg lmao HORRIBLE sytling

## 10. Complementary colors, not monochrome

> 2026-09-05, Xuban, clarification of the palette request

i mean not horrible but whats wrong with the blue? Also I was expecting you to pick complementary colors to blue. Maybe an orange, maybe a red, or whatever. As an expert in colors you can pick what's complementary.

## 11. Final palette choice and generalization question

> 2026-09-05, Xuban, after comparing the earlier and complementary colors

I actually prefer the earlier colors so put that on the skill. By the way are you confident that the skill would be able to create arbitrary charts that are coherent with what we've been working on?
