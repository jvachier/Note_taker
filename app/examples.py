"""Hand-written demo inputs (not from any dataset)."""

CLEANUP = {
    "Voice note": (
        "hi sarah its tom i just got out of the meeting with the client they loved the new "
        "design but they want the logo bigger can you send me the updated files by friday "
        "also did you book the room for tuesday let me know thanks"
    ),
    "Meeting transcript": (
        "okay so lets start with the budget we spent about twenty percent more than planned "
        "on marketing in march john do you know why that happened i think it was the new "
        "campaign on instagram we should check the numbers with the finance team next week"
    ),
    "Chat message": "are you coming to the party tonight i can pick you up at eight if you want",
}

DIALOGUES = {
    "Sprint planning (longer)": """\
Maria: Okay, let's start. Today we plan the next sprint and agree on the release.
Tom: Before that, a quick status. The search feature is done, but the export to PDF is still broken.
Maria: How bad is the export problem?
Tom: It crashes on files over ten megabytes. About five percent of our customers hit it.
Priya: Support got twelve tickets about it last week, so it is hurting us.
Maria: Then the export fix comes first in this sprint.
Tom: Agreed. I can fix it, but I need help testing on large files.
Priya: I will prepare a set of large test files by Wednesday.
Maria: Good. What about the new onboarding screens?
Priya: Design is ready, but we have not tested them with users yet.
Tom: Maybe we could skip the user tests this time?
Maria: No, we keep the user tests. Priya, can you book three sessions for next week?
Priya: Yes, I will book them today.
Maria: So we release on the 30th, if the export fix is done by the 25th.
Tom: That works for me.
Maria: Great, thanks everyone. See you on Thursday.""",
    "Project meeting": """Anna: Thanks for joining. We need to decide on the launch date today.
Ben: The app is almost ready, but the payment page still has two bugs.
Anna: How long do you need to fix them?
Ben: About a week, if nobody pulls me onto other work.
Chloe: Marketing can be ready by the 15th, but not earlier.
Anna: Then let's launch on the 15th. Ben, please fix the payment bugs by Friday.
Ben: Okay, I'll send an update on Wednesday.
Chloe: I'll book the newsletter for the morning of the 15th.
Anna: Great. Anything else? No? Thanks, everyone.""",
    "Doctor's appointment": """Receptionist: Good morning, Riverside Clinic, how can I help?
Paul: Hi, I'd like to book an appointment with Dr. Lee. I've had a cough for two weeks.
Receptionist: Dr. Lee is free on Thursday at 10 or Friday at 3.
Paul: Thursday at 10 works for me.
Receptionist: Fine. Please bring your insurance card and arrive ten minutes early.
Paul: Will do. Do I need to fill in any forms?
Receptionist: Yes, I'll email them to you today. Please send them back before Thursday.
Paul: Perfect, thank you.""",
    "Raw transcript (no punctuation)": """a: so did you talk to the landlord about the heating
b: yes he said someone will come on monday to fix it
a: monday is fine but we need to be home right
b: one of us has to be there between nine and twelve
a: i can work from home that day
b: great then i will tell him to come at ten
a: and ask him about the rent increase too
b: ok i will email him tonight""",
}
