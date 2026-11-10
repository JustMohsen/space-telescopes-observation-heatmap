"""Public project story, AI disclosure, and map-detail note."""

PERSONAL_STORY_PARAGRAPHS = (
    (
        "I love maps, graphs, and visualizations, like many other space and science enthusiasts. One day, the "
        "Hubble Ultra Deep Field image came across my feed, and I thought: I really want to see a heatmap of "
        "where we have been pointing our space telescopes. The story I remember about the first Deep Field "
        "image (which I was not sure how accurate it was) is that a proposal came up to point the Hubble "
        "Space Telescope at an empty patch of sky that we hadn't studied before to see what was out there, "
        "and there was resistance to wasting precious telescope time. Then that first Deep Field image came "
        "back and changed our estimate of the number of galaxies in the universe by a factor of 10—aside from "
        "the visual beauty and the sense of awe it brought."
    ),
    (
        "So I thought: I want to see where in the sky we have been exploring and pointing our telescopes. I "
        "searched a lot and couldn't find anything resembling what I was looking for, so I decided to take on "
        "the project of building a heatmap to show where we have been pointing our telescopes. I planned the "
        "project and thought about how to represent it, the coordinate systems, and the different targets. "
        "Solar System targets are moving targets that don't make sense on a fixed-sky map; we expect galactic "
        "targets to be denser along the Galactic Plane, whereas deep-space targets should be denser away from "
        "the Galactic Plane to avoid an obstructed view."
    ),
    (
        "A few months later, I finally started working on this. I built it in a few days, which was possible "
        "thanks to AI. It is not the most polished, but I am really happy with it and went through many "
        "iterations. I also learned that there is a bias toward the Northern Galactic Pole—partly because "
        "most historical ground-based sky surveys were built in Earth's northern hemisphere (which faces the "
        "North Galactic Pole and the Virgo Supercluster), and we try to point our space telescopes at the "
        "same targets for follow-up and comparison, and partly because the Southern Galactic Hemisphere "
        "contains the Large and Small Magellanic Clouds, which obstruct a chunk of the southern extragalactic "
        "sky. Pointing space telescopes at the same targets observed by ground-based telescopes within their "
        "observable regions is also the reason we see a dense belt of observations along the equator in the "
        "Equatorial reference frame."
    ),
)

AI_DISCLOSURE = (
    "I built this project with AI assistance: AI wrote most of the text and created the design, while I "
    "focused on the technical and scientific structure, iterative cleanup, functional testing, and "
    "checking the accuracy of the coordinate systems, telescope information, observation categories, and "
    "target classifications."
)

MAP_DETAIL_NOTE = (
    "To keep memory use low and updates responsive, interactive maps use a capped display grid. "
    "Fine and Ultra-Fine change the underlying HEALPix bins but may look similar. Calculations still "
    "use the selected binning; PNG exports use a denser display grid."
)
