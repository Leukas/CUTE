# prompts.py
PROMPTS = {
    "spell": 'Spell out the word, putting spaces between each letter, based on the following examples:\n\
            \n\
            1. Spell out the word "alphabet". Answer: "a l p h a b e t"\n\
            2. Spell out the word "hello". Answer: "h e l l o"\n\
            3. Spell out the word "zebra". Answer: "z e b r a"\n\
            4. Spell out the word "tongue". Answer: "t o n g u e"\n\
            \n\
            Question: Spell out the word "{}".',
    "spell_inverse": 'Write the word that is spelled out, without any spaces, based on the following examples:\n\
            \n\
            1. Write the word "a l p h a b e t".  Answer: "alphabet"\n\
            2. Write the word "h e l l o". Answer: "hello"\n\
            3. Write the word "z e b r a". Answer: "zebra"\n\
            4. Write the word "t o n g u e". Answer: "tongue"\n\
            \n\
            Question: Write the word "{}".',
    "contains_char": 'Answer whether the specified letter is in the given word, based on the following examples:\n\
            \n\
            1. Is there a "a" in "alphabet"? Answer: "Yes"\n\
            2. Is there a "z" in "alphabet"? Answer: "No"\n\
            3. Is there a "u" in "hello"? Answer: "No"\n\
            4. Is there a "o" in "hello"? Answer: "Yes"\n\
            \n\
            Question: Is there a "{}" in "{}"?',
    "contains_word": 'Answer whether the specified word is in the given sentence (case insensitive), based on the following examples:\n\
            \n\
            1. Is there a "the" in "The cow goes moo."? Answer: "Yes"\n\
            2. Is there a "goat" in "The cow goes moo."? Answer: "No"\n\
            3. Is there a "glad" in "I am very happy."? Answer: "No"\n\
            4. Is there a "happy" in "I am very happy."? Answer: "Yes"\n\
            \n\
            Question: Is there a "{}" in "{}"?',
    "orth": 'Select the word that is closer in Levenshtein distance to the given word based on the following examples:\n\
            \n\
            1. Closer in Levenshtein distance to "bold": "cold", "brave". Answer: "cold"\n\
            2. Closer in Levenshtein distance to "computer": "completed", "laptop". Answer: "completed"\n\
            3. Closer in Levenshtein distance to "happy": "glad, "apply". Answer: "apply"\n\
            4. Closer in Levenshtein distance to "camp": "ramp", "tent". Answer: "ramp"\n\
            \n\
            Question: Closer in Levenshtein distance to "{}": "{}", "{}".',
    "sem": 'Select the word that is more semantically related to the given word based on the following examples:\n\
            \n\
            1. More semantically related to "bold": "cold", "brave". Answer: "brave"\n\
            2. More semantically related to "computer": "completed", "laptop". Answer: "laptop"\n\
            3. More semantically related to "happy": "glad", "apply". Answer: "glad"\n\
            4. More semantically related to "camp": "ramp", "tent". Answer: "tent"\n\
            \n\
            Question: More semantically related to "{}": {}, {}.',
    "ins_char": 'Add the specified letter after every instance of the second specified letter in a given word, based on the following examples:\n\
            \n\
            1. Add an "e" after every "a" in "alphabet". Answer: "aelphaebet"\n\
            2. Add an "l" after every "l" in "hello". Answer: "hellllo"\n\
            3. Add an "t" after every "z" in "zebra". Answer: "ztebra"\n\
            4. Add an "f" after every "u" in "tongue". Answer: "tongufe"\n\
            \n\
            Question: Add an "{}" after every "{}" in "{}".',
    "ins_word": 'Add the specified word after every instance of the second specified word in a given sentence, based on the following examples:\n\
            \n\
            1. Add "bad" after every "beautiful" in "it is a beautiful day". Answer: "it is a beautiful bad day"\n\
            2. Add "hello" after every "day" in "it is a beautiful day". Answer: "it is a beautiful day hello"\n\
            3. Add "not" after every "i" in "i think i can do it". Answer: "i not think i not can do it"\n\
            4. Add "can" after every "can" in "i think i can do it". Answer: "i think i can can do it"\n\
            \n\
            Question: Add "{}" after every "{}" in "{}".',
    "del_char": 'Delete every instance of a specified letter in a given word, based on the following examples:\n\
            \n\
            1. Delete every instance of "a" in "alphabet". Answer: "lphbet"\n\
            2. Delete every instance of "l" in "hello". Answer: "heo"\n\
            3. Delete every instance of "z" in "zebra". Answer: "ebra"\n\
            4. Delete every instance of "u" in "tongue". Answer: "tonge"\n\
            \n\
            Question: Delete every instance of "{}" in "{}".',
    "del_word": 'Delete every instance of a specified word in a given sentence, based on the following examples:\n\
            \n\
            1. Delete every instance of "a" in "it is a beautiful day". Answer: "it is beautiful day"\n\
            2. Delete every instance of "beautiful" in "it is a beautiful day". Answer: "it is a day"\n\
            3. Delete every instance of "i" in "i think i can do it". Answer: "think can do it"\n\
            4. Delete every instance of "can" in "i think i can do it". Answer: "i think i do it"\n\
            \n\
            Question: Delete every instance of "{}" in "{}".',
    "swap_char": 'Swap the positions of two specified letters in a given word, based on the following examples:\n\
            \n\
            1. Swap "l" and "b" in "alphabet". Answer: "abphalet"\n\
            2. Swap "h" and "e" in "hello". Answer: "ehllo"\n\
            3. Swap "z" and "a" in "zebra". Answer: "aebrz"\n\
            4. Swap "u" and "e" in "tongue". Answer: "tongeu"\n\
            \n\
            Question: Swap "{}" and "{}" in "{}".',
    "swap_word": 'Swap the positions of two specified words in a given sentence, based on the following examples:\n\
            \n\
            1. Swap "it" and "a" in "it is a beautiful day". Answer: "a is it beautiful day"\n\
            2. Swap "it" and "beautiful" in "it is a beautiful day". Answer: "beautiful is a it day"\n\
            3. Swap "think" and "do" in "i think i can do it". Answer: "i do i can think it"\n\
            4. Swap "can" and "do" in "i think i can do it". Answer: "i think i do can it"\n\
            \n\
            Question: Swap "{}" and "{}" in "{}".',
    "sub_char": 'Substitute the first specified letter with the second specified letter in a given word, based on the following examples:\n\
            \n\
            1. Substitute "a" with "b" in "alphabet". Answer: "blphbbet"\n\
            2. Substitute "h" with "e" in "hello". Answer: "eello"\n\
            3. Substitute "z" with "a" in "zebra". Answer: "aebra"\n\
            4. Substitute "u" with "e" in "tongue". Answer: "tongee"\n\
            \n\
            Question: Substitute "{}" with "{}" in "{}".',
    "sub_word": 'Substitute the first specified word with the second specified word in a given sentence, based on the following examples:\n\
            \n\
            1. Substitute "it" with "a" in "it is a beautiful day". Answer: "a is a beautiful day"\n\
            2. Substitute "it" with "beautiful" in "it is a beautiful day". Answer: "beautiful is a beautiful day"\n\
            3. Substitute "i" with "do" in "i think i can do it". Answer: "do think do can do it"\n\
            4. Substitute "think" with "can" in "i think i can do it". Answer: "i can i can do it"\n\
            \n\
            Question: Substitute "{}" with "{}" in "{}".',
}