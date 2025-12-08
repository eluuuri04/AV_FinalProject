# variables.py

#for common categories
yes_no = {
    0: "No",
    1: "Yes",
}

gender = {
    1: "Male",
    2: "Female",
}

attendance = {
    1: "Daytime",
    0: "Evening",
}

marital_status = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated",
}

application_mode = {
    1: "1st phase – general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase – special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase – special contingent (Madeira Island)",
    17: "2nd phase – general contingent",
    18: "3rd phase – general contingent",
    26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)",
}

application_order = list(range(10))

courses = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)",
}

previous_qualification = {
    1: "Secondary education",
    2: "Higher education – bachelor’s degree",
    3: "Higher education – degree",
    4: "Higher education – master’s",
    5: "Higher education – doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling – not completed",
    10: "11th year of schooling – not completed",
    12: "Other – 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling – not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year)",
    38: "Basic education 2nd cycle (6th/7th/8th year)",
    39: "Technological specialization course",
    40: "Higher education – degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education – master (2nd cycle)",
}

nationalities = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldovan",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian",
}

mother_qual = {
    1: "Secondary Education – 12th Year",
    2: "Higher Education – Bachelor’s",
    3: "Higher Education – Degree",
    4: "Higher Education – Master’s",
    5: "Higher Education – Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year – Not Completed",
    10: "11th Year – Not Completed",
    11: "7th Year (Old)",
    12: "Other – 11th Year",
    14: "10th Year",
    18: "General Commerce Course",
    19: "Basic Education 3rd Cycle",
    22: "Technical-professional Course",
    26: "7th Year of Schooling",
    27: "2nd Cycle General High School",
    29: "9th Year – Not Completed",
    30: "8th Year",
    34: "Unknown",
    35: "Can’t Read or Write",
    36: "Can Read Without 4th Year",
    37: "Basic Education 1st Cycle",
    38: "Basic Education 2nd Cycle",
    39: "Technological Specialization Course",
    40: "Higher Education – Degree (1st Cycle)",
    41: "Specialized Higher Studies Course",
    42: "Professional Higher Technical Course",
    43: "Higher Education – Master (2nd Cycle)",
    44: "Higher Education – Doctorate (3rd Cycle)",
}

fathers_qualification = mother_qual | {
    13: "2nd Year Complementary High School",
    20: "Complementary High School Course",
    25: "Complementary High School – Not Concluded",
    31: "General Course of Administration and Commerce",
    33: "Supplementary Accounting and Administration",
}

mothers_occupation = {
    1: "Student",
    2: "Teacher",
    3: "Health professional",
    4: "Services",
    5: "Commerce",
    6: "Industry",
    7: "Retired",
    8: "Unemployed",
    9: "Other",
}

fathers_occupation = mothers_occupation #are the same

displaced_map = yes_no
special_needs_map = yes_no
scholarship_map = yes_no
international_map = yes_no
debtor_map = yes_no
fees_map = yes_no
