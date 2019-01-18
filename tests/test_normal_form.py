from gamepy import normal_form

actions = [
    "stays_silent",
    "betrays",
]

players = {
    "Alice": actions,
    "Bob": actions,
}

def test_c():
    game = normal_form.c (players)
