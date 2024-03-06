from tkinter import Tk, END, E, W
from tkinter.ttk import Button, Combobox, Entry, LabelFrame, Treeview, Style

from model import predict

LANGUAGES = ["Assembly", "C", "C++", "C#", "CSS", "Go", "HTML", "Java", "JavaScript", "Kotlin",
                   "Matlab", "Perl", "PHP", "Python", "R", "Ruby", "Scala", "SQL", "Swift", "TypeScript",
                   "Unix Shell"]

PROGRAM_EXAMPLES = {
    "Assembly": "ADD CX, [BX+SI*2+10]",
    "C": "int *numbers = malloc(sizeof(int));",
    "C++": "std::vector<int> integers = {1, 2, 3};",
    "C#": "Console.WriteLine(numbers);",
    "CSS": "-webkit-transition: all 0.3s ease;",
    "Go": "func (Person p) Name() string {",
    "HTML": '<input id="name" style="color: blue;">',
    "Java": "super(name, lastName, age);",
    "JavaScript": "http.createServer(function (req, res) {",
    "Kotlin": "val odds = integers.filter { it % 2 != 0 }",
    "Matlab": "fprintf('%4u is greater than 5 \\r', num1)",
    "Perl": "my $message = Email::MIME->create(header_str => [ From=> 'you@example.com']);",
    "PHP": "<?php echo '<p>Hello World</p>'; ?>",
    "Python": "codes = [ord(char) for char in line]",
    "R": "plot(c(1, 8), c(3, 10))",
    "Ruby": "def sum_eq_n?(arr, n)",
    "Scala": "val odds = integers.filter(_ % 2 != 0)",
    "SQL": "SELECT * FROM Customers;",
    "Swift": "var capitalCity = [\"Nepal\": \"Kathmandu\", \"Italy\": \"Rome\", \"England\": \"London\"]",
    "TypeScript": "let ourTuple: [number, boolean, string];",
    "Unix Shell": "echo \"Wow, you look younger than $age years old\"",
}