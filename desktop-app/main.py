from tkinter import Tk, END, E, W, ttk, StringVar, BooleanVar, NORMAL, DISABLED, Text, Event
from tkinter.ttk import Button, Combobox, LabelFrame, Treeview, Style

from configuration import PROGRAM_EXAMPLES, LANGUAGES
from model import predict


def sort_number(table: Treeview, column_name: str, ascending: bool) -> None:
    """Sorts the table tree view by prediction probability"""
    rows = [(table.set(item, column_name), item) for item in table.get_children()]
    rows.sort(reverse=ascending)

    # rearrange items in sorted positions
    for index, (values, item) in enumerate(rows):
        table.move(item, '', index)

    table.heading(column_name, command=lambda: sort_number(table, column_name, not ascending))


def sort_string(table: Treeview, column_name: str, asc: bool) -> None:
    """Sorts the table tree view by language name"""
    rows = [(table.set(item, column_name).lower(), item) for item in table.get_children()]
    rows.sort(reverse=asc)
    # rearrange items in sorted positions
    for index, (values, item) in enumerate(rows):
        table.move(item, '', index)
    table.heading(column_name, command=lambda: sort_string(table, column_name, not asc))


def main() -> None:
    """Creates the main window and runs the application"""
    # Main window
    mw = Tk()
    mw.title("Source Code Language Identification")
    mw.geometry("700x240")
    mw.resizable(False, False)
    mw.columnconfigure(0, weight=1)
    style = Style()
    style.theme_use("vista")

    # LabelFrame to create the query with source code text
    query_frame = LabelFrame(mw, text="Write a text of code of select one example from the combo")
    query_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky=W + E)
    query_frame.columnconfigure(1, weight=1)

    # Combobox inside LabelFrame to pick one predefined text of source code
    cmb = Combobox(query_frame, values=LANGUAGES, state="readonly")
    cmb.grid(row=1, column=0, padx=5, pady=(15, 0))
    cmb.current(0)  # Default pick

    # Entry inside LabelFrame to visualize the text picked or insert a new one
    text_var = StringVar()
    text = Text(query_frame, font=('Courier', 10), wrap="word", height=5, width=40)
    text.grid(row=2, column=0, columnspan=2, padx=5, pady=(5, 0), sticky=W + E)

    # Function to update StringVar when the text in the Text widget changes
    def update_text_var(*_) -> None:
        text_var.set(text.get("1.0", "end-1c"))

    # Trace changes in the Text widget and call the update_text_var function
    text.bind("<KeyRelease>", update_text_var)

    # Vertical scrollbar for the Text widget
    vsb_line = ttk.Scrollbar(query_frame, orient="vertical", command=text.yview)
    vsb_line.grid(row=2, column=2, pady=(5, 0), sticky="ns")
    text.configure(yscrollcommand=vsb_line.set)

    text.focus()
    text.insert("1.0", PROGRAM_EXAMPLES[cmb.get()])  # Visualize the text picked

    # Event on select new predefined text in Combobox
    def select_lang(_: Event) -> None:
        text.delete("1.0", END)
        text.insert(END, PROGRAM_EXAMPLES[cmb.get()])

    cmb.bind("<<ComboboxSelected>>", select_lang)

    def get_prediction() -> None:
        for item in results_tree.get_children():
            results_tree.delete(item)
        predictions = sorted(predict(text.get("1.0", END)).items(), key=lambda x: x[1])
        for data in predictions:
            results_tree.insert(parent="", index=0, values=data)

    # Checkbox to enable/disable the Predict button
    predict_while_typing_var = BooleanVar()
    predict_as_typing_checkbox = ttk.Checkbutton(query_frame, text="Predict language while typing",
                                                 variable=predict_while_typing_var)
    predict_as_typing_checkbox.grid(row=3, column=0, columnspan=2, padx=(10, 0), pady=(10, 0), sticky=W + E)

    # Button inside LabelFrame to send query to model
    predict_button = Button(query_frame, text="Predict", padding=[20, 5], command=get_prediction)
    predict_button.grid(row=4, columnspan=2, padx=5, pady=(5, 5), sticky=W + E)

    results_frame = LabelFrame(mw, text="Results")
    results_frame.grid(row=0, column=3, rowspan=5, padx=(0, 10), pady=10, sticky=W + E + "ns")  # Set rowspan and sticky
    results_frame.columnconfigure(0, weight=1)

    # Function to enable/disable the Predict button based on checkbox state
    def update_button_state(*_) -> None:
        if predict_while_typing_var.get():
            predict_button["state"] = DISABLED
        else:
            predict_button["state"] = NORMAL

    # Trace changes in the checkbox state and call the update_button_state function
    predict_while_typing_var.trace_add("write", update_button_state)

    results_tree = Treeview(results_frame, columns=("lang", "probability"), show="headings", height=8)
    results_tree.grid(row=4, column=0, rowspan=5, padx=5, pady=5, sticky=W + E + "ns")
    results_frame.columnconfigure(0, weight=1)
    results_tree.column("lang", width=100)
    results_tree.column("probability", width=100)
    results_tree.heading("lang", text="Language", command=lambda: sort_string(results_tree, "lang", False))
    results_tree.heading("probability", text="Prob. (%)",
                         command=lambda: sort_number(results_tree, "probability", False))

    # Create a vertical scrollbar for results_tree
    vsb_results = ttk.Scrollbar(results_frame, orient="vertical", command=results_tree.yview)
    vsb_results.grid(row=4, column=1, padx=(0, 5), pady=5, rowspan=8, sticky="ns")
    # Configure the Treeview to use the vertical scrollbar
    results_tree.configure(yscrollcommand=vsb_results.set)

    # When the text of the entry box is modified, the language is predicted
    # if the "predict while typing" checkbox is enabled
    def text_changed() -> None:
        if predict_while_typing_var.get():
            get_prediction()
    # Trace changes in the StringVar and call on_text_change function
    text_var.trace_add("write", lambda a, b, c: text_changed())

    # Loads into memory the first prediction
    get_prediction()
    # run the main loop
    mw.mainloop()


if __name__ == "__main__":
    main()
