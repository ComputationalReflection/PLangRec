from tkinter import Tk, END, E, W, ttk, StringVar, BooleanVar, NORMAL, DISABLED
from tkinter.ttk import Button, Combobox, Entry, LabelFrame, Treeview, Style

from configuration import PROGRAM_EXAMPLES, LANGUAGES
from model import predict


def sort_number(table, column, asc):
    rows = [(table.set(item, column), item) for item in table.get_children()]
    rows.sort(reverse=asc)

    # rearrange items in sorted positions
    for index, (values, item) in enumerate(rows):
        table.move(item, '', index)

    table.heading(column, command=lambda: sort_number(table, column, not asc))


def sort_string(table, column, asc):
    rows = [(table.set(item, column).lower(), item) for item in table.get_children()]
    rows.sort(reverse=asc)

    # rearrange items in sorted positions
    for index, (values, item) in enumerate(rows):
        table.move(item, '', index)

    table.heading(column, command=lambda: sort_string(table, column, not asc))


def main() -> None:
    """Creates the main window and runs the application"""
    # Main window
    mw = Tk()
    mw.title("Source Code Language Identification")
    mw.geometry("600x200")
    mw.resizable(False, False)
    mw.columnconfigure(0, weight=1)
    style = Style()
    style.theme_use("vista")

    # LabelFrame to create the query with source code line
    query_frame = LabelFrame(mw, text="Write a line of code of select one example from the combo")
    query_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky=W + E)
    query_frame.columnconfigure(1, weight=1)

    # Combobox inside LabelFrame to pick one predefined line of source code
    cmb = Combobox(query_frame, values=LANGUAGES, state="readonly")
    cmb.grid(row=1, column=0, padx=5, pady=(15, 0))
    cmb.current(0)  # Default pick

    # Entry inside LabelFrame to visualize the line picked or insert a new one
    line_var = StringVar()
    line = Entry(query_frame, font=('Courier', 10), textvariable=line_var)
    line.grid(row=2, column=0, columnspan=2, padx=5, pady=(5, 0), sticky=W + E)
    line.focus()
    line.insert(0, PROGRAM_EXAMPLES[cmb.get()])  # Visualize the line picked

    # Event on select new predefined line in Combobox
    def select_lang(_):
        line.delete(0, END)
        line.insert(0, PROGRAM_EXAMPLES[cmb.get()])

    cmb.bind("<<ComboboxSelected>>", select_lang)

    def get_prediction():
        for item in results_tree.get_children():
            results_tree.delete(item)
        predictions = sorted(predict(line.get()).items(), key=lambda x: x[1])
        for data in predictions:
            results_tree.insert(parent="", index=0, values=data)

    # When the text of the entry box is modified, the language is predicted
    line_var.trace_add("write", lambda a, b, c: get_prediction())

    # Checkbox to enable/disable the Predict button
    predict_as_typing_var = BooleanVar()
    predict_as_typing_checkbox = ttk.Checkbutton(query_frame, text="Predict language while typing",
                                                 variable=predict_as_typing_var)
    predict_as_typing_checkbox.grid(row=3, column=0, columnspan=2, padx=(10, 0), pady=(10, 0), sticky=W + E)

    # Button inside LabelFrame to send query to model
    predict_button = Button(query_frame, text="Predict", padding=[20, 5], command=get_prediction)
    predict_button.grid(row=4, columnspan=2, padx=5, pady=(35, 5), sticky=W + E)
    results_frame = LabelFrame(mw, text="Results")
    results_frame.grid(row=0, column=3, rowspan=4, padx=(0, 10), pady=10, sticky=W + E)

    # Function to enable/disable the Predict button based on checkbox state
    def update_button_state(*args):
        if predict_as_typing_var.get():
            predict_button["state"] = DISABLED
        else:
            predict_button["state"] = NORMAL

    # Trace changes in the checkbox state and call the update_button_state function
    predict_as_typing_var.trace_add("write", update_button_state)


    results_tree = Treeview(results_frame, columns=("lang", "probability"), show="headings", height=6)
    results_tree.grid(row=4, column=0, padx=5, pady=5, sticky=W + E)
    results_frame.columnconfigure(0, weight=1)
    results_tree.column("lang", width=100)
    results_tree.column("probability", width=100)
    results_tree.heading("lang", text="Language", command=lambda: sort_string(results_tree, "lang", False))
    results_tree.heading("probability", text="Prob. (%)", command=lambda: sort_number(results_tree, "accuracy", False))

    # Create a vertical scrollbar for results_tree
    vsb_results = ttk.Scrollbar(results_frame, orient="vertical", command=results_tree.yview)
    vsb_results.grid(row=4, column=1, padx=(0, 5), pady=5, sticky="ns")
    # Configure the Treeview to use the vertical scrollbar
    results_tree.configure(yscrollcommand=vsb_results.set)
    # Loads into memory the first prediction
    #kkk get_prediction()
    # run the main loop
    mw.mainloop()


if __name__ == "__main__":
    main()


