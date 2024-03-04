from tkinter import Tk, END, E, W, ttk, StringVar
from tkinter.ttk import Button, Combobox, Entry, LabelFrame, Treeview, Style

from configuration import LANGUAGES, PROGRAM_EXAMPLES
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


def main():
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
    line = Entry(query_frame, font=('Courier', 10))
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

    # Button inside LabelFrame to send query to model
    Button(query_frame, text="Predict", padding=[20, 5], command=get_prediction).grid(row=3, columnspan=2, padx=5, pady=(35, 5), sticky=W + E)
    results_frame = LabelFrame(mw, text="Results")
    results_frame.grid(row=0, column=3, rowspan=4, padx=(0, 10), pady=10, sticky=W + E)

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
    get_prediction()
    # run the main loop
    mw.mainloop()


if __name__ == "__main__":
    main()


