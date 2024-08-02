from fasthtml.common import *

def Toolbar():
    return Div(
        Select(
            Option("WGSL", value="wgsl"),
            Option("C++", value="c++"),
            id="language",
            cls="mr-2 p-2 border rounded"
        ),
        cls="bg-gray-200 p-4 shadow-md flex items-center w-full"
    )
