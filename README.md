# Autonomous Fact Checking Tool

> This is a project for my Master's in Computer Science with the [International University of Applied Sciences](https://www.iu.org/)

## Abstract

> Light description here.

Building an AI model to help users determine if claims are true or false.

## Licensing

I am not a super familiar with all the different types of licenses.
However, I found [this link](https://choosealicense.com/licenses/) provided when creating the repository that explains the ones available on Github.
Not sure if it is new, I've never noticed it since creating the previous ~40 repositories, or I did notice it and forgot the link.
Either way, it is a gem; clearly explains and summarizes main points.

I went with the GNU GPLv3 license because it makes this work available to eveyone, it's free and open.
Users can freely run, study, share, and modify the software as they like.
Companies can even use the software for commercial use!
However, it has 'anti-proprietary' protection - companies cannot hijack the code and make it proprietary.
Any _distributed_ software produced by this software must also be distributed with the **same** license.

That last statement might be off-putting for some companies or small businesses.
If you try to integrate this software into your system and distribute the system, you'd have to release the system under this copyleft license.
Since this project is focused on finding the truth, and rooting out the lies, I believe transparency is crucial.
What I do recommend if the project ever takes off,
and for companies concerned about the license,
is to incorporating this tool as a microservice.
The GLP requirements are triggered by distribution - using the software as a standalone service won't require opening your entire system.
Keep the microservice open for transparency and use it as a standalone service.

If there are any issues with the license, please feel free to reach out and I will consider changes.

## Python Libraries

For context, I downloaded Python 3.11.13 with the hopes everything will remain compatable with my project.

Firstly installing [JupyterLab](https://jupyter.org/install).

```
source ./.venv/bin/activate
jupyter lab
```

Jupyter Notebooks have no autocomplete,
[StackOverflow](https://stackoverflow.com/questions/45390326/how-to-get-autocomplete-in-jupyter-notebook-without-using-tab)
has some ideas though.
But I think the industry is leaning away from Tabnine and more towards the LSP:

```bash
pip install jupyterlab-lsp python-lsp-server[all]
# OR -- but not me
pip install jedi ipywidgets
# restart jupyter
# ... MORE ISSUES
# Stop Jupyter first (Ctrl+C)
# Then install/enable the extension
jupyter labextension install @jupyterlab/lsp
# Or if using newer versions:
pip install jupyterlab-lsp[extension]
# Then this?
pip install ipywidgets # avoids another error - ugh
# Restart Jupyter
jupyter lab
# OR
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

```
pip install transformers torch datasets scikit-learn pandas numpy
```

I've installed Cuda previously on WSL and do not remember the exact steps.
However, checking with the following hopefully returns TRUE:

```python
import torch
torch.cuda.is_available() # TRUE/FALSE
```

Trying to work with transformers and torch I get an error to do:

```bash
pip install transformers[torch]
```

## Chrome Extension

Using the model is the most important part.
I think an easy to use Chrome extension is the best path currently to integrate into social media platforms.

Current installation process:

1. Open Chrome.
2. go to `chrome://extensions/`.
3. Enable "Developer mode" (should be toggle in top right).
4. Click "Load unpacked".
5. Select the "chrome_ext" folder.

If all goes well, you should see the chrome extension icon in the toolbar.

## Diagrams

```mermaid
sequenceDiagram
        actor user as User
        participant be as Browser Ext.
        participant back as Back End
        participant ce as Claim Extractor
        participant rag as RAG
        participant judge as Judge AI

        user->>be: send post
        be->>back: send post
        activate back
        back->>ce: post
        activate ce
        ce->>ce: process post
        ce->>back: claims
        deactivate ce
        loop over claims
            back->>rag: claims
            activate rag
            rag->>back: evidence
            deactivate rag
        end
        loop over claims
            back->>judge: claims + evidence
            activate judge
            judge->>back: verdict
            deactivate judge
        end
        back->>be: json response
        deactivate back
        be->>user: response
```
