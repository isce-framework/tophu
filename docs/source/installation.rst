Installation
############

Install from source (conda + pip)
=================================

#. Use git to clone the tophu repository:

    .. code-block:: console

        $ git clone https://github.com/opera-adt/tophu
        $ cd tophu

#. Install the dependencies using conda:

    .. code-block:: console

        $ conda install --file requirements.txt --channel conda-forge

    .. Tip::

        Use conda `environments
        <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
        to isolate packages from different projects

#. Install from source using pip:

    .. code-block:: console

        $ pip install --no-deps .

#. (Optional) Run the test suite:

    .. code-block:: console

        $ conda install --file test/requirements.txt --channel conda-forge
        $ pytest
