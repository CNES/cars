.. _get_stereo_products:

How to get full stereo products
===============================


Pléiades / SPOT 6-7 products (DINAMIS)
--------------------------------------

| `DINAMIS <https://dinamis.data-terra.org/catalogue/>`_ is a platform that acquires and distributes satellite Earth imagery (Pléaides and Spot 6-7) for french and foreign institutional users under `specific subscription conditions <https://dinamis.data-terra.org/en/eligible-users/>`_.


AIRBUS Pleiades NEO example files
---------------------------------
Example files are available here: https://intelligence.airbus.com/imagery/sample-imagery/pleiades-neo-tristereo-marseille/ (A form must be filled out to access the data).

.. _maxar_example_files:

Maxar WorldView example files
-----------------------------

| Example files are available on AWS S3 through the SpaceNet challenge here: `s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/WV3/PAN/`
| You need to install `aws-cli <https://github.com/aws/aws-cli>`_:

.. code-block:: console

   python -m venv venv-aws-cli # create a virtual environment
   source ./venv-aws-cli/bin/activate # activate it
   pip install --upgrade pip # upgrade pip
   pip install awscli


And download a stereo:

.. code-block:: console

   aws s3 cp --no-sign-request s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/WV3/PAN/18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.NTF .
   aws s3 cp --no-sign-request s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/WV3/PAN/18DEC15WV031000015DEC18140554-P1BS-500515572030_01_P001_________AAE_0AAAAABPABJ0.NTF  .


