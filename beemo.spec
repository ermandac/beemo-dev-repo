# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['Main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('BeemoApp/*', 'BeemoApp'),
        ('Model/*', 'Model'),
        ('Icon/*', 'Icon'),
        ('SavedIMG/*', 'SavedIMG'),
        ('BeemoApp/GsheetAPI/*', 'BeemoApp/GsheetAPI')
    ],
    hiddenimports=[
        'tkinter', 'PIL', 'sounddevice', 'scipy', 'matplotlib', 'numpy', 'pyaudio', 'tensorflow', 
        'google-api-python-client', 'google-auth-oauthlib', 'google-auth', 'google-auth-httplib2', 
        'discord', 'blynklib', 'absl-py', 'requests', 'pandas', 'sklearn', 'schedule'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['keras.src.backend.torch', 'tensorflow-plugins'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='beemo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='beemo',
)