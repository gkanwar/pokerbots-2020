import argparse
import dis
import os
import shutil

'''
Run as e.g.:
python build.py simple_bot
python build.py simple_bot simple_bot2

By default, this script checks in ./robots/ for the bot and in ./lib/
for the modules. This will check the imports of in
./robots/simple_bot/player.py against the list of modules in ./lib/.
Any imports that match will be copied over to ./robots/simple_bot/.

If --nozip isn't specified, ./robots/simple_bot/ will then be zipped
into ./robots/simple_bot.zip.

Run this if you've just pulled some changes to ./lib/ modules from git
and want to propagate these to your bot. Also run if you want to generate
an archive of the bot for upload.

Running without any files listed will check dependencies of all bots in
./robots/.
'''

parser = argparse.ArgumentParser()
parser.add_argument('bots', nargs='*', help='list of folders containing bots to be built')
parser.add_argument('--nozip', action='store_true', help='flag to skip zipping (i.e. only copy dependencies)')
parser.add_argument('--library', default='lib', help='location of modules to copy')
parser.add_argument('--dir', default='robots', help='location of robots')

args = parser.parse_args()

if len(args.bots) == 0:
    bots = [o for o in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, o))]
else:
    bots = args.bots

SCRIPT_NAME = 'player.py'

lib_scripts = [os.path.splitext(s)[0] for s in os.listdir(args.library)]

# copy required imports
for bot in bots:
    bot_path = os.path.join(args.dir, bot)
    with open(os.path.join(bot_path, SCRIPT_NAME)) as script:
        statements = script.read()

    imports = [__ for __ in dis.get_instructions(statements) if 'IMPORT' in __.opname]

    for im in imports:
        if im.argval in lib_scripts:
            shutil.copyfile(os.path.join(args.library, im.argval+'.py'),
                                os.path.join(bot_path, im.argval+'.py'))

    # zip archive
    if not args.nozip:
        shutil.make_archive(bot_path, 'zip', bot_path)