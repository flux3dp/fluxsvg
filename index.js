const core = require('@actions/core');
const exec = require('@actions/exec');
const os = require('os');

const main = async () => {
    try {
        const options = {
            cwd: __dirname,
        };
        console.log(os.platform());
        if (os.platform() === 'win32') {
            await exec.exec('python', [`./setup.py`, 'install'], options);
        } else {
            await exec.exec('python3',  [`./setup.py`, 'install'], options);
        }
    } catch (error) {
        core.setFailed(error.message);
    }
}
main();