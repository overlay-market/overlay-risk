import nox


@nox.session(python=['3.9'])
def tests(session):
    session.install('poetry')
    session.run('poetry', 'install')
    session.run('coverage', 'run', '-m', 'pytest')
    session.run('coverage', 'report')


@nox.session
def lint(session):
    session.install('poetry')
    session.run('poetry', 'install')
    session.run('flake8', 'scripts', 'tests')
