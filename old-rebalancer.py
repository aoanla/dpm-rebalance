#!/usr/bin/python

import sys
import os
from optparse import OptionParser
import string
import time
import math
import MySQLdb
import gridpp_dpm
import random
import errno
import paramiko
import re
import socket
import collections

class Move(object):
  """Representation of a Move.  That is, a file to be moved from it's curent location, and a note of where it should be moved to."""
  def __init__(self, file, dest):
    self.file = file
    self.dest = dest

  def __eq__(self, other):
    if None == other:
      return False
    return self.file == other.file

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    return "Move: " + str(self.file.fileid) + " from " + str(self.file.host) + str(self.file.filesystem) + " to " + self.dest.desc() + ")"


# Get a space token to try to optimise
#	Lookup the UUID, or do it as a join?
#	Do it as a lookup - we need other information from the space token, and thus it's less bandiwdth this way round.

class SpaceToken(object):

  def __init__(self, name, uuid, pool):
    self.name = name
    self.uuid = uuid
    self.pool = pool

  @staticmethod
  def get(cursor, dpmdbName, name):
    """Factory method for retrieveing SpaceTokens from the db.  Only retrieves the first spacetoken that matches the name."""
    # XXX: Maybe also check that there is free space in the token?
    try:
      cursor.execute('''
        SELECT s_token, poolname
        FROM  %(dpm)s.dpm_space_reserv spacetoken
        WHERE spacetoken.u_token = '%(spacetokenName)s'
        ''' % {"dpm" : dpmdbName, "spacetokenName" : name})
      row = cursor.fetchone()
      return SpaceToken(name, row[0], row[1])
    except MySQLdb.Error, e:
      print "Error %d: %s" % (e.args[0], e.args[1])
      sys.exit(1)

  def listWorstDirs(self, cursor, validFilesystems, target):
    """Gives the 'target' worst balanced dirs, and the fuzzfactor of the least worst of them. If there aren't enough, then it will return them all, which will be less than n"""
    dirs = list()
    fuzz = None
    for (dir, f) in self.listUnevenDirs(cursor, validFilesystems):
      dirs.append(dir)
      fuzz = f
      if len(dirs) == target:
        # Need to cast to a float, as it's a Decimal that comes back from MySql
        return (dirs, float(f))
    return (dirs, float(f))

  def listUnevenDirs(self, cursor, validFilesystems, threshold=40):
    """Gives a list of tuples that list a dir Id, and the fuzzfactor that would cause it to start to improve that dir."""
    try:
      cursor.execute('''
        SELECT b.parent_fileid, b.files, b.peak, files / %(vfs)s AS target, b.ratio, (peak - CEILING(files / %(vfs)s)) / CEILING(files / %(vfs)s) AS badness
        FROM (
          SELECT a.parent_fileid, sum(a.num) AS files, max(a.num) AS peak, max(a.num) / sum(a.num) AS ratio
          FROM (
            SELECT  m.parent_fileid, r.host, r.fs, count(m.fileid) AS num
            FROM Cns_file_replica r
              JOIN Cns_file_metadata m USING (fileid)
            WHERE r.setname = '%(uuid)s'
            GROUP BY parent_fileid, host, fs
            ) a
          GROUP BY parent_fileid
        ) b
        WHERE b.files > %(threshold)s
        ORDER BY badness DESC''' % {"uuid" : self.uuid, "vfs" : validFilesystems, "threshold" : threshold})
      ret = list()
      for row in cursor.fetchall():
        ret.append( (row[0], row[5]) )
      return ret
    except MySQLdb.Error, e:
      print "Error %d: %s" % (e.args[0], e.args[1])
      sys.exit(1)
    

# Get all files for that spaceToken
#	With size and parent namespace
# 	Looking at just under 600 000 files, so handle carefully

class SRMFile(object):

  def __init__(self, fileid, parent_fileid, size, ctime, name, poolname, host, filesystem):
    self.fileid = fileid
    self.parent_fileid = parent_fileid
    self.size = size
    self.ctime = ctime
    self.name = name
    self.poolname = poolname
    self.host = host
    self.filesystem = filesystem

  @staticmethod
  def getAllFor(cursor, dpmdbName, spacetoken):
    """ Factory method to get all the files for a single SpaceToken.  Gives a List of files.  Note that this can return a _lot_ of files"""
    try:
      cursor.execute('''
        SELECT m.fileid, m.parent_fileid, m.filesize, m.ctime, m.name
          , r.poolname, r.host, r.fs
        FROM Cns_file_replica r 
          JOIN Cns_file_metadata m USING (fileid)
        WHERE r.setname = '%(spacetokenUUID)s'
        ''' % {"spacetokenUUID" : spacetoken.uuid})
      ret = list()
      for row in cursor.fetchall():
        ret.append(SRMFile(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
      return ret
    except MySQLdb.Error, e:
      print "Error %d: %s" % (e.args[0], e.args[1])
      sys.exit(1)

  @staticmethod
  def getAllByDir(cursor, dpmdbName, spacetoken):
    """ Factory method to get all the files for a single SpaceToken.  Gives a dict of dirname to list of files.  Note that this can return a _lot_ of files"""
    try:
      cursor.execute('''
        SELECT m.fileid, m.parent_fileid, m.filesize, m.ctime, m.name
          , r.poolname, r.host, r.fs
        FROM Cns_file_replica r
          JOIN Cns_file_metadata m USING (fileid)
        WHERE r.setname = '%(spacetokenUUID)s'
        ORDER BY m.parent_fileid
        ''' % {"spacetokenUUID" : spacetoken.uuid})
      ret = dict()
      lastParentid = None
      buildList = None
      # Loop over multipl sections of data, and place directly into the map. 
      # We know that when the parent_fileid changes, then we know the previous one is done
      while True:
        rows = cursor.fetchmany(20)
	if 0 == len(rows):
          return ret
        for row in rows:
          if row[1] != lastParentid:
	    # Found a new (or first) dir
            if None != buildList:
              ret[lastParentid] = buildList
            buildList = list()
            lastParentid = row[1]
          buildList.append(SRMFile(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
      return ret
    except MySQLdb.Error, e:
      print "Error %d: %s" % (e.args[0], e.args[1])
      sys.exit(1)

  @staticmethod
  def listDirs(cursor, dpmsbName, spacetoken, threshold):
    """Factory method to get all directories within a single space token, sorted from most files to least.  Does not walk the hierachy, so if it contains sub dirs, then they will be ignored.
    Note that this returns a list of dirID's, not SRMFiles."""
    try:
      cursor.execute('''
        SELECT parent_fileid, numEntries FROM (
          SELECT m.parent_fileid, count(*) as numEntries
          FROM Cns_file_replica r 
            JOIN Cns_file_metadata m USING (fileid)
          WHERE r.setname = '%(spacetokenUUID)s'
	  GROUP BY m.parent_fileid
	  ORDER BY numEntries DESC) a
        WHERE numEntries > %(threshold)d
        ''' % {"spacetokenUUID" : spacetoken.uuid, "threshold" : threshold})
      ret = list()
      for row in cursor.fetchall():
        ret.append(row[0]);
      return ret
    except MySQLdb.Error, e:
      print "Error %d: %s" % (e.args[0], e.args[1])
      sys.exit(1)

  @staticmethod
  def getForDir(cursor, dpmdbName, dirID):
    """Factory method to get all the files for a single dirID.  Note that this does not walk the hierachy, so it will ignore sub dirs."""
    try:
      cursor.execute('''
        SELECT m.fileid, m.parent_fileid, m.filesize, m.ctime, m.name
          , r.poolname, r.host, r.fs
        FROM Cns_file_replica r
          JOIN Cns_file_metadata m USING (fileid)
        WHERE m.parent_fileid = '%(dirID)s'
        ''' % {"dirID" : dirID})
      ret = list()
      for row in cursor.fetchall():
        ret.append(SRMFile(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
      return ret
    except MySQLdb.Error, e:
      print "Error %d: %s" % (e.args[0], e.args[1])
      sys.exit(1)
 
# Get a list of places where we can put files (pools?)
# 	And space available on those
class FileSystem(object):

  def __init__(self, poolname, server, name, status, weight):
    self.poolname = poolname
    self.server = server
    self.name = name
    self.status = status
    self.weight = weight
    self.files = list()
    # Filled in by annotateFreeSpace()
    self.size = None
    self.avail = None

  def desc(self):
    """Short description of this location - i.e. a minimal human readable description of what this is"""
    return str(self.server) + str(self.name)

  def fileCount(self):
    """Support for using these to track dirs at a time"""
    return len(self.files)

  @staticmethod
  def getForPool(cursor, dpmdbName, spacetoken):
    """ Factory method to get all filesystems in a pool"""
    try:
      cursor.execute('''
        SELECT poolname, server, fs, status, weight
        FROM %(dpmdbName)s.dpm_fs 
        WHERE poolname = '%(pool)s'
        ''' % {"dpmdbName": dpmdbName, "pool" : spacetoken.pool })
      ret = list()
      for row in cursor.fetchall():
        ret.append(FileSystem(row[0], row[1], row[2], row[3], row[4]))
      return ret
    except MySQLdb.Error, e:
      print "Error %d: %s" % (e.args[0], e.args[1])
      sys.exit(1)

  def __repr__(self):
    return "FileSystem(poolname=" + self.poolname + ", server=" + self.server + ", name=" + self.name + ", status=" + str(self.status) + ", weight=" + str(self.weight) + ", with " + str(len(self.files)) + " files and " + str(self.avail) + " 1k blocks avail)"

def main():
  parser = OptionParser( usage = 'usage: %prog')
  parser.add_option('--spacetoken', dest='spacetoken', default='ATLASDATADISK', help='Spacetoken name to rebalance')
  parser.add_option('--threshold', dest='threshold', type='int', default=20, help='Minimum files in a dir before trying to rebalance')
  parser.add_option('--fuzz-factor', dest='fuzzfactor', type='float', default=2.0, help='Controls the agressiveness of the rebalanceing - How many times the the truely even number of files a node must have, before action is taken.  For example, a factor of 2 (the defualt) will take no action unless a filesystem has more than twice the mean level - so for a dir with 100 files, across 10 filesystems, no action would be taken unless a filesystem had over 20 files. The larger this value, the less work will be done, so in general, run with this large to begin with (say, 0.1), and gradually reduce it once things balance out better. Verbose settings 2+ will print the actual factor for each dir')
  parser.add_option('--auto-fuzz', dest='autofuzz', type='int', default=0, help='Instead of using the manually specified fuzzfactor, determine the correct factor to cause rebalancing to occur over the worst n dirs, for the given n. ')
  parser.add_option('--free-servers', dest='freeServers', type='int', default=2, help='Number of servers (not filesystems) to keep clear for each directory.  This ensures that loss of this number or fewer servers will not result in the loss of all datasets.')
  parser.add_option('--fs-per-server', dest='fsPerServer', type='int', default=4, help='Average number of filesystems per server - used for planning only.')
  parser.add_option('--verbose', dest='verbose', default=0, action='count',  help='Print out the analysis and decision details.  Once gives the overview, twice gives analysis of courrent, and thrice gives internal metrics')
  parser.add_option('--for-dir', dest='targetDir', type='int', default=None, help='Skip the detection, and balance only the files with the given parent_fileid.  Note that you have to have the right --spacetoken specified as well')
  parser.add_option('--dry-run', dest='dryrun', action='store_true', help='Do the analysis, but stop short of actually moving files.')

  # Transfer control settings
  parser.add_option('--max-reads', dest='maxreads', type='int', default=1, help='Maximum number of simultanious reads from each server')
  parser.add_option('--max-writes', dest='maxwrites', type='int', default=1, help='Maximum number of simulanious writes to each server')
  parser.add_option('--spare-transfers', dest='sparetransfers', type='int', default=1, help='Number of possbile transfers destinations to leave free.  This set to non-zero will improve the distribution of moves (i.e. stop deep searching for repeasting the same move over and over.  We use only the destinations, becuse in general there are far more destinations than sources.')

  (options, args) = parser.parse_args()
   
  (db,cursor,dpmdbName) = gridpp_dpm.MySQLConnect(True)

  spaceToken = SpaceToken.get(cursor, dpmdbName, options.spacetoken)
  fsf = FileSystem.getForPool(cursor, dpmdbName, spaceToken)

  targetFuzz = options.fuzzfactor

  # XXX: Assumptive hack, but only used to plan, not execute, so not too bad.
  validfs = len(fsf) - (options.fsPerServer * options.freeServers)
  if options.autofuzz > 0:
    (dirs, targetFuzz) = spaceToken.listWorstDirs(cursor, validfs, options.autofuzz)
    if options.verbose > 1:
      print "Target fuzz of " + str(targetFuzz) + ", going to rebalance " + str(dirs)
  elif options.targetDir != None:
    dirs = [options.targetDir]
  else:
    # Check everything
    dirs = SRMFile.listDirs(cursor, dpmdbName, spaceToken, options.threshold)

  if options.verbose > 1:
    print "Collecting free space information"
  fsf = annotateFreeSpace(fsf);

  consolidatedActions = dict() # dict of server name to list of actions
  for dir in dirs:
    print dir
    files = SRMFile.getForDir(cursor, dpmdbName, dir)
    actions = calculateMoveList(files, fsf, minfree=options.freeServers, fuzzfactor=targetFuzz, verbose=options.verbose)
    if None != actions:
      actionCount = 0
      for a in actions:
        actionCount += len(a)
      consolidateActions(consolidatedActions, actions)

  # And show what the plan is
  if options.verbose > 0:
    print "Planned actions"
    printConsolidatedActions(consolidatedActions)

  # And then do the moves here.
  sequenceMoves(consolidatedActions, fsf, options.maxreads, options.maxwrites, options.sparetransfers)

  cursor.close()
  db.close()

def gen_primes():
    """ Generate an infinite sequence of prime numbers. Taken from http://stackoverflow.com/questions/567222/simple-prime-generator-in-python
    """
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}  

    # The running integer that's checked for primeness
    q = 2  

    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            # 
            yield q        
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next 
            # multiples of its witnesses to prepare for larger
            # numbers
            # 
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        q += 1

def primeEqualOrGreaterThan(threshold):
  for prime in gen_primes():
    if prime >= threshold:
      return prime


class ExecutionEngine(object):

  # Takes a collection of actions, and then sequences them.

  # A token that represnets no host. Used to pad the sizes of the lists to a prime number in size
  BLANK = "-blank_" # RFC 1123 bans hostnames starting with a hyphen, or contaning an underscore. So this should be a good token.

  def __init__(self, actions, maxreads, maxwrites, sparethreads):
    sources = actions.keys()
    dests = set()
    # Rip out the _actual_ set of sources and destinations
    for actionList in actions.itervalues():
      for action in actionList:
        print str(action)
        #Strip out the dest of each.
        dests.add(action.dest.server)
    # actions is the set of actions to perform, indexed on the src server
    self.actions       = actions
    # Collection of the sources, in a deque, which is filled to prime size.
    self.sourceControl = collections.deque(sources) 
    # Collection of the destinations, in a deque, that is filled to prime size
    self.destControl   = collections.deque(dests)
    # dict of source to list of dest servers, used as a tag to indicate that a particular combination is completed.
    self.completed     = dict()
    for src in sources:
      self.completed[src] = set()
    self.maxThreads    = min(len(self.sourceControl), len(self.destControl)) # FIXME: Adjust for spare threads
    self.balanceQueueLengths()

  def markCompleted(self, src, dest):
    """Sets a flag to indicate that there are no more transfers from src -> dest
	Returns True if there are more cases where this dest _might_ be used (but is allowed to be wrong on that point.
	Will only return False if it can proove that there is no case where there might be a valid combo left"""
    self.completed[src].add(dest)
    print "Marking " + src + " -> " + dest + " as completed"
    # Now check to see if there are other cases where it is _not_ completed
    countOfDests = 0
    for completedDestSets in self.completed.itervalues():
      if dest in completedDestSets:
        countOfDests += 1
    # If there is one entry per source, then it's done.  So count the entries, and if that matches the number of sources we're done.
    # Note that we check against actions, in case the sourceControl has shrunk.
    return countOfDests != len(self.actions)

  def isCompleted(self, src, dest):
    """True if there is a flag, set by calling markCompleted(), in place for hte given src->dest comboination"""
    return dest in self.completed[src]

  def rebalanceQueueLengths(self):
    """Strips out all the blanks, and then balance them to prime lengths again.
	FIXME: Shoudl lock everything whilst it's running."""
    print "Rebalancing Queuse"

    for i in range(len(self.sourceControl)):
	src = self.sourceControl.popleft()
	if src != ExecutionEngine.BLANK:
		self.sourceControl.append(src)
    for i in range(len(self.destControl)):
	dst = self.destControl.popleft()
	if dst != ExecutionEngine.BLANK:
		self.destControl.append(dst)
    if len(self.sourceControl) == 1 or len(self.destControl) == 1:
        print "Down to only one source (or dest): exiting - bump up the fuzz factor for the next run?"
	sys.exit(0)
    self.balanceQueueLengths()


  def balanceQueueLengths(self):
    """Add blanks to the queues, so that they are both prime in length.  Do this by the simple (but marginally sub optimal) method of adding blanks to them till it is prime.
	Previous versions only added to one - but that allows for the longer one to be an exact multiple.  So have to add to both."""
    toAdd = primeEqualOrGreaterThan(len(self.sourceControl)) - len(self.sourceControl)
    for i in range(toAdd):
      self.sourceControl.append(ExecutionEngine.BLANK)
    random.shuffle(self.sourceControl)
    toAdd = primeEqualOrGreaterThan(len(self.destControl)) - len(self.destControl)
    for i in range(toAdd):
      self.destControl.append(ExecutionEngine.BLANK)
    random.shuffle(self.destControl)

  def descQueues(self):
    """Development / debugging description of the queues"""
    actionsLeft = 0
    for acts in self.actions.itervalues():
      actionsLeft += len(acts)
    return "src size = " + str(len(self.sourceControl)) + ", dest size = " + str(len(self.destControl)) + ", actions left = " + str(actionsLeft)

  def doMove(self, action):
    abstract

  def getNextAction(self, action):
    abstract

  def startExecution(self):
    abstract


class SourceOnlySingleMoveEngine(ExecutionEngine):
  # Ignores the destination, when selecting a move target

  def doMove(self, action):
    """Minimal function to do the move. It need not worry about internal state, and should just do the move, and return once complete."""
    print "Doing: " + str(action)
    time.sleep(1)
    return

  def getNextAction(self):
    """Returns a (src, dest) pair, which is the next set to move.  
	Should: raise an exeception when no more to do.
	should: Lock things so that this can be called in a re-entrant way"""
    while(True):
      src = self.sourceControl.popleft()
      dest = self.destControl.popleft()
      if(src == ExecutionEngine.BLANK or dest == ExecutionEngine.BLANK):
	# Retry, and hope not to get a blank this time
        self.sourceControl.append(src)
        self.destControl.append(src)
        # And fall throhgh, re-do the loop
      else:
    	# We now have a valid (i.e.) non-blank set to work with.
	# Find the next action for this.
	possibleActions = self.actions.get(src)
	print "Possible actions for " + src + " =  " + str(len(possibleActions)) + "; target is " + dest
	
	# FIXME: Ignore dest for hte moment
	action = possibleActions.pop();
        doRebalance = False
	if len(possibleActions) > 0:
	  self.sourceControl.append(src)
	else:
	  doRebalanace = True
	# FIXME:Check if there are more dests, else drop it and rebalabne.
	self.destControl.append(dest)
	if doRebalance:
          self.rebalanceQueueLengths()
        return action
 
  def startExecution(self):
    """This should do a single move at a time, and operate over all the sources (one at a time), and call doMove() one at a time for each."""
    while(True):
      action = self.getNextAction()
      self.doMove(action)
  
    # FIXME: Check about cyclign srcs
   

class SingleMoveEngine(ExecutionEngine):
  # Ignores the destination, when selecting a move target

  def doMove(self, action):
    """Minimal function to do the move. It need not worry about internal state, and should just do the move, and return once complete."""
    print "Doing: " + str(action)
    time.sleep(1)
    return

  def getNextAction(self):
    """Returns a (src, dest) pair, which is the next set to move.  
	Should: raise an exeception when no more to do.
	should: Lock things so that this can be called in a re-entrant way"""
    doubleBlanks = 0
    while(True):
      src = self.sourceControl.popleft()
      dest = self.destControl.popleft()
      if(src == ExecutionEngine.BLANK or dest == ExecutionEngine.BLANK):
	# Retry, and hope not to get a blank this time
        self.sourceControl.append(src)
        self.destControl.append(src)
	if src == dest:
	  print "Double blank for " + src + " -> " + dest
          if doubleBlanks > 5:
	    self.rebalanceQueueLengths()
            print self.descQueues()
            #sys.exit(0)
          else:
            doubleBlanks += 1
	  continue
      else:
    	# We now have a valid (i.e.) non-blank set to work with.
	# Find the next action for this.
	if self.isCompleted(src, dest):
          print "" + src + " -> " + dest + " is completed"
          continue

	possibleActions = self.actions.get(src)
	print "Possible actions for " + src + " =  " + str(len(possibleActions)) + "; target is " + dest
	
	action = None
	moreDests = False # Used to scan to see if there is another in this case.
	for trialAction in possibleActions:
	  # First matching one, save it and remove.
	  if None == action and trialAction.dest.server == dest:
            action = trialAction
	    possibleActions.remove(action)
	  elif trialAction.dest.server == dest:
	   moreDests = True
	   break
	
	# We now know if there is an action at all, and also if there is a second.

	if None == action:
	  # No matching dest found!
	  # That's actually ok, in a sense - it's always possible for a particular set to be disjoint
	  self.markCompleted(src, dest)
          self.sourceControl.append(src)
          self.destControl.append(dest)
	  continue

	doRebalance = False
	if len(possibleActions) > 0:
	  self.sourceControl.append(src)
	else:
	  # Shrinking a queue, so rebalance needed
	  doRebalanace = True
	
	if moreDests:
	  self.destControl.append(dest)
	else:
	  # We know that there are no more cases for this combination, so flag as such.
	  if self.markCompleted(src, dest):
            self.destControl.append(dest)
	  else:
	    doRebalance = True
	if doRebalance:
          self.rebalanceQueueLengths()
        return action
 
  def startExecution(self):
    """This should do a single move at a time, and operate over all the sources (one at a time), and call doMove() one at a time for each."""
    count = 0
    while(True):
      action = self.getNextAction()
      self.doMove(action)
      count += 1
      if count >= 20:
         print self.descQueues()
	 count = 1
  
    # FIXME: Check about cyclign srcs


def sequenceMoves(consolidatedActions, filesystems, maxreads, maxwrites, sparedests):
  engine = SingleMoveEngine(consolidatedActions, maxreads, maxwrites, sparedests)
  #maxThreads = min(len(sources) * maxreads, len(dests) * maxwrites - sparedests) # Number of threads needed to get things moving.
  print engine.descQueues()
  engine.startExecution()
 
def annotateFreeSpace(fss):
  """Find out the free space on the given filesystems, via an SSH call.  Returns a (potentially reduced) set of filesystems that we know are abel to recieve data."""
  #NOTE: this should be adjusted to get information via parsing dpm-qryconf or equiv (assuming enough resolution available)
  ret = list()
  fsInfo = dict()
  fails = set()
  for fs in fss:
    if fs.server in fails:
      continue
    if fs.server not in fsInfo.keys():
      try:
        fsInfo[fs.server] = readDFInfo(fs.server)
      except socket.error, v:
        errorcode = v[0]
	if errorcode == errno.EHOSTDOWN or errorcode == errno.EHOSTUNREACH:
	  print "Unable to contact " + fs.server + ": " + v[1]
          fails.add(fs.server)
	  continue
        else:
	  raise
    if fs.name not in fsInfo[fs.server]:
      print "Unable to find " + fs.name + " for " + fs.server + ": All I've got is " + str(fsInfo[fs.server].keys())
      continue
    fs.size = fsInfo[fs.server][fs.name].size
    fs.avail = fsInfo[fs.server][fs.name].avail
    ret.append(fs)
  return ret

def readDFInfo(host):
  """Read the file system info from df via ssh.  Could use some more error handling!"""
  class DFInfo(object):
    def __init__(self, server, name, size, used, avail):
      self.server = server
      self.name = name
      self.size = size
      self.used = used
      self.avail = avail
    def calcFree(self):
      return self.size - self.used
  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  ssh.connect(host)
  stdin, out, err = ssh.exec_command("df --portability")
  poolData = dict()
  header = out.readline() # And discard the headers
  for line in out: # XXX: Could read line at a time?
    if 0 == len(line):
      return poolData
    [filesystem, size, used, avail, percentUsed, loc, blank] = re.split('\s+', line)
    # Filesystem           1K-blocks      Used Available Use% Mounted on
    fs = DFInfo(host, loc, int(size), int(used), int(avail))
    poolData[loc] = fs
  return poolData

def consolidateActions(consolidatedActions, actions):
  """Take a list of list of actions, and put them all in a map of lists, with the map based on server name, and the list the moves"""
  for actionList in actions:
    for action in actionList:
      if not action.file.host in consolidatedActions:
        consolidatedActions[action.file.host] = list()
      moveList = consolidatedActions[action.file.host]
      moveList.append(action)

def printConsolidatedActions(consolidatedActions):
  """Pretty printer, to give an overview of what's planned to be done."""
  totalMoves = 0
  # Firstly, get a list of the targets
  targets = set()
  for targetList in consolidatedActions.itervalues():
    for target in targetList:
      totalMoves += 1
      targets.add(target.dest.server)

  print str(totalMoves) + " total file moves to be carried out"

  for target in sorted(targets):
    sys.stdout.write("\t" + target.split(".")[0])
  print ""

  for host in sorted(consolidatedActions.keys()):
    moves = consolidatedActions[host]
    movSplit = dict()
    for move in moves:
      if not move.dest.server in movSplit:
        movSplit[move.dest.server] = 0
      movSplit[move.dest.server] += 1
    sys.stdout.write(host.split(".")[0])
    for target in sorted(targets):
      if not target in movSplit:
        sys.stdout.write("\t  ")
      else:
        sys.stdout.write("\t" + str(movSplit[target]))
    print ""

def calculateMoveList(files, fsf, minfree, fuzzfactor, verbose=0):
  totalFiles = len(files)
  if verbose > 0:
    print "Total files = " + str(totalFiles)
  # First, partition the available filesystems into 2 sets.
  fslookup = dict()
  for fs in fsf:
    fs.files = list()
    fslookup[ (fs.server, fs.name)] = fs
  for file in files:
    if not (file.host, file.filesystem) in fslookup:
      print "Missing filesystem for " + str(file)
    else:
      fslookup[ (file.host, file.filesystem) ].files.append(file)

  (srcThreshold, destThreshold) = calcThresholds(totalFiles, len(fsf), minfree, fuzzfactor=fuzzfactor, verbose=verbose)

  if verbose > 0:
    print "(srcThreshold, destThreshold) = (" + str(srcThreshold) + ", " + str(destThreshold) + ")"

  srcs = list()
  dests = list()

  if verbose > 1:
    printFileCountByServer(fslookup.values())

  for fs in fslookup.itervalues():
    if fs.fileCount() >= srcThreshold:
      srcs.append(fs)
    elif fs.fileCount() <= destThreshold:
      dests.append(fs)

  if verbose > 2:
    printFileCountByServer(srcs)
  elif verbose > 1:
    print "Moving files from " + str(len(srcs)) + " filesystems"
  elif verbose > 0:
    print "Moving files from " + str(len(srcs)) + " filesystems"

  if verbose > 2:
    printFileCountByServer(dests)

  if 0 == len(srcs):
    return None

  # From here, we are doing real work.  Firstly, we want to ensure that we preserve the 'minfree' aspect, so that if we loose a server, then we don't loose all datasets.
  (moves, validDests) = preserveFreeServers(dests, srcs, minfree, verbose=verbose)

  srcs.sort(key=lambda x: x.fileCount(), reverse=True)

  for src in srcs:
    (moveSet, validDests) = spreadFilesOver(src, validDests, minLeft=srcThreshold, maxTarget=srcThreshold, verbose=verbose)
    moves.append(moveSet)

  return moves

def preserveFreeServers(possibleTargets, requiredSources, minFree, verbose=0):
  """Ensure that there are some servers without files for this set.  We are given a list of possible places where they can go, and we want to remove at least minfree _servers_ (not filesystems) from that list.
  We have the list of sources, so if one filesystem is loaded, then that can't be the free server.  Then, if there are empty servers, allow those to remain empty to get the next pass.  Finally, if the minimum is not met, then 
  select randomly from the possibles, _without_ recourse to minimising work."""
  # XXX: This function has not been well excercised - it may be buggy!
  possibleTargetServers = set()
  maybeEmptyTargets = set()
  nonEmptyTargets = set()
  for pt in possibleTargets:
    possibleTargetServers.add(pt.server)
    # A cheap way to pre-populate the empty/non-empty sets
    if pt.fileCount() == 0:
      maybeEmptyTargets.add(pt.server)
    else:
      nonEmptyTargets.add(pt.server)

  emptyTargets = maybeEmptyTargets.difference(nonEmptyTargets);

  for rs in requiredSources:
    possibleTargetServers.discard(rs.server)

  if verbose > 2:
     print "Possible free servers: " + str(possibleTargetServers)

  if len(emptyTargets) >= minFree:
    # Nice and simple - no moved needed, just give an emptymove list, and some of the empty servers
    preservedEmpty = random.sample(emptyTargets, minFree)
    if verbose > 0:
      print "Kept free: " + str(preservedEmpty)
    validTargets = possibleTargetServers.difference(preservedEmpty)
    return (list(), filesystemsWithServers(validTargets, possibleTargets) )

  # Less simple - need to pick out servers, and empty them
  neededServers = minFree - len(emptyTargets)
  toClear = random.sample(possibleTargetServers, neededServers)

  keptEmpty = toClear.union(emptyTargets)

  # Although they are not yet cleared, we want a lost of the valid targets to move things to, so that we can generate the move lists
  validTargets = possibleTargetServers.difference(keptEmpty)

  # this is a list of list of moves.  Each list of moves is a separeate 'taskset' - the exact meaning is not strictly defined, but finishing a task set should result in some measurable benefit.
  # It also prevents a lot of work concatenating moves...
  movesToSequence = list()

  # Rather then search for the required filesystems, and then dig them out, just loop over where all the files are, and if they are on the server to clear, then do so.
  for maybeClear in possibleTargets:
    if maybeClear.server in toClear:
      # Clear it, and maybe shrink the targets
      (moves, validTargets) = spreadFilesOver(maybeClear, validTargets, minLeft=0, verbose=verbose)
      movesToSequence.append(moves)
      print printFileCountByServer(validTargets)
    
  if verbose > 0:
    print "Kept free: " + str(emptyTargets)
    print "Cleared:   " + str(toClear)

  return (movesToSequence, filesystemsWithServers(validTargets, possibleTargets) )

def filesystemsWithServers(validServers, possibleFS):
  """Give a set of all the filesystems that have a server in validServers"""
  ret = set()
  for fs in possibleFS:
    if fs.server in validServers:
      ret.add(fs)

  return ret

def spreadFilesOver(source, dests, minLeft=None, maxTarget=None, verbose=0):
  """Corefunction to generate a moveList.  Spread files away from source, so that less than minLeft are left on the source.  Do not put more than maxTarget on any of the dests. 
  Note that this fuction returns a (potentiall) reduced set of valid targets, as it fills up possible places."""
  if len(dests) == 0:
    if verbose > 3:
      print "No valid dests, skipping spreading "  + source.desc()
    return (list(), dests)

  if verbose > 2:
    print "Spreading down to " + str(minLeft) + " files on : " + str(source)

  if minLeft == 0:
    # Move everything
    listOfFilesToMove = source.files
  else:
    numToMove = len(source.files) - minLeft
    listOfFilesToMove = random.sample(source.files, int(numToMove))
    
  moves = list()
  for fileToMove in listOfFilesToMove:
    if len(dests) == 0:
      print "Unable to complete move lists"
      return (moves, dests)
    target = random.sample(dests, 1).pop() # Can't use choice() on a set
    moves.append(Move(fileToMove, target))
    source.files.remove(fileToMove)
    target.files.append(fileToMove)
    if maxTarget != None and target.fileCount() >= maxTarget:
      if verbose > 3:
        print target.desc() + " filled with " + str(target.fileCount())
      dests.remove(target)

  return (moves, dests)

def printFileCountByServer(fsf):
  """Print out the number of files in each filesystem, on each server.  This is used to show data in a more compact form, hence this function does not change anything."""
  toDisp = sorted(fsf, key=lambda x: (x.server, x.name))
  prevHost = None
  for fs in toDisp:
    if fs.server != prevHost:
      prevHost = fs.server
      print "\n" + fs.server,
    print "\t" + str(fs.fileCount()),
  print "\n",

def calcThresholds(totalFiles, totalFilesystems, minfree, fuzzfactor=2.0, verbose=0):
  # Work out the targets
  # destThreshold: Number of files, if a filesystem has less than this, then can be a target for a move
  # srcThreshold:  Number of files, if a filesystem has more than this, then will be a source for a move
  availFilesystems = totalFilesystems - minfree

  evenDistribTarget = math.ceil( (totalFiles / float(availFilesystems)) )

  # fuzz: Gap from the true values.  Allow for there to be a section in both categories
  fuzz = (fuzzfactor - 1.0) * evenDistribTarget
  if fuzz < 0:
    # Just in case some put something silly in
    fuzz = 0 
  
  if verbose > 2:
    print "Fuzz = " + str(fuzz)

  srcThreshold = evenDistribTarget + fuzz
  destThreshold = evenDistribTarget

  return ( int(srcThreshold), int(destThreshold))

if __name__ == '__main__':
  main()



# ASIDE:
# To explore what the 'bunching' is currently like
"""
SELECT  m.parent_fileid, concat(r.host) AS loc, count(m.fileid) AS num, SUM(m.filesize) AS size 
INTO OUTFILE 'bunchingByHost'
FROM Cns_file_replica r
  JOIN Cns_file_metadata m USING (fileid)
WHERE r.setname = '9dc7a6dc-3f30-426a-8508-ac5bc6ca428a'
group by parent_fileid, loc
ORDER BY parent_fileid, num DESC
"""



"""

SELECT b.parent_fileid, b.files, b.peak, files / 151 AS target, b.ratio, (peak - CEILING(files / 151)) / CEILING(files / 151) AS badness
FROM (
  SELECT a.parent_fileid, sum(a.num) AS files, max(a.num) AS peak, max(a.num) / sum(a.num) AS ratio
  FROM (
    SELECT  m.parent_fileid, r.host, r.fs, count(m.fileid) AS num
    FROM Cns_file_replica r
      JOIN Cns_file_metadata m USING (fileid)
    WHERE r.setname = '9dc7a6dc-3f30-426a-8508-ac5bc6ca428a'
    GROUP BY parent_fileid, host, fs
    ) a
  GROUP BY parent_fileid
) b
WHERE b.files > 40
ORDER BY badness ASC;
"""
