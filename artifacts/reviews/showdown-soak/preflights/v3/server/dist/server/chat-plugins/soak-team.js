// Local audit endpoint only. Returns the caller's original simulator team privately;
// neither a target player nor a battle mutation is accepted.
exports.commands = {
  async soakteam(target, room, user) {
    if (Config.bindaddress !== '127.0.0.1' || Config.serverid !== 'local-baseline' ||
        target || !room?.battle || !user.named) throw new Chat.ErrorMessage('Local own-team receipt only');
    const team = await room.battle.getTeam(user);
    if (!team) throw new Chat.ErrorMessage('Only a participant may request their own team');
    this.sendReply(`|soakteam|${Teams.pack(team)}`);
  },
};
