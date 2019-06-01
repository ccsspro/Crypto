
<?php 

$command = escapeshellcmd('python mine.py');
$output = shell_exec($command);
echo $output;

?>