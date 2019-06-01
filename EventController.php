<?php
namespace App\Http\Controllers;
use Illuminate\Database\Eloquent\Collection;
use Illuminate\Support\Facades\DB;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use App\Helpers\AppHelper;
use \App\Model\Ema;
use \App\Model\Client;
use \App\Model\Interval;
use \App\Model\Test;
use \App\Model\Trend;
use \App\Model\Alert;
use \App\Model\Currency;
use \App\Model\Condition;
use \App\Model\Primary;
use \App\Model\Secondary;
use \App\Model\Conditions;


class EventController extends Controller
{
    public function index(){
    	$currencyFlag=0;
        $subCondition=0;
        $conditionFlag=0;
        $flg=0;
        
        $client_obj = Client::where('alertStatus', 1)->orderby('client_id','desc')->get();
        foreach($client_obj as $client_objs){
            $flg=0;

/*            $shift=Interval::where('client_id',$client_objs->client_id)->get();
            foreach($shift as $shifts){
                print_r(date('H:i:s',strtotime($shifts->fromInterval)));echo "<br>"; print_r(date('H:i:s'));
                if( date('H:i:s',strtotime($shifts->fromInterval)) > date('H:i:s') && date('H:i:s',strtotime($shifts->fromInterval)) > date('H:i:s')){
                    
                    $flg++;
                }
            }
            print_r($flg);die;*/
            if($flg==0){
                if((date("Y-m-d H:i:s", strtotime($client_objs->created_at . "+".$client_objs->scanInterval."minutes"))) > (date('Y-m-d H:i:s'))){
                $timeZone=str_replace('-','%2F',$client_objs->timeZone) ;
                $alert= Ema::where('client_id',$client_objs->client_id)->orderby('condition_id','desc')->first();
                for($i=2;$i<=$alert->condition_id;$i++){
                   $ema_obj[]=Ema::where('client_id',$client_objs->client_id)->where('condition_id',$i)->get();
                }
                foreach ($ema_obj as $key => $value) {
                   foreach ($value as $key => $values) {
                        $currency3=explode(",",$values->currency);
                        foreach($currency3 as $currency){
                            $currency=str_replace("/", "_", $currency);  
                            $ema1=AppHelper::getEMA($values->emaPeriod1,$currency,$values->timeFrame,$timeZone);      
                            $ema2=AppHelper::getEMA($values->emaPeriod2,$currency,$values->timeFrame,$timeZone);
                            if($values->crossing=="Above"){
                                if($ema1[0]>=$ema2[0]){
                                    $currencyFlag=1;
                                }else{
                                    $currencyFlag=0;
                                    break;
                                }   
                            }
                            if($values->crossing=='Under'){
                                if($ema1[0]<$ema2[0]){
                                    $currencyFlag=1;
                                }else{
                                    $currencyFlag=0;
                                    break;
                                }
                            }
                        }
                       // print_r($currencyFlag);
                        if($currencyFlag==1){
                            $subCondition=1;
                            $condition_obj=Condition::where('alertId',$client_objs->client_id)->where('conditionId',$values->condition_id)->first();
                            if($condition_obj){//print_r($condition_obj);
                                $condition_obj->status=1;
                                $condition_obj->save();
                            }
                        }    
                    }
                
                    if($subCondition==1){
                        $conditionFlag=1;
                    }else{
                        $conditionFlag=0;
                    }   
                }

                $condition_obj1=Condition::selectRaw('alertId, sum(status) as s,count(*) as c')->groupBy('alertId')->having('s','<>','c')->get();
                
                $ema=array();
                foreach ($condition_obj1 as $key => $value) {
                    if($value->alertId==$client_objs->client_id){
                        $currency=explode(",",$client_objs->currency);
                        $scanInterval=$client_objs->emaPeriod;
                        $timeFrame=$client_objs->timeFrame;
                        $condition=$client_objs->crossing;
                        foreach($currency as $currencies){
                            $currency2=str_replace("/", "_", $currencies);
                            $alertObj=Alert::where('alertName', $client_objs->alertName)->where('currency',$currency2)->first();
                            if($alertObj){
                                if((date("Y-m-d H:i:s", strtotime($alertObj->created_at . "+".$client_objs->reactivateInterval."minutes"))) < (date('Y-m-d H:i:s'))){
                                    $ema[]=AppHelper::getEMA($scanInterval,$currency2,$timeFrame,$timeZone);
                                    $alertObj->created_at=date('Y-m-d H:i:s');
                                    $status=$alertObj->save();
                                }else{
                                    break;
                                }
                            }else{
                                $ema[]=AppHelper::getEMA($scanInterval,$currency2,$timeFrame,$timeZone);
                                $altobj = new Alert();
                                $altobj->alertName = $client_objs->alertName;
                                $altobj->currency  = $currency2;
                                $altobj->save();
                            }
                        }
                        $icon1=AppHelper::getEmoji('\xF0\x9F\x94\xB5');
                        $icon2=AppHelper::getEmoji('\xF0\x9F\x94\xB4');
                        $cur=AppHelper::getEmoji('\xF0\x9F\x92\xB0');
                        $loc=AppHelper::getEmoji('\xF0\x9F\x93\x8D');
                        $hod=AppHelper::getEmoji('\xF0\x9F\x93\x88');
                        $lod=AppHelper::getEmoji('\xF0\x9F\x93\x89');
                        $news=AppHelper::getEmoji('\xF0\x9F\x93\xB0');
                        $chk=AppHelper::getEmoji('\xF0\x9F\x94\x96');
                        foreach ($ema as $value) {
                            $cross="";
                            $text="";
                            
                            if($value[1]>=$value[0]){
                                $text=$icon1.' BUY '.$icon1.chr(10);
                                $cross="over";
                            }
                             if($value[1]<$value[0]){
                                $text=$icon2.' SELL '.$icon2.chr(10);
                                $cross="under";
                            }
                            $text.=$cur.'Currency : '.$value[4].chr(10);
                            $text.=$loc.'Location : EMA  '.$value[5].chr(10);
                            $text.=$hod.'HOD :'.$value[2].chr(10);
                            $text.=$lod.'LOD :'.$value[3].chr(10);
                            $text.=$news.'NEWS : <a href="https://www.forexfactory.com/calendar.php">Forex Factory</a>'.chr(10); 
                            $text.=$chk.'CHECKLIST + CORRELATION'.chr(10);
                            //$status=AppHelper::sendMessage('-386420414', $text, false, 1, false); 
                            if($cross==$condition){
                                $obj = Trend::where('coin',$value[4])->where('alert',$client_objs->client_id)->first();
                                if($obj){
                                    if($cross=!$obj->trend){
                                        //$status=AppHelper::sendMessage('-386420414', $text, false, 1, false); 
                                        //$status=AppHelper::sendMessage('-1001162032776', $text, false, 1, false);
                                        $obj->trend=$cross;
                                        $obj->save();
                                    }
                                }else{
                                    $obj1=new Trend();
                                    $obj1->coin=$value[4];
                                    $obj1->alert=$client_objs->client_id;
                                    $obj1->trend=$cross;
                                    $obj1->save();
                                    //$status=AppHelper::sendMessage('-386420414', $text, false, 1, false); 
                                    //$status=AppHelper::sendMessage('-1001162032776', $text, false, 1, false);
                                }
                            }
                                         
                            // 
                        } 
                    }
                }    
            }
            else{
                $client_objs->created_at=date('Y-m-d H:i:s');
                //print_r($client_objs->created_at);
                $client_objs->save();
                print_r($client_objs->client_id);die;
                $obj= Condition::where('alertId', $client_objs->client_id)->get();
                print_r($obj);die;
                foreach($obj as $value){
                    $value->status=0;
                    $value->save();
                }
            }
            }
        }       
    }

    public function dashboard_save(Request $request){
        date_default_timezone_set('Europe/Berlin');
        $client_obj = new Client();
        $status=1;
        $client_obj->timeZone=$request->input('timeZone');
        $client_obj->scanInterval=$request->input('scanInterval');
        $client_obj->crossing=$request->input('crossing');
        $client_obj->emaPeriod=$request->input('emaPeriod');
        $currency=implode(',', $request->input('currency'));
        $client_obj->currency=$currency;
        $client_obj->timeFrame=$request->input('timeFrame');
        $client_obj->reactivateInterval=$request->input('reactivateInterval');
        $client_obj->alertName=$request->input('alertName');
        $client_obj->alertNote=$request->input('alertNote');
        $client_obj->alertStatus=$status;
        $client_obj->started_at=date('Y-m-d H:i:s');
        $client_obj->scan_start=date('Y-m-d H:i:s');
        $save=$client_obj->save();

        if($save){
            $emaPeriod1=$request->input('emaPeriod1');
            $crossing2=$request->input('crossing2');
            $emaPeriod2=$request->input('emaPeriod2');
            $currency2=$request->input('currency2');
            $condition=$request->input('condition');
            //print_r($currency2);die;
            $timeFrame2=$request->input('timeFrame2');
            //print_r($emaPeriod1);die;
            for($i=1;$i<=count($emaPeriod1);$i++){
                $ema_obj = new Ema();
                
                $ema_obj->client_id = $client_obj->client_id;
                $ema_obj->emaPeriod1 = $emaPeriod1[$i];
                $ema_obj->condition_id=$condition[$i];
                $ema_obj->crossing= $crossing2[$i];
                $ema_obj->emaPeriod2 = $emaPeriod2[$i];
                $currency=implode(',', $currency2[$i]);
                $ema_obj->currency=$currency;
                $ema_obj->timeFrame=$timeFrame2[$i];
                $save1=$ema_obj->save();
                $condition_obj= new Conditions();
                $condition_obj->client_id=$client_obj->client_id;              
                $condition_obj->alertId=$ema_obj->alert_id;
                $condition_obj->conditionId=$condition[$i];
                $condition_obj->status=0;
                $condition_obj->save();    //
            }
  
           
           if($save1){
                $fromInterval= $request->input('fromInterval');
                $toInterval= $request->input('toInterval');
                for($i=1;$i<=count($fromInterval);$i++){
                    $interval_obj = new Interval();
                    $interval_obj->client_id = $client_obj->client_id;
                    $interval_obj->fromInterval = $fromInterval[$i];
                    $interval_obj->toInterval = $toInterval[$i];
                    $save2=$interval_obj->save();
                }
               
            }
        }

        if ($save2) {
//            $this->showlist();
            return redirect()->route('showlist');
        } else {
            return redirect()->back()->with('error', "Failed to Send Please Try Again!");
        }
      
    }

    public function dashboard_update(Request $request){
        $id=$request->input('getId');
        $client_obj = Client::where('client_id',$id)->first();
       
        $status=1;
        $client_obj->timeZone=$request->input('timeZone');

        $client_obj->scanInterval=$request->input('scanInterval');
        $client_obj->crossing=$request->input('crossing');
        $client_obj->emaPeriod=$request->input('emaPeriod');
        $currency=implode(',', $request->input('currency'));
        $client_obj->currency=$currency;
        $client_obj->timeFrame=$request->input('timeFrame');
        $client_obj->reactivateInterval=$request->input('reactivateInterval');
        $client_obj->alertName=$request->input('alertName');
        $client_obj->alertNote=$request->input('alertNote');
        $client_obj->alertStatus=$status;
        $client_obj->started_at=date('Y-m-d H:i:s');
        $save=$client_obj->save();


        if($save){
            $conditions=$request->input('conditions');
            $emaPeriod1=$request->input('emaPeriod1');
            $crossing2=$request->input('crossing2');
            $emaPeriod2=$request->input('emaPeriod2');
            $currency2=$request->input('currency2');
            $condition=$request->input('condition');
            
            $timeFrame2=$request->input('timeFrame2');
            //print_r($emaPeriod1);die;
            foreach ($conditions as $key => $values) {
                $ema_obj = Ema::where('alert_id',$values)->first();
                $ema_obj->emaPeriod1 = $emaPeriod1[$key];

                $ema_obj->condition_id=$condition[$key];
                $ema_obj->crossing= $crossing2[$key];
                $ema_obj->emaPeriod2 = $emaPeriod2[$key];
                $currency=implode(',', $currency2[$key]);
                $ema_obj->currency=$currency;
                $ema_obj->timeFrame=$timeFrame2[$key];

                $save1=$ema_obj->save();

            	$condition_obj= Conditions::where('alertId',$values)->where('conditionId',$condition[$key])->first();
            	if($condition_obj){
            		$condition_obj->status=0;
                	$condition_obj->save();
            	}          
                   //
            }

           
           if($save1){
                $fromInterval= $request->input('fromInterval');
                $toInterval= $request->input('toInterval');
                for($i=1;$i<=count($fromInterval);$i++){
                    $interval_obj = Interval::where('client_id',$client_obj->client_id)->first();
                    $interval_obj->client_id = $client_obj->client_id;
                    $interval_obj->fromInterval = $fromInterval[$i];
                    $interval_obj->toInterval = $toInterval[$i];
                    $save2=$interval_obj->save();
                }
               
            }
        }

        if ($save2) {
            return $this->showlist();
        } else {
            return redirect()->back()->with('error', "Failed to Send Please Try Again!");
        }
      
    }

    public function editdata(){
        date_default_timezone_set('Europe/Berlin');
        $id=$_POST['id'];
        $data=$_POST['data'];
        $timeZone='';
        $condition='';
        $scanInterval='';
        $crossing='';
        $emaPeriod='';
        $currency= array();
        $timeFrame='';
        $emaPeriod1=array(array());
        $crossing2=array(array());
        $emaPeriod2=array(array());
        $currency2=array(array());
        $timeFrame2=array(array());
        $fromInterval=array(array());
        $toInterval=array(array());
        $array=array(array());
        $array1=array(array());
        $k=0;
        $j=0;
        $client_obj = Client::where('client_id',$id)->first();
        for($i=0;$i<count($data);$i++){
            if($data[$i]['name']=='timeZone'){
                $timeZone=$data[$i]['value'];
            }
            if($data[$i]['name']=='scanInterval'){
                $scanInterval=$data[$i]['value'];
            }
            if($data[$i]['name']=='crossing'){
                $crossing=$data[$i]['value'];
            }
            if($data[$i]['name']=='emaPeriod'){
                $emaPeriod=$data[$i]['value'];
            }
            if($data[$i]['name']=='currency'){
            	//print_r($data[$i]['value']);die;
               array_push($currency,$data[$i]['value']);
            }
            if($data[$i]['name']=='timeFrame'){
                $timeFrame=$data[$i]['value'];
            }
            if($data[$i]['name']=='condition'){
                $condition=$data[$i]['value'];
            }
            if($data[$i]['name']=='emaPeriod1'){
                //array_push($emaPeriod1[$condition],$data[$i]['value']);
                $array[$condition][$k]['emaPeriod1']=$data[$i]['value'];
            }
            if($data[$i]['name']=='crossing2'){
                $array[$condition][$k]['crossing2']=$data[$i]['value'];
            }
            if($data[$i]['name']=='emaPeriod2'){
                $array[$condition][$k]['emaPeriod2']=$data[$i]['value'];
            }
            if($data[$i]['name']=='currency2'){
                $array[$condition][$k]['currency2'][]=$data[$i]['value'];
            }
            if($data[$i]['name']=='timeFrame2'){
                $array[$condition][$k]['timeFrame2']=$data[$i]['value'];
            }
            if($data[$i]['name']=='end'){
                $k++;
            }
            if($data[$i]['name']=='reactivateInterval'){
                $reactivateInterval=$data[$i]['value'];
            }
            if($data[$i]['name']=='alertName'){
                $alertName=$data[$i]['value'];
            }
            if($data[$i]['name']=='alertNote'){
                $alertNote=$data[$i]['value'];
            }
            if($data[$i]['name']=='fromInterval'){
            	$array1[$j]['fromInterval']=$data[$i]['value'];
            }
            if($data[$i]['name']=='toInterval'){
            	$array1[$j]['toInterval']=$data[$i]['value'];
            	$j++;
            }

        }
        //print_r($currency);die;
        DB::table('alertLog')->where('alertId',$id)->delete();
        DB::table('trend')->where('alert',$id)->delete();
        DB::table('primary_values')->where('alertId',$id)->delete();
        $ema_obj1=Ema::where('client_id',$id)->get();
        foreach ($ema_obj1 as $value) {
        	DB::table('conditionLog')->where('alertId', $value->alert_id)->delete();
        	DB::table('secondary_values')->where('alertId', $value->alert_id)->delete();
        }
        DB::table('condition_status')->where('client_id',$id)->delete();
        DB::table('ema_alert')->where('client_id',$id)->delete();
        $client_obj->timeZone=$timeZone;
        $client_obj->scanInterval=$scanInterval;
        $client_obj->crossing=$crossing;
        $client_obj->emaPeriod=$emaPeriod;
        $currency=implode(',', $currency);
        $client_obj->currency=$currency;
        $client_obj->timeFrame=$timeFrame;
        $client_obj->reactivateInterval=$reactivateInterval;
        $client_obj->alertName=$alertName;
        $client_obj->alertNote=$alertNote;
        $client_obj->save();
       
        for($i=2;$i<=count($array);$i++){
           foreach ($array[$i] as  $value) {
                $emaobj=new EMA();
                $emaobj->client_id=$id;
                $emaobj->condition_id=$i;
                $emaobj->emaPeriod1=$value['emaPeriod1'];
                $emaobj->crossing=$value['crossing2'];
                $emaobj->emaPeriod2=$value['emaPeriod2'];
                $currency=implode(',', $value['currency2']);
                $emaobj->currency=$currency;
                $emaobj->timeFrame=$value['timeFrame2'];
                $emaobj->save();
                $condition= new Conditions();
                $condition->client_id=$id;
                $condition->alertId=$emaobj->alert_id;
                $condition->conditionId=$i;
                $condition->status=0;
                $condition->save();
           }
        }
        DB::table('interval_details')->where('client_id',$id)->delete();
        foreach ($array1 as  $value) {
        	$interval=new Interval();
        	$interval->client_id=$id;
        	$interval->fromInterval=$value['fromInterval'];
        	$interval->toInterval=$value['toInterval'];
        	$interval->save();
        }
        return 1;       
    }

    public function entrydata(){
        date_default_timezone_set('Europe/Berlin');
        $data=$_POST['data'];
        //print_r($data);die;
        $timeZone='';
        $condition='';
        $scanInterval='';
        $crossing='';
        $emaPeriod='';
        $currency= array();
        $timeFrame='';
        $emaPeriod1=array(array());
        $crossing2=array(array());
        $emaPeriod2=array(array());
        $currency2=array(array());
        $timeFrame2=array(array());
        $fromInterval=array(array());
        $toInterval=array(array());
        $array=array(array());
        $array1=array(array());
        $k=0;
        $j=0;
        for($i=0;$i<count($data);$i++){
            if($data[$i]['name']=='timeZone'){
                $timeZone=$data[$i]['value'];
            }
            if($data[$i]['name']=='scanInterval'){
                $scanInterval=$data[$i]['value'];
            }
            if($data[$i]['name']=='crossing'){
                $crossing=$data[$i]['value'];
            }
            if($data[$i]['name']=='emaPeriod'){
                $emaPeriod=$data[$i]['value'];
            }
            if($data[$i]['name']=='currency'){
            	//print_r($data[$i]['value']);die;
               array_push($currency,$data[$i]['value']);
            }
            if($data[$i]['name']=='timeFrame'){
                $timeFrame=$data[$i]['value'];
            }
            if($data[$i]['name']=='condition'){
                $condition=$data[$i]['value'];
            }
            if($data[$i]['name']=='emaPeriod1'){
                //array_push($emaPeriod1[$condition],$data[$i]['value']);
                $array[$condition][$k]['emaPeriod1']=$data[$i]['value'];
                
            }
            if($data[$i]['name']=='crossing2'){
                $array[$condition][$k]['crossing2']=$data[$i]['value'];
            }
            if($data[$i]['name']=='emaPeriod2'){
                $array[$condition][$k]['emaPeriod2']=$data[$i]['value'];
            }
            if($data[$i]['name']=='currency2'){
                $array[$condition][$k]['currency2'][]=$data[$i]['value'];
            }
            if($data[$i]['name']=='timeFrame2'){
                $array[$condition][$k]['timeFrame2']=$data[$i]['value'];
            }
            if($data[$i]['name']=='end'){
                $k++;
            }
            if($data[$i]['name']=='reactivateInterval'){
                $reactivateInterval=$data[$i]['value'];
            }
            if($data[$i]['name']=='alertName'){
                $alertName=$data[$i]['value'];
            }
            if($data[$i]['name']=='alertNote'){
                $alertNote=$data[$i]['value'];
            }
            if($data[$i]['name']=='fromInterval'){
            	$array1[$j]['fromInterval']=$data[$i]['value'];
            }
            if($data[$i]['name']=='toInterval'){
            	$array1[$j]['toInterval']=$data[$i]['value'];
            	$j++;
            }

        }
        //print_r($array[2]);
        $client_obj= new Client();
        $client_obj->timeZone=$timeZone;
        $client_obj->scanInterval=$scanInterval;
        $client_obj->crossing=$crossing;
        $client_obj->emaPeriod=$emaPeriod;
        $currency=implode(',', $currency);
        $client_obj->currency=$currency;
        $client_obj->timeFrame=$timeFrame;
        $client_obj->alertStatus=1;
        $client_obj->reactivateInterval=$reactivateInterval;
        $client_obj->alertName=$alertName;
        $client_obj->alertNote=$alertNote;
        $client_obj->save();
        for($i=2;$i<=count($array);$i++){
        	
           foreach ($array[$i] as  $value) {
           		
                $emaobj=new EMA();
                $emaobj->client_id=$client_obj->client_id;
                $emaobj->condition_id=$i;
                $emaobj->emaPeriod1=$value['emaPeriod1'];
                $emaobj->crossing=$value['crossing2'];
                $emaobj->emaPeriod2=$value['emaPeriod2'];
                $currency=implode(',', $value['currency2']);
                $emaobj->currency=$currency;
                $emaobj->timeFrame=$value['timeFrame2'];
                $emaobj->save();
                $condition= new Conditions();
                $condition->client_id=$client_obj->client_id;
                $condition->alertId=$emaobj->alert_id;
                $condition->conditionId=$i;
                $condition->status=0;
                $condition->save();
           }
        }

        foreach ($array1 as  $value) {
        	$interval=new Interval();
        	$interval->client_id=$client_obj->client_id;;
        	$interval->fromInterval=$value['fromInterval'];
        	$interval->toInterval=$value['toInterval'];
        	$interval->save();
        }
        $status=AppHelper::newalert();
        return 1;       
    }

    public function alertDetails(){
        $id=$_POST['id'];
        $client_obj = Client::where('client_id',$id)->get();
        $ema_obj = Ema::where('client_id',$id)->orderby('condition_id','desc')->first();
        for($i=2;$i<=$ema_obj->condition_id;$i++){
            $ema_obj1[]=Ema::where('client_id',$id)->where('condition_id',$i)->get();
        }
        $interval_obj = Interval::where('client_id',$id)->get();
        $response['client']=$client_obj;
        $response['ema']=$ema_obj1;
        $response['interval']=$interval_obj;
        return $response;
        // return view("list")->with(compact("client_obj","ema_obj","interval_obj"));

    }

    public function showlist(){
        $client_obj = Client::where('alertStatus', 1)->orderby('client_id','desc')->get();
        return view('list')->with(compact("client_obj"));
    }

    public function ema_check(){
        AppHelper::ema_calc();
        $test_obj = Test::orderby('id','desc')->get();
        return view('test')->with(compact("test_obj"));
    }

    public function duplicateDetails(){
        $id=$_POST['id'];
        $client_obj = Client::where('client_id',$id)->first();
        $client_obj1 = $client_obj->replicate();
        $client_obj1->started_at=date('Y-m-d H:i:s');
        $client_obj1->scan_start=date('Y-m-d H:i:s');
        $client_obj1->save();
        $ema_obj = Ema::where('client_id',$id)->get();
        foreach($ema_obj as $ema_objs){
            $ema_obj1 = $ema_objs->replicate();
            $ema_obj1->client_id=$client_obj1->client_id;
            $ema_obj1->save();

        }
        $ema_alert=Conditions::where('client_id',$id)->get();
        foreach ($ema_alert as  $value) {
            $alert=$value->replicate();
            $alert->client_id=$client_obj1->client_id;
            $alert->save();
        }
        $interval_obj = Interval::where('client_id',$id)->get();
        foreach($interval_obj as $interval_objs){
            $interval_obj1 = $interval_objs->replicate();
            $interval_obj1->client_id=$client_obj1->client_id;
            $interval_obj1->save();
        }  
        return 1;
    }

    public function cancelAlert(){
        $id=$_POST['id'];
        $client_obj = Client::where('client_id',$id)->first();
        $client_obj->alertStatus = 0;
        $client_obj->save();
        
        return 1;
    }

    public function testAlert(){
        $currency="EUR_USD";
        $period="M15";
        $timeZone="America%2FNew_York";
        $scanInterval=800;
        $ema=AppHelper::getEMA($scanInterval,$currency,'M15',$timeZone);
        $ema1=AppHelper::getEMA($scanInterval,$currency,'H1',$timeZone);
        $ema2=AppHelper::getEMA($scanInterval,$currency,'H4',$timeZone);
        //$ema3=AppHelper::getEMA($scanInterval,$currency,$period,$timeZone);
        print_r($ema);echo "<br>";
        print_r($ema1);echo "<br>";
        print_r($ema2);
       // print_r($ema3);
    }

    public function updateDetails($id){
        //$id=$_GET['id'];
        $currency= DB::table('ema10m15')->distinct()->get(['currency']);
        //print_r($currency);die;
        $client_obj = Client::where('client_id',$id)->get();
        $ema_obj = Ema::where('client_id',$id)->orderby('condition_id','desc')->first();
        for($i=2;$i<=$ema_obj->condition_id;$i++){
            $ema_obj1[]=Ema::where('client_id',$id)->where('condition_id',$i)->get();
        }
        $interval_obj = Interval::where('client_id',$id)->get();
        
        //return $response;
        return view("edit")->with(compact("client_obj","ema_obj1","interval_obj","currency"));

    }

    public function primaryCheck(){
        
        $client_obj = Client::where('alertStatus', 1)->orderby('client_id','desc')->get();

        $trend="";
        foreach($client_obj as $client_objs){
            if((date("Y-m-d H:i:s", strtotime($client_objs->started_at . "+".$client_objs->reactivateInterval."minutes"))) > (date('Y-m-d H:i:s'))){
                if((date("Y-m-d H:i:s", strtotime($client_objs->started_at . "+".$client_objs->scanInterval."minutes"))) > (date('Y-m-d H:i:s'))){

                    $currency=explode(",",$client_objs->currency);
                    $scanInterval=$client_objs->emaPeriod;
                    $timeFrame=$client_objs->timeFrame;
                    $condition=$client_objs->crossing;
                    $timeZone=str_replace('-','%2F',$client_objs->timeZone);
                    foreach($currency as $currencies){
                        $currency=str_replace("/", "_", $currencies);
                        $alertObj=Alert::where('alertId', $client_objs->client_id)->where('currency',$currency)->first();
                        if($alertObj){    
                            $ema=AppHelper::getEMA($scanInterval,$currency,$timeFrame,$timeZone);
                        }else{
                            $alertObj= new Alert();
                            $alertObj->alertId=$client_objs->client_id;
                            $alertObj->currency=$currency;
                            $alertObj->plotting=0;
                            $alertObj->save();
                            $ema=AppHelper::getEMA($scanInterval,$currency,$timeFrame,$timeZone);
                            if($ema[1]>$ema[0]){
                                $alertObj->prime="over";
                                $alertObj->save();
                            }
                            else{
                                $alertObj->prime="under";
                                $alertObj->save();
                            }
                        }
                        $emaobj=Primary::where('alertId',$client_objs->client_id)->first();
                        if($emaobj){
                            $emaobj->price=$ema[1];
                            $emaobj->ema=$ema[0];
                            $emaobj->save();
                        }else{                      
                            $emaobj=new Primary();
                            $emaobj->alertId=$client_objs->client_id;
                            $emaobj->price=$ema[1];
                            $emaobj->ema=$ema[0];
                            $emaobj->save();
                        }

                        if($ema[1]>$ema[0]){
                            $trend="over";
                        }
                        else{
                            $trend="under";
                        }
                        print_r($ema);
                        print_r($condition);
                        print_r($trend);
                        print_r($alertObj->prime);
                        
                        
                        echo "<br>";
                        if($trend!=$alertObj->prime){
                            $alertObj->prime=$trend;
                            $alertObj->save();
                            if($condition==$trend){
                                print_r("TEST1");
                                if($alertObj->plotting==0){
                                    $alertObj->plotting=1;
                                    $alertObj->status=1;
                                    $alertObj->save();
                                }
                            }                  
                        }
                    }
                }else{
                    //$client_objs->started_at=date('Y-m-d H:i:s');
                    $client_objs->status=0;
                    $client_objs->save();
                    $alertObj=Alert::where('alertId', $client_objs->client_id)->get();
                    foreach($alertObj as $value){
                        $value->status=0;
                        $value->plotting=0;
                        $value->save();
                    }
                    $emaobj=EMA::where('client_id', $client_objs->client_id)->get();
                    foreach($emaobj as $value){
                        $value->status=0;
                        $value->save();
                        $alertObj=Condition::where('alertId', $value->alert_id)->first();
                        if($alertObj){
                            $alertObj->status=0;
                            $alertObj->save();
                        }    
                    }
                    $condition=Conditions::where('alertId',$client_objs->client_id)->get();
                    foreach($condition as $value){
                        $value->status=0;
                        $value->save();
                    }
                }
            }else{
                $client_objs->started_at=date('Y-m-d H:i:s');
                $client_objs->status=0;
                $client_objs->save();
                $status=DB::table('alertLog')->where('alertId',$client_objs->client_id)->delete();
                $trend=Trend::where('alert',$client_objs->client_id)->get();
                if(count($trend)>0){
                    $status=DB::table('trend')->where('alert',$client_objs->client_id)->delete();
                }
                
                $condition=EMA::where('client_id',$client_objs->client_id)->get();
                if(count($condition)>0){
                    foreach($condition as $value){
                        $value->status=0;
                        $value->save();
                        $alertObj=Condition::where('alertId', $value->alert_id)->first();
                        if($alertObj){
                            $alertObj->status=0;
                            $alertObj->save();
                        }
                        $status=DB::table('conditionLog')->where('alertId',$value->client_id)->delete();
                    }   
                }
                
            }
        }
    }

    public function secondarycheck(){
        $client_obj = Client::where('alertStatus', 1)->orderby('client_id','desc')->get();

        foreach($client_obj as $client_objs){
            if($client_objs->status==1){
                
                if((date("Y-m-d H:i:s", strtotime($client_objs->started_at . "+".$client_objs->scanInterval."minutes"))) > (date('Y-m-d H:i:s'))){
                    $timeZone=str_replace('-','%2F',$client_objs->timeZone);
                    $ema_obj=Ema::where('client_id',$client_objs->client_id)->where('status',0)->get();
                    //print_r($ema_obj);die;
                    foreach ($ema_obj as $value){
                        $currencys=explode(",",$value->currency);
                        $condition=$value->crossing;
                        
                        foreach ($currencys as $currency) {
                            $currency=str_replace("/", "_", $currency);
                            $alertObj=Condition::where('alertId', $value->alert_id)->where('currency',$currency)->first();
                            //print_r($alertObj);
                            if($alertObj){
                                if($alertObj->status!=1){
                                    $ema1=AppHelper::getEMA($value->emaPeriod1,$currency,$value->timeFrame,$timeZone);      
                                    $ema2=AppHelper::getEMA($value->emaPeriod2,$currency,$value->timeFrame,$timeZone);
                                }else{
                                    break;
                                }
                            }else{
                                $alertObj= new Condition();
                                $alertObj->alertId=$value->alert_id;
                                $alertObj->condition_id=$value->condition_id;
                                $alertObj->currency=$currency;
                                $alertObj->save();
                                $ema1=AppHelper::getEMA($value->emaPeriod1,$currency,$value->timeFrame,$timeZone);      
                                $ema2=AppHelper::getEMA($value->emaPeriod2,$currency,$value->timeFrame,$timeZone);
                            }
                            $emaobj=Secondary::where('alertId',$alertObj->id)->first();
                            if($emaobj){
                                $emaobj->ema1=$ema1[0];
                                $emaobj->ema2=$ema2[0];
                                $emaobj->save();
                            }else{
                                $emaobj=new Secondary();
                                $emaobj->alertId=$alertObj->id;
                                $emaobj->ema1=$ema1[0];
                                $emaobj->ema2=$ema2[0];
                                $emaobj->save();
                            }

                            if($condition=="Above"){
                                if($ema1[0]>$ema2[0]){
                                    $alertObj->status=1;
                                    $alertObj->save();
                                }
                            }
                            if($condition=="Under"){
                                if($ema1[0]<$ema2[0]){
                                    $alertObj->status=1;
                                    $alertObj->save();
                                }
                            } 
                        }
                    }
                }        
            }            
        }
    }

    public function statusUpdate(){
        date_default_timezone_set('Europe/Berlin');
        $data = DB::table('ema10m15')->get();
        foreach ($data as $value) {
            $timeZone='Europe%2FBerlin';
            if($value->period=='H1'){
                $ema=AppHelper::getEMA($value->ema,$value->currency,$value->period,$timeZone);
                print_r($ema);
                print_r('<br>');
                DB::table('ema10m15')->where('id', $value->id)->update(['value' => $ema[0]],['date' => date('Y-m-d H:i:s') ]);
            }
        }

    }

    public function details($id){
        //$id=$_GET['id'];
        
        //print_r($id);
        $client_obj = Client::where('client_id',$id)->join('alertLog', 'client_preference.client_id','=','alertLog.alertId')->join('primary_values','alertLog.alertId','=','primary_values.alertId')->get();
       // $client_obj = Client::where('client_id',$id)->get();       
        
        //print_r($client_obj);die;
        //$ema_obj1 = Ema::where('client_id',$id)->orderby('condition_id','desc')->first();
        //print_r($ema_obj1);die;
        //for($i=2;$i<=$ema_obj1->condition_id;$i++){
            $ema_obj= Ema::where('client_id',$id)->join('conditionLog', 'ema_alert.alert_id','=','conditionLog.alertId')->join('secondary_values','conditionLog.id','=','secondary_values.alertId')->get();
            //$ema_obj= Ema::where('client_id',$id)->get();
           // print_r($ema_obj);die;
        //}
        $interval_obj = Interval::where('client_id',$id)->get();
        $trend=Trend::where('alert',$id)->first();
        if(count($client_obj)>0){
        	return view("details")->with(compact("client_obj","ema_obj","interval_obj","trend"));
        }else{
        	return $this->showlist();
        }

        
        

    }

     public function detailsnew($id){
       
        $client_obj = Client::where('client_id',$id)->join('alertLog', 'client_preference.client_id','=','alertLog.alertId')->join('primary_values','alertLog.alertId','=','primary_values.alertId')->get();
        $ema_obj= Ema::where('client_id',$id)->join('conditionLog', 'ema_alert.alert_id','=','conditionLog.alertId')->join('secondary_values','conditionLog.id','=','secondary_values.alertId')->get();
        $result['client']=$client_obj;
        $result['ema']=$ema_obj;
        return response()->JSON($result);

    }

    public function sendMessage(){/*
        $icon1=AppHelper::getEmoji('\xF0\x9F\x94\xB5');
        $icon2=AppHelper::getEmoji('\xF0\x9F\x94\xB4');
        $cur=AppHelper::getEmoji('\xF0\x9F\x92\xB0');
        $loc=AppHelper::getEmoji('\xF0\x9F\x93\x8D');
        $hod=AppHelper::getEmoji('\xF0\x9F\x93\x88');
        $lod=AppHelper::getEmoji('\xF0\x9F\x93\x89');
        $news=AppHelper::getEmoji('\xF0\x9F\x93\xB0');
        $chk=AppHelper::getEmoji('\xF0\x9F\x94\x96');

        $client_obj = Client::where('client_preference.status',1)->join('alertLog', 'client_preference.client_id','=','alertLog.alertId')->join('primary_values','alertLog.alertId','=','primary_values.alertId')->get(); 
        foreach ($client_obj as $value) {
            $f=0;
            $ema_obj= Ema::where('client_id',$value->client_id)->get();
            foreach ($ema_obj as $values) {
                print_r($values);
                if($values->status==1){
                    $f++;
                }
            }
            print_r($f);
            if($f==count($ema_obj)){
                if($value->crossing=="over")
                {
                    $text=$icon1.' BUY '.$icon1.chr(10);
                }
                if($value->crossing=="under"){
                    $text=$icon2.' SELL '.$icon2.chr(10);
                }
                
                $timeZone=str_replace('-','%2F',$value->timeZone);
               // print_r($value->timeFrame);die;
                $ema=AppHelper::getEMA($value->emaPeriod,$value->currency,$value->timeFrame,$timeZone);
                $text.=$cur.'Currency : '.$value->currency.chr(10);
                $text.=$loc.'Location : EMA  '.$value->emaPeriod.chr(10);
                $text.=$hod.'HOD :'.$ema[2].chr(10);
                $text.=$lod.'LOD :'.$ema[3].chr(10);
                $text.=$news.'NEWS : <a href="https://www.forexfactory.com/calendar.php">Forex Factory</a>'.chr(10); 
                $text.=$chk.$value->alertNote.chr(10);
                $obj = Trend::where('coin',$value->currency)->where('alert',$value->client_id)->first();
                //print_r($obj);die;
                if($obj){
                    if((date("Y-m-d H:i:s", strtotime($obj->updated_at . "+".$value->reactivateInterval."minutes"))) > (date('Y-m-d H:i:s'))){
                        if($value->crossing=!$obj->trend){
                            //$status=AppHelper::sendMessage('-386420414', $text, false, 1, false); 
                            //$status=AppHelper::sendMessage('-1001162032776', $text, false, 1, false);
                            print_r($text);
                            $obj->trend=$value->crossing;
                            $obj->save();
                        }
                    }
                }else{
                    $obj1=new Trend();
                    $obj1->coin=$value->currency;
                    $obj1->alert=$value->client_id;
                    $obj1->trend=$value->crossing;
                    $obj1->save();
                    print_r($text);
                    //$status=AppHelper::sendMessage('-386420414', $text, false, 1, false); 
                    //$status=AppHelper::sendMessage('-1001162032776', $text, false, 1, false);
                }
            }
        }*/
    $status=AppHelper::sendMessage('--1001395879666', "WELCOME MESSAGE", false, 1, false);
    print_r($status);
    }

    public function removeedit($id){
        $ema=EMA::where('alert_id',$id)->first();
        $obj=EMA::where('client_id',$ema->client_id)->get();
        if(count($obj)>1){
            $status=DB::table('ema_alert')->where('alert_id',  $id)->delete();
            $status=DB::table('condition_status')->where('alertId', $id)->delete();        
            $status=DB::table('conditionLog')->where('alertId',  $id)->delete();
            return 1;
        }else
        {
            return 0;
        }
        
    }
}

?>