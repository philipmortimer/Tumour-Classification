/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aitumourclasification;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 *
 * @author mortimer
 */
public class AITumourClasification {

    /**
     * @param args the command line arguments
     */
    public static final String HAS_BEEN_PROCESSED="dataBeenPreProcessed.txt";
    public static final String PROCESSED="true";public static final String NOT_PROCESSED="false";
    public static final String TEST_DATA="testData.txt";public static final String TRAIN_DATA ="trainData.txt";
    public static final int MAX_NO_OF_TEST_IMAGES_M=1499;public static final int MAX_NO_OF_TEST_IMAGES_B= 1800;
    public static final int MAX_NO_OF_TRAIN_IMAGES_M=1500;public static final int MAX_NO_OF_TRAIN_IMAGES_B=1799;
    public static final String SEP_CHAR=",";
    public static final String TRAIN_PATH_M="archive"+File.separator+"train"+File.separator+"malignant"+File.separator;
    public static final String TRAIN_PATH_B="archive"+File.separator+"train"+File.separator+"benign"+File.separator;
    public static final String TEST_PATH_M="archive"+File.separator+"test"+File.separator+"malignant"+File.separator;
    public static final String TEST_PATH_B="archive"+File.separator+"test"+File.separator+"benign"+File.separator;
    public static final String BEST_NET="bestNet.txt";
    public static final int NO_OF_TRAIN_IMAGES=2637; public static final int NO_OF_TEST_IMAGES = 660;
    public static final int WIDTH_OF_DATA = 3*(224*224)+2;
    //public static final double NORMALISE_VALUE=-16777216;
    public static final double NORMALISE_VALUE=255.0/2.0;//255.0/2.0;
    public static final double BIAS_TO_DATA=-1.0;//-1.0
    public static final String[]TRAIN_M={"2.jpg","5.jpg","6.jpg","7.jpg","9.jpg","10.jpg","11.jpg","12.jpg","14.jpg","15.jpg","16.jpg","18.jpg","19.jpg","20.jpg","21.jpg","22.jpg","23.jpg","25.jpg","26.jpg","27.jpg","28.jpg","29.jpg","30.jpg","31.jpg","32.jpg","34.jpg","35.jpg","36.jpg","37.jpg","38.jpg","40.jpg","41.jpg","42.jpg","43.jpg","45.jpg","47.jpg","48.jpg","49.jpg","50.jpg","51.jpg","53.jpg","55.jpg","56.jpg","59.jpg","60.jpg","61.jpg","63.jpg","64.jpg","67.jpg","69.jpg","70.jpg","71.jpg","72.jpg","75.jpg","76.jpg","77.jpg","78.jpg","79.jpg","80.jpg","82.jpg","83.jpg","84.jpg","86.jpg","88.jpg","89.jpg","90.jpg","94.jpg","95.jpg","96.jpg","97.jpg","98.jpg","99.jpg","100.jpg","101.jpg","102.jpg","103.jpg","104.jpg","105.jpg","106.jpg","107.jpg","108.jpg","109.jpg","111.jpg","112.jpg","113.jpg","114.jpg","115.jpg","116.jpg","117.jpg","118.jpg","119.jpg","120.jpg","121.jpg","122.jpg","123.jpg","124.jpg","125.jpg","126.jpg","127.jpg","128.jpg","130.jpg","131.jpg","132.jpg","133.jpg","134.jpg","135.jpg","136.jpg","137.jpg","138.jpg","140.jpg","141.jpg","142.jpg","143.jpg","144.jpg","145.jpg","146.jpg","147.jpg","148.jpg","149.jpg","150.jpg","151.jpg","152.jpg","153.jpg","154.jpg","155.jpg","157.jpg","158.jpg","159.jpg","160.jpg","162.jpg","163.jpg","164.jpg","165.jpg","166.jpg","167.jpg","168.jpg","169.jpg","170.jpg","171.jpg","172.jpg","174.jpg","175.jpg","176.jpg","177.jpg","178.jpg","180.jpg","181.jpg","183.jpg","187.jpg","188.jpg","189.jpg","191.jpg","192.jpg","194.jpg","197.jpg","198.jpg","199.jpg","201.jpg","202.jpg","203.jpg","204.jpg","205.jpg","206.jpg","207.jpg","209.jpg","210.jpg","211.jpg","212.jpg","213.jpg","214.jpg","215.jpg","216.jpg","217.jpg","218.jpg","219.jpg","220.jpg","221.jpg","222.jpg","223.jpg","224.jpg","225.jpg","226.jpg","227.jpg","228.jpg","229.jpg","230.jpg","233.jpg","234.jpg","235.jpg","236.jpg","239.jpg","240.jpg","241.jpg","242.jpg","243.jpg","244.jpg","245.jpg","246.jpg","248.jpg","249.jpg","250.jpg","251.jpg","252.jpg","253.jpg","254.jpg","255.jpg","259.jpg","261.jpg","262.jpg","263.jpg","264.jpg","265.jpg","266.jpg","267.jpg","268.jpg","269.jpg","270.jpg","271.jpg","272.jpg","273.jpg","274.jpg","277.jpg","278.jpg","279.jpg","280.jpg","281.jpg","282.jpg","283.jpg","284.jpg","285.jpg","286.jpg","287.jpg","288.jpg","289.jpg","290.jpg","291.jpg","292.jpg","293.jpg","294.jpg","295.jpg","296.jpg","297.jpg","298.jpg","299.jpg","300.jpg","301.jpg","302.jpg","304.jpg","305.jpg","306.jpg","308.jpg","309.jpg","311.jpg","312.jpg","313.jpg","314.jpg","315.jpg","316.jpg","319.jpg","320.jpg","321.jpg","323.jpg","324.jpg","325.jpg","326.jpg","327.jpg","328.jpg","329.jpg","330.jpg","331.jpg","332.jpg","333.jpg","334.jpg","335.jpg","336.jpg","337.jpg","339.jpg","340.jpg","341.jpg","344.jpg","345.jpg","346.jpg","347.jpg","349.jpg","350.jpg","351.jpg","352.jpg","353.jpg","354.jpg","357.jpg","358.jpg","359.jpg","361.jpg","362.jpg","363.jpg","365.jpg","366.jpg","367.jpg","368.jpg","369.jpg","370.jpg","371.jpg","372.jpg","373.jpg","374.jpg","376.jpg","377.jpg","378.jpg","379.jpg","380.jpg","383.jpg","385.jpg","386.jpg","387.jpg","388.jpg","391.jpg","392.jpg","393.jpg","395.jpg","396.jpg","397.jpg","398.jpg","400.jpg","401.jpg","403.jpg","404.jpg","406.jpg","408.jpg","410.jpg","411.jpg","412.jpg","413.jpg","414.jpg","415.jpg","416.jpg","417.jpg","418.jpg","420.jpg","422.jpg","423.jpg","425.jpg","427.jpg","429.jpg","431.jpg","432.jpg","433.jpg","434.jpg","435.jpg","438.jpg","439.jpg","440.jpg","441.jpg","442.jpg","443.jpg","444.jpg","446.jpg","447.jpg","448.jpg","449.jpg","451.jpg","452.jpg","453.jpg","454.jpg","455.jpg","456.jpg","457.jpg","458.jpg","460.jpg","461.jpg","462.jpg","464.jpg","465.jpg","466.jpg","467.jpg","468.jpg","469.jpg","470.jpg","471.jpg","472.jpg","473.jpg","475.jpg","476.jpg","477.jpg","478.jpg","480.jpg","481.jpg","482.jpg","484.jpg","485.jpg","486.jpg","487.jpg","488.jpg","489.jpg","490.jpg","491.jpg","492.jpg","493.jpg","494.jpg","495.jpg","496.jpg","497.jpg","498.jpg","499.jpg","500.jpg","502.jpg","503.jpg","505.jpg","506.jpg","509.jpg","510.jpg","512.jpg","513.jpg","514.jpg","515.jpg","516.jpg","517.jpg","518.jpg","520.jpg","521.jpg","522.jpg","523.jpg","524.jpg","525.jpg","526.jpg","527.jpg","529.jpg","530.jpg","531.jpg","532.jpg","533.jpg","534.jpg","535.jpg","537.jpg","538.jpg","539.jpg","540.jpg","541.jpg","542.jpg","543.jpg","544.jpg","545.jpg","547.jpg","548.jpg","549.jpg","550.jpg","551.jpg","552.jpg","553.jpg","555.jpg","557.jpg","558.jpg","559.jpg","560.jpg","561.jpg","563.jpg","564.jpg","565.jpg","566.jpg","567.jpg","569.jpg","571.jpg","574.jpg","576.jpg","577.jpg","578.jpg","579.jpg","580.jpg","581.jpg","582.jpg","583.jpg","584.jpg","585.jpg","586.jpg","587.jpg","588.jpg","589.jpg","591.jpg","594.jpg","595.jpg","597.jpg","598.jpg","599.jpg","600.jpg","601.jpg","604.jpg","605.jpg","606.jpg","607.jpg","608.jpg","609.jpg","610.jpg","611.jpg","612.jpg","613.jpg","616.jpg","617.jpg","618.jpg","620.jpg","621.jpg","622.jpg","623.jpg","624.jpg","625.jpg","626.jpg","627.jpg","629.jpg","630.jpg","631.jpg","632.jpg","633.jpg","634.jpg","635.jpg","636.jpg","637.jpg","639.jpg","640.jpg","642.jpg","644.jpg","645.jpg","646.jpg","647.jpg","648.jpg","649.jpg","650.jpg","651.jpg","652.jpg","654.jpg","655.jpg","656.jpg","657.jpg","659.jpg","661.jpg","662.jpg","663.jpg","664.jpg","665.jpg","666.jpg","668.jpg","669.jpg","671.jpg","672.jpg","673.jpg","675.jpg","676.jpg","677.jpg","678.jpg","680.jpg","681.jpg","682.jpg","683.jpg","684.jpg","685.jpg","687.jpg","688.jpg","689.jpg","690.jpg","691.jpg","692.jpg","693.jpg","695.jpg","697.jpg","698.jpg","699.jpg","700.jpg","701.jpg","702.jpg","703.jpg","704.jpg","705.jpg","707.jpg","708.jpg","710.jpg","711.jpg","712.jpg","714.jpg","715.jpg","716.jpg","717.jpg","718.jpg","719.jpg","720.jpg","721.jpg","722.jpg","723.jpg","724.jpg","725.jpg","726.jpg","727.jpg","728.jpg","729.jpg","730.jpg","731.jpg","732.jpg","733.jpg","734.jpg","736.jpg","737.jpg","738.jpg","740.jpg","741.jpg","742.jpg","743.jpg","745.jpg","746.jpg","748.jpg","750.jpg","751.jpg","752.jpg","753.jpg","754.jpg","755.jpg","756.jpg","758.jpg","760.jpg","761.jpg","764.jpg","765.jpg","766.jpg","768.jpg","769.jpg","770.jpg","772.jpg","774.jpg","775.jpg","777.jpg","779.jpg","780.jpg","781.jpg","782.jpg","783.jpg","784.jpg","785.jpg","786.jpg","787.jpg","789.jpg","790.jpg","791.jpg","793.jpg","795.jpg","796.jpg","798.jpg","799.jpg","800.jpg","801.jpg","802.jpg","803.jpg","805.jpg","806.jpg","807.jpg","809.jpg","810.jpg","811.jpg","812.jpg","813.jpg","814.jpg","815.jpg","816.jpg","818.jpg","819.jpg","820.jpg","821.jpg","823.jpg","824.jpg","825.jpg","827.jpg","828.jpg","829.jpg","830.jpg","831.jpg","833.jpg","834.jpg","835.jpg","837.jpg","840.jpg","841.jpg","842.jpg","843.jpg","844.jpg","845.jpg","846.jpg","847.jpg","848.jpg","849.jpg","850.jpg","851.jpg","853.jpg","854.jpg","855.jpg","856.jpg","857.jpg","858.jpg","860.jpg","861.jpg","862.jpg","863.jpg","865.jpg","867.jpg","869.jpg","870.jpg","872.jpg","873.jpg","875.jpg","876.jpg","877.jpg","878.jpg","879.jpg","880.jpg","882.jpg","883.jpg","884.jpg","885.jpg","886.jpg","888.jpg","889.jpg","890.jpg","891.jpg","892.jpg","895.jpg","896.jpg","897.jpg","898.jpg","899.jpg","900.jpg","901.jpg","902.jpg","903.jpg","904.jpg","906.jpg","907.jpg","908.jpg","909.jpg","910.jpg","911.jpg","912.jpg","913.jpg","914.jpg","915.jpg","916.jpg","917.jpg","918.jpg","919.jpg","920.jpg","921.jpg","922.jpg","923.jpg","924.jpg","925.jpg","926.jpg","928.jpg","929.jpg","930.jpg","931.jpg","932.jpg","933.jpg","935.jpg","936.jpg","938.jpg","939.jpg","941.jpg","942.jpg","943.jpg","944.jpg","945.jpg","946.jpg","950.jpg","951.jpg","952.jpg","954.jpg","955.jpg","956.jpg","957.jpg","958.jpg","959.jpg","960.jpg","961.jpg","962.jpg","963.jpg","964.jpg","965.jpg","966.jpg","967.jpg","968.jpg","969.jpg","970.jpg","971.jpg","972.jpg","973.jpg","974.jpg","975.jpg","976.jpg","977.jpg","978.jpg","979.jpg","980.jpg","981.jpg","982.jpg","983.jpg","984.jpg","986.jpg","987.jpg","988.jpg","989.jpg","990.jpg","991.jpg","993.jpg","996.jpg","997.jpg","998.jpg","999.jpg","1000.jpg","1001.jpg","1002.jpg","1004.jpg","1006.jpg","1008.jpg","1010.jpg","1011.jpg","1013.jpg","1014.jpg","1015.jpg","1016.jpg","1017.jpg","1018.jpg","1020.jpg","1021.jpg","1023.jpg","1024.jpg","1025.jpg","1028.jpg","1029.jpg","1030.jpg","1031.jpg","1032.jpg","1033.jpg","1034.jpg","1035.jpg","1036.jpg","1039.jpg","1040.jpg","1041.jpg","1042.jpg","1043.jpg","1044.jpg","1045.jpg","1047.jpg","1048.jpg","1050.jpg","1051.jpg","1052.jpg","1053.jpg","1054.jpg","1055.jpg","1057.jpg","1059.jpg","1060.jpg","1061.jpg","1062.jpg","1063.jpg","1064.jpg","1066.jpg","1067.jpg","1068.jpg","1069.jpg","1070.jpg","1071.jpg","1072.jpg","1073.jpg","1075.jpg","1076.jpg","1077.jpg","1078.jpg","1079.jpg","1081.jpg","1082.jpg","1084.jpg","1085.jpg","1086.jpg","1087.jpg","1088.jpg","1090.jpg","1092.jpg","1093.jpg","1095.jpg","1096.jpg","1097.jpg","1098.jpg","1099.jpg","1100.jpg","1101.jpg","1102.jpg","1103.jpg","1104.jpg","1105.jpg","1106.jpg","1107.jpg","1108.jpg","1109.jpg","1110.jpg","1111.jpg","1114.jpg","1115.jpg","1116.jpg","1117.jpg","1118.jpg","1121.jpg","1122.jpg","1123.jpg","1124.jpg","1125.jpg","1126.jpg","1127.jpg","1129.jpg","1130.jpg","1131.jpg","1132.jpg","1133.jpg","1134.jpg","1136.jpg","1137.jpg","1138.jpg","1139.jpg","1140.jpg","1142.jpg","1143.jpg","1144.jpg","1145.jpg","1146.jpg","1147.jpg","1148.jpg","1149.jpg","1152.jpg","1153.jpg","1154.jpg","1157.jpg","1158.jpg","1159.jpg","1161.jpg","1162.jpg","1163.jpg","1165.jpg","1167.jpg","1168.jpg","1169.jpg","1171.jpg","1172.jpg","1174.jpg","1175.jpg","1177.jpg","1178.jpg","1179.jpg","1180.jpg","1181.jpg","1182.jpg","1183.jpg","1184.jpg","1188.jpg","1190.jpg","1191.jpg","1192.jpg","1193.jpg","1195.jpg","1196.jpg","1199.jpg","1200.jpg","1201.jpg","1202.jpg","1203.jpg","1204.jpg","1205.jpg","1206.jpg","1207.jpg","1208.jpg","1209.jpg","1210.jpg","1212.jpg","1213.jpg","1214.jpg","1217.jpg","1219.jpg","1222.jpg","1223.jpg","1224.jpg","1225.jpg","1226.jpg","1230.jpg","1231.jpg","1232.jpg","1233.jpg","1235.jpg","1236.jpg","1237.jpg","1239.jpg","1240.jpg","1241.jpg","1243.jpg","1244.jpg","1245.jpg","1246.jpg","1247.jpg","1248.jpg","1249.jpg","1250.jpg","1252.jpg","1253.jpg","1255.jpg","1257.jpg","1258.jpg","1259.jpg","1260.jpg","1261.jpg","1263.jpg","1264.jpg","1265.jpg","1269.jpg","1270.jpg","1271.jpg","1272.jpg","1273.jpg","1274.jpg","1275.jpg","1276.jpg","1277.jpg","1278.jpg","1279.jpg","1281.jpg","1282.jpg","1283.jpg","1284.jpg","1285.jpg","1287.jpg","1288.jpg","1290.jpg","1291.jpg","1293.jpg","1294.jpg","1296.jpg","1298.jpg","1299.jpg","1301.jpg","1302.jpg","1303.jpg","1304.jpg","1305.jpg","1306.jpg","1307.jpg","1308.jpg","1309.jpg","1311.jpg","1312.jpg","1313.jpg","1314.jpg","1315.jpg","1316.jpg","1317.jpg","1318.jpg","1319.jpg","1321.jpg","1323.jpg","1324.jpg","1325.jpg","1326.jpg","1327.jpg","1328.jpg","1329.jpg","1331.jpg","1332.jpg","1334.jpg","1335.jpg","1337.jpg","1338.jpg","1340.jpg","1341.jpg","1343.jpg","1344.jpg","1345.jpg","1346.jpg","1348.jpg","1349.jpg","1350.jpg","1351.jpg","1352.jpg","1353.jpg","1356.jpg","1357.jpg","1360.jpg","1361.jpg","1362.jpg","1363.jpg","1365.jpg","1367.jpg","1368.jpg","1369.jpg","1371.jpg","1372.jpg","1373.jpg","1375.jpg","1376.jpg","1377.jpg","1381.jpg","1384.jpg","1385.jpg","1386.jpg","1387.jpg","1388.jpg","1389.jpg","1390.jpg","1391.jpg","1392.jpg","1393.jpg","1394.jpg","1396.jpg","1397.jpg","1398.jpg","1399.jpg","1400.jpg","1403.jpg","1404.jpg","1405.jpg","1406.jpg","1407.jpg","1408.jpg","1409.jpg","1410.jpg","1411.jpg","1412.jpg","1413.jpg","1414.jpg","1415.jpg","1416.jpg","1418.jpg","1419.jpg","1420.jpg","1421.jpg","1423.jpg","1424.jpg","1426.jpg","1428.jpg","1429.jpg","1431.jpg","1432.jpg","1433.jpg","1434.jpg","1435.jpg","1437.jpg","1439.jpg","1440.jpg","1441.jpg","1442.jpg","1444.jpg","1445.jpg","1446.jpg","1448.jpg","1449.jpg","1450.jpg","1452.jpg","1453.jpg","1454.jpg","1455.jpg","1456.jpg","1457.jpg","1458.jpg","1459.jpg","1460.jpg","1462.jpg","1463.jpg","1465.jpg","1466.jpg","1467.jpg","1468.jpg","1469.jpg","1470.jpg","1471.jpg","1472.jpg","1473.jpg","1474.jpg","1475.jpg","1476.jpg","1477.jpg","1478.jpg","1479.jpg","1480.jpg","1481.jpg","1482.jpg","1483.jpg","1485.jpg","1486.jpg","1487.jpg","1488.jpg","1490.jpg","1491.jpg","1492.jpg","1493.jpg","1494.jpg","1496.jpg","1497.jpg","1498.jpg","1500.jpg"};
    public static final String[]TRAIN_B={"3.jpg","4.jpg","6.jpg","7.jpg","12.jpg","13.jpg","14.jpg","17.jpg","19.jpg","20.jpg","21.jpg","22.jpg","23.jpg","24.jpg","25.jpg","26.jpg","27.jpg","28.jpg","29.jpg","30.jpg","32.jpg","33.jpg","34.jpg","35.jpg","36.jpg","38.jpg","39.jpg","40.jpg","41.jpg","42.jpg","43.jpg","46.jpg","47.jpg","48.jpg","49.jpg","50.jpg","51.jpg","52.jpg","53.jpg","55.jpg","56.jpg","58.jpg","59.jpg","60.jpg","62.jpg","64.jpg","65.jpg","66.jpg","67.jpg","68.jpg","69.jpg","70.jpg","72.jpg","73.jpg","75.jpg","76.jpg","77.jpg","78.jpg","79.jpg","80.jpg","81.jpg","82.jpg","85.jpg","86.jpg","87.jpg","89.jpg","91.jpg","92.jpg","93.jpg","94.jpg","96.jpg","99.jpg","100.jpg","101.jpg","102.jpg","103.jpg","104.jpg","106.jpg","107.jpg","108.jpg","109.jpg","110.jpg","111.jpg","112.jpg","113.jpg","114.jpg","115.jpg","116.jpg","117.jpg","118.jpg","120.jpg","121.jpg","123.jpg","124.jpg","125.jpg","126.jpg","128.jpg","129.jpg","130.jpg","131.jpg","132.jpg","133.jpg","134.jpg","135.jpg","136.jpg","137.jpg","138.jpg","139.jpg","141.jpg","142.jpg","143.jpg","145.jpg","146.jpg","148.jpg","149.jpg","150.jpg","151.jpg","152.jpg","153.jpg","155.jpg","156.jpg","157.jpg","158.jpg","159.jpg","160.jpg","162.jpg","163.jpg","164.jpg","165.jpg","166.jpg","167.jpg","168.jpg","169.jpg","170.jpg","171.jpg","172.jpg","173.jpg","174.jpg","175.jpg","176.jpg","177.jpg","178.jpg","179.jpg","180.jpg","181.jpg","182.jpg","183.jpg","184.jpg","187.jpg","188.jpg","189.jpg","190.jpg","191.jpg","192.jpg","193.jpg","194.jpg","195.jpg","196.jpg","197.jpg","198.jpg","199.jpg","200.jpg","201.jpg","202.jpg","203.jpg","204.jpg","206.jpg","207.jpg","208.jpg","209.jpg","210.jpg","212.jpg","213.jpg","214.jpg","217.jpg","218.jpg","219.jpg","220.jpg","221.jpg","222.jpg","223.jpg","225.jpg","227.jpg","228.jpg","229.jpg","230.jpg","231.jpg","232.jpg","233.jpg","234.jpg","235.jpg","236.jpg","238.jpg","239.jpg","240.jpg","241.jpg","242.jpg","243.jpg","244.jpg","246.jpg","247.jpg","248.jpg","249.jpg","252.jpg","254.jpg","255.jpg","257.jpg","258.jpg","260.jpg","261.jpg","262.jpg","263.jpg","264.jpg","265.jpg","266.jpg","269.jpg","270.jpg","271.jpg","272.jpg","274.jpg","275.jpg","276.jpg","277.jpg","278.jpg","279.jpg","280.jpg","281.jpg","282.jpg","283.jpg","285.jpg","286.jpg","288.jpg","290.jpg","292.jpg","293.jpg","294.jpg","295.jpg","296.jpg","297.jpg","298.jpg","299.jpg","301.jpg","302.jpg","305.jpg","306.jpg","307.jpg","308.jpg","309.jpg","310.jpg","312.jpg","314.jpg","315.jpg","316.jpg","318.jpg","319.jpg","320.jpg","321.jpg","324.jpg","325.jpg","326.jpg","327.jpg","328.jpg","329.jpg","331.jpg","333.jpg","334.jpg","336.jpg","337.jpg","338.jpg","339.jpg","340.jpg","341.jpg","342.jpg","343.jpg","345.jpg","346.jpg","347.jpg","349.jpg","350.jpg","351.jpg","352.jpg","353.jpg","355.jpg","357.jpg","358.jpg","359.jpg","360.jpg","361.jpg","362.jpg","363.jpg","364.jpg","365.jpg","366.jpg","368.jpg","370.jpg","371.jpg","372.jpg","373.jpg","374.jpg","375.jpg","376.jpg","377.jpg","380.jpg","381.jpg","382.jpg","383.jpg","384.jpg","386.jpg","387.jpg","390.jpg","391.jpg","392.jpg","393.jpg","395.jpg","396.jpg","397.jpg","398.jpg","399.jpg","400.jpg","401.jpg","402.jpg","403.jpg","405.jpg","406.jpg","408.jpg","409.jpg","410.jpg","411.jpg","412.jpg","413.jpg","415.jpg","416.jpg","417.jpg","418.jpg","420.jpg","421.jpg","422.jpg","423.jpg","424.jpg","425.jpg","426.jpg","429.jpg","430.jpg","431.jpg","432.jpg","434.jpg","435.jpg","438.jpg","439.jpg","440.jpg","441.jpg","442.jpg","443.jpg","445.jpg","446.jpg","447.jpg","448.jpg","450.jpg","451.jpg","452.jpg","453.jpg","455.jpg","456.jpg","458.jpg","459.jpg","460.jpg","461.jpg","463.jpg","464.jpg","465.jpg","466.jpg","467.jpg","468.jpg","469.jpg","470.jpg","471.jpg","472.jpg","473.jpg","474.jpg","476.jpg","477.jpg","478.jpg","480.jpg","481.jpg","482.jpg","483.jpg","485.jpg","486.jpg","487.jpg","488.jpg","489.jpg","490.jpg","491.jpg","492.jpg","493.jpg","494.jpg","496.jpg","497.jpg","498.jpg","499.jpg","500.jpg","501.jpg","502.jpg","503.jpg","504.jpg","505.jpg","506.jpg","507.jpg","508.jpg","509.jpg","510.jpg","511.jpg","512.jpg","513.jpg","514.jpg","515.jpg","516.jpg","517.jpg","518.jpg","519.jpg","521.jpg","522.jpg","523.jpg","524.jpg","525.jpg","526.jpg","528.jpg","529.jpg","530.jpg","531.jpg","533.jpg","534.jpg","537.jpg","538.jpg","539.jpg","540.jpg","541.jpg","542.jpg","543.jpg","544.jpg","546.jpg","547.jpg","549.jpg","550.jpg","551.jpg","553.jpg","554.jpg","555.jpg","556.jpg","557.jpg","563.jpg","564.jpg","565.jpg","567.jpg","568.jpg","569.jpg","570.jpg","571.jpg","576.jpg","577.jpg","578.jpg","579.jpg","581.jpg","582.jpg","583.jpg","584.jpg","585.jpg","586.jpg","587.jpg","589.jpg","590.jpg","591.jpg","592.jpg","593.jpg","594.jpg","595.jpg","597.jpg","598.jpg","599.jpg","600.jpg","601.jpg","602.jpg","604.jpg","605.jpg","606.jpg","607.jpg","608.jpg","609.jpg","610.jpg","611.jpg","613.jpg","614.jpg","615.jpg","616.jpg","617.jpg","618.jpg","619.jpg","620.jpg","621.jpg","622.jpg","625.jpg","627.jpg","628.jpg","629.jpg","630.jpg","632.jpg","633.jpg","637.jpg","638.jpg","639.jpg","640.jpg","641.jpg","643.jpg","645.jpg","646.jpg","648.jpg","649.jpg","650.jpg","651.jpg","652.jpg","653.jpg","654.jpg","655.jpg","656.jpg","657.jpg","661.jpg","662.jpg","665.jpg","666.jpg","667.jpg","669.jpg","671.jpg","672.jpg","673.jpg","674.jpg","675.jpg","677.jpg","679.jpg","680.jpg","682.jpg","683.jpg","684.jpg","685.jpg","686.jpg","687.jpg","688.jpg","691.jpg","692.jpg","693.jpg","694.jpg","695.jpg","696.jpg","697.jpg","698.jpg","699.jpg","701.jpg","702.jpg","703.jpg","704.jpg","705.jpg","706.jpg","707.jpg","708.jpg","709.jpg","710.jpg","711.jpg","713.jpg","714.jpg","715.jpg","716.jpg","718.jpg","719.jpg","722.jpg","723.jpg","726.jpg","727.jpg","728.jpg","729.jpg","730.jpg","731.jpg","732.jpg","733.jpg","734.jpg","735.jpg","736.jpg","737.jpg","738.jpg","739.jpg","740.jpg","741.jpg","742.jpg","743.jpg","744.jpg","745.jpg","746.jpg","747.jpg","748.jpg","749.jpg","750.jpg","752.jpg","753.jpg","754.jpg","755.jpg","756.jpg","757.jpg","759.jpg","760.jpg","761.jpg","764.jpg","765.jpg","766.jpg","767.jpg","769.jpg","771.jpg","772.jpg","775.jpg","776.jpg","777.jpg","779.jpg","780.jpg","781.jpg","782.jpg","783.jpg","784.jpg","785.jpg","786.jpg","787.jpg","788.jpg","789.jpg","791.jpg","792.jpg","793.jpg","794.jpg","795.jpg","796.jpg","797.jpg","798.jpg","799.jpg","800.jpg","801.jpg","803.jpg","804.jpg","807.jpg","808.jpg","809.jpg","810.jpg","811.jpg","812.jpg","814.jpg","815.jpg","816.jpg","817.jpg","818.jpg","821.jpg","822.jpg","827.jpg","829.jpg","831.jpg","832.jpg","833.jpg","834.jpg","835.jpg","836.jpg","837.jpg","838.jpg","839.jpg","841.jpg","842.jpg","843.jpg","844.jpg","845.jpg","846.jpg","847.jpg","848.jpg","849.jpg","850.jpg","851.jpg","852.jpg","853.jpg","855.jpg","856.jpg","857.jpg","858.jpg","859.jpg","860.jpg","861.jpg","862.jpg","863.jpg","864.jpg","865.jpg","866.jpg","867.jpg","868.jpg","870.jpg","871.jpg","872.jpg","873.jpg","875.jpg","876.jpg","877.jpg","879.jpg","881.jpg","882.jpg","883.jpg","884.jpg","885.jpg","886.jpg","889.jpg","890.jpg","891.jpg","896.jpg","897.jpg","898.jpg","899.jpg","900.jpg","901.jpg","902.jpg","904.jpg","905.jpg","907.jpg","908.jpg","910.jpg","911.jpg","913.jpg","914.jpg","915.jpg","917.jpg","918.jpg","920.jpg","921.jpg","922.jpg","923.jpg","924.jpg","926.jpg","927.jpg","928.jpg","929.jpg","930.jpg","931.jpg","932.jpg","933.jpg","934.jpg","935.jpg","937.jpg","938.jpg","939.jpg","940.jpg","941.jpg","942.jpg","945.jpg","947.jpg","948.jpg","949.jpg","950.jpg","951.jpg","952.jpg","953.jpg","954.jpg","955.jpg","956.jpg","957.jpg","959.jpg","960.jpg","961.jpg","963.jpg","965.jpg","966.jpg","967.jpg","968.jpg","969.jpg","970.jpg","971.jpg","973.jpg","974.jpg","975.jpg","976.jpg","977.jpg","978.jpg","979.jpg","980.jpg","981.jpg","983.jpg","984.jpg","985.jpg","986.jpg","987.jpg","988.jpg","989.jpg","990.jpg","991.jpg","992.jpg","994.jpg","995.jpg","996.jpg","997.jpg","999.jpg","1000.jpg","1001.jpg","1002.jpg","1004.jpg","1005.jpg","1007.jpg","1008.jpg","1009.jpg","1010.jpg","1011.jpg","1012.jpg","1014.jpg","1015.jpg","1016.jpg","1017.jpg","1020.jpg","1021.jpg","1022.jpg","1024.jpg","1026.jpg","1027.jpg","1028.jpg","1030.jpg","1031.jpg","1032.jpg","1035.jpg","1036.jpg","1037.jpg","1038.jpg","1039.jpg","1040.jpg","1041.jpg","1043.jpg","1045.jpg","1046.jpg","1047.jpg","1048.jpg","1049.jpg","1050.jpg","1051.jpg","1052.jpg","1053.jpg","1054.jpg","1056.jpg","1057.jpg","1058.jpg","1059.jpg","1060.jpg","1062.jpg","1063.jpg","1064.jpg","1065.jpg","1066.jpg","1067.jpg","1068.jpg","1069.jpg","1071.jpg","1072.jpg","1073.jpg","1075.jpg","1076.jpg","1078.jpg","1079.jpg","1081.jpg","1082.jpg","1083.jpg","1084.jpg","1085.jpg","1086.jpg","1089.jpg","1090.jpg","1091.jpg","1093.jpg","1094.jpg","1095.jpg","1096.jpg","1097.jpg","1098.jpg","1099.jpg","1100.jpg","1101.jpg","1102.jpg","1103.jpg","1104.jpg","1105.jpg","1106.jpg","1107.jpg","1108.jpg","1109.jpg","1111.jpg","1113.jpg","1114.jpg","1116.jpg","1117.jpg","1119.jpg","1120.jpg","1121.jpg","1122.jpg","1123.jpg","1124.jpg","1127.jpg","1128.jpg","1129.jpg","1131.jpg","1132.jpg","1133.jpg","1136.jpg","1137.jpg","1138.jpg","1140.jpg","1141.jpg","1142.jpg","1144.jpg","1145.jpg","1147.jpg","1148.jpg","1149.jpg","1150.jpg","1152.jpg","1153.jpg","1154.jpg","1156.jpg","1157.jpg","1158.jpg","1159.jpg","1160.jpg","1161.jpg","1162.jpg","1163.jpg","1164.jpg","1165.jpg","1167.jpg","1168.jpg","1169.jpg","1171.jpg","1172.jpg","1174.jpg","1175.jpg","1176.jpg","1178.jpg","1179.jpg","1181.jpg","1182.jpg","1184.jpg","1185.jpg","1186.jpg","1187.jpg","1188.jpg","1189.jpg","1190.jpg","1192.jpg","1193.jpg","1194.jpg","1195.jpg","1196.jpg","1197.jpg","1198.jpg","1199.jpg","1201.jpg","1202.jpg","1204.jpg","1205.jpg","1206.jpg","1207.jpg","1209.jpg","1210.jpg","1211.jpg","1212.jpg","1213.jpg","1216.jpg","1217.jpg","1218.jpg","1219.jpg","1220.jpg","1221.jpg","1222.jpg","1223.jpg","1224.jpg","1225.jpg","1227.jpg","1228.jpg","1229.jpg","1230.jpg","1232.jpg","1233.jpg","1234.jpg","1235.jpg","1236.jpg","1237.jpg","1238.jpg","1239.jpg","1240.jpg","1241.jpg","1245.jpg","1246.jpg","1247.jpg","1248.jpg","1249.jpg","1250.jpg","1251.jpg","1252.jpg","1254.jpg","1255.jpg","1256.jpg","1257.jpg","1258.jpg","1259.jpg","1260.jpg","1262.jpg","1263.jpg","1264.jpg","1267.jpg","1268.jpg","1270.jpg","1271.jpg","1272.jpg","1274.jpg","1275.jpg","1279.jpg","1280.jpg","1281.jpg","1282.jpg","1283.jpg","1284.jpg","1285.jpg","1286.jpg","1287.jpg","1289.jpg","1291.jpg","1292.jpg","1293.jpg","1294.jpg","1295.jpg","1296.jpg","1298.jpg","1300.jpg","1301.jpg","1302.jpg","1303.jpg","1304.jpg","1305.jpg","1306.jpg","1307.jpg","1308.jpg","1309.jpg","1311.jpg","1312.jpg","1313.jpg","1314.jpg","1315.jpg","1316.jpg","1317.jpg","1318.jpg","1320.jpg","1323.jpg","1324.jpg","1325.jpg","1326.jpg","1327.jpg","1328.jpg","1329.jpg","1330.jpg","1331.jpg","1332.jpg","1333.jpg","1335.jpg","1336.jpg","1337.jpg","1338.jpg","1339.jpg","1340.jpg","1341.jpg","1344.jpg","1345.jpg","1346.jpg","1347.jpg","1348.jpg","1349.jpg","1351.jpg","1352.jpg","1353.jpg","1355.jpg","1356.jpg","1358.jpg","1360.jpg","1361.jpg","1363.jpg","1364.jpg","1366.jpg","1368.jpg","1369.jpg","1370.jpg","1371.jpg","1372.jpg","1374.jpg","1375.jpg","1379.jpg","1380.jpg","1382.jpg","1383.jpg","1384.jpg","1385.jpg","1386.jpg","1387.jpg","1388.jpg","1389.jpg","1390.jpg","1391.jpg","1392.jpg","1394.jpg","1395.jpg","1396.jpg","1397.jpg","1398.jpg","1399.jpg","1402.jpg","1403.jpg","1404.jpg","1405.jpg","1406.jpg","1407.jpg","1408.jpg","1410.jpg","1411.jpg","1413.jpg","1414.jpg","1416.jpg","1417.jpg","1418.jpg","1419.jpg","1420.jpg","1421.jpg","1422.jpg","1423.jpg","1424.jpg","1425.jpg","1426.jpg","1427.jpg","1428.jpg","1429.jpg","1430.jpg","1431.jpg","1432.jpg","1433.jpg","1434.jpg","1435.jpg","1436.jpg","1437.jpg","1438.jpg","1439.jpg","1440.jpg","1442.jpg","1444.jpg","1446.jpg","1447.jpg","1449.jpg","1450.jpg","1451.jpg","1452.jpg","1453.jpg","1454.jpg","1455.jpg","1456.jpg","1457.jpg","1458.jpg","1459.jpg","1460.jpg","1461.jpg","1462.jpg","1463.jpg","1464.jpg","1465.jpg","1467.jpg","1469.jpg","1470.jpg","1472.jpg","1474.jpg","1475.jpg","1476.jpg","1477.jpg","1478.jpg","1480.jpg","1482.jpg","1484.jpg","1486.jpg","1487.jpg","1489.jpg","1490.jpg","1491.jpg","1493.jpg","1494.jpg","1495.jpg","1496.jpg","1497.jpg","1498.jpg","1500.jpg","1501.jpg","1503.jpg","1504.jpg","1505.jpg","1506.jpg","1507.jpg","1508.jpg","1509.jpg","1510.jpg","1511.jpg","1512.jpg","1513.jpg","1515.jpg","1517.jpg","1518.jpg","1520.jpg","1522.jpg","1523.jpg","1524.jpg","1525.jpg","1526.jpg","1527.jpg","1528.jpg","1529.jpg","1531.jpg","1532.jpg","1533.jpg","1534.jpg","1535.jpg","1536.jpg","1539.jpg","1540.jpg","1541.jpg","1542.jpg","1543.jpg","1544.jpg","1546.jpg","1547.jpg","1548.jpg","1549.jpg","1550.jpg","1551.jpg","1552.jpg","1553.jpg","1554.jpg","1555.jpg","1556.jpg","1557.jpg","1558.jpg","1560.jpg","1561.jpg","1562.jpg","1563.jpg","1564.jpg","1565.jpg","1566.jpg","1567.jpg","1568.jpg","1569.jpg","1570.jpg","1571.jpg","1572.jpg","1573.jpg","1574.jpg","1575.jpg","1577.jpg","1580.jpg","1581.jpg","1582.jpg","1583.jpg","1584.jpg","1585.jpg","1586.jpg","1587.jpg","1588.jpg","1589.jpg","1590.jpg","1591.jpg","1593.jpg","1594.jpg","1595.jpg","1596.jpg","1597.jpg","1598.jpg","1599.jpg","1600.jpg","1601.jpg","1603.jpg","1604.jpg","1605.jpg","1606.jpg","1608.jpg","1609.jpg","1612.jpg","1613.jpg","1614.jpg","1615.jpg","1616.jpg","1617.jpg","1618.jpg","1619.jpg","1620.jpg","1621.jpg","1622.jpg","1623.jpg","1624.jpg","1625.jpg","1626.jpg","1627.jpg","1629.jpg","1630.jpg","1631.jpg","1633.jpg","1635.jpg","1636.jpg","1638.jpg","1639.jpg","1640.jpg","1641.jpg","1643.jpg","1644.jpg","1645.jpg","1647.jpg","1649.jpg","1650.jpg","1651.jpg","1652.jpg","1653.jpg","1654.jpg","1656.jpg","1658.jpg","1659.jpg","1660.jpg","1661.jpg","1662.jpg","1663.jpg","1664.jpg","1665.jpg","1666.jpg","1667.jpg","1668.jpg","1669.jpg","1670.jpg","1671.jpg","1673.jpg","1674.jpg","1675.jpg","1676.jpg","1677.jpg","1678.jpg","1679.jpg","1680.jpg","1681.jpg","1683.jpg","1684.jpg","1685.jpg","1686.jpg","1688.jpg","1689.jpg","1690.jpg","1691.jpg","1692.jpg","1693.jpg","1694.jpg","1695.jpg","1697.jpg","1698.jpg","1699.jpg","1700.jpg","1702.jpg","1703.jpg","1705.jpg","1706.jpg","1707.jpg","1708.jpg","1709.jpg","1712.jpg","1713.jpg","1715.jpg","1716.jpg","1717.jpg","1718.jpg","1720.jpg","1722.jpg","1723.jpg","1724.jpg","1725.jpg","1726.jpg","1727.jpg","1728.jpg","1729.jpg","1730.jpg","1732.jpg","1733.jpg","1734.jpg","1735.jpg","1736.jpg","1738.jpg","1739.jpg","1740.jpg","1741.jpg","1742.jpg","1743.jpg","1744.jpg","1746.jpg","1749.jpg","1750.jpg","1751.jpg","1752.jpg","1753.jpg","1754.jpg","1755.jpg","1759.jpg","1760.jpg","1761.jpg","1762.jpg","1763.jpg","1765.jpg","1766.jpg","1767.jpg","1769.jpg","1770.jpg","1771.jpg","1772.jpg","1774.jpg","1775.jpg","1776.jpg","1777.jpg","1778.jpg","1779.jpg","1780.jpg","1781.jpg","1782.jpg","1783.jpg","1784.jpg","1785.jpg","1786.jpg","1787.jpg","1788.jpg","1789.jpg","1790.jpg","1791.jpg","1792.jpg","1793.jpg","1794.jpg","1795.jpg","1796.jpg","1799.jpg"};
    public static final String[]TEST_M={"1.jpg","3.jpg","4.jpg","8.jpg","13.jpg","17.jpg","24.jpg","33.jpg","39.jpg","44.jpg","46.jpg","52.jpg","54.jpg","57.jpg","58.jpg","62.jpg","65.jpg","66.jpg","68.jpg","73.jpg","74.jpg","81.jpg","85.jpg","87.jpg","91.jpg","92.jpg","93.jpg","110.jpg","129.jpg","139.jpg","156.jpg","161.jpg","173.jpg","179.jpg","182.jpg","184.jpg","185.jpg","186.jpg","190.jpg","193.jpg","195.jpg","196.jpg","200.jpg","208.jpg","231.jpg","232.jpg","237.jpg","238.jpg","247.jpg","256.jpg","257.jpg","258.jpg","260.jpg","275.jpg","276.jpg","303.jpg","307.jpg","310.jpg","317.jpg","318.jpg","322.jpg","338.jpg","342.jpg","343.jpg","348.jpg","355.jpg","356.jpg","360.jpg","364.jpg","375.jpg","381.jpg","382.jpg","384.jpg","389.jpg","390.jpg","394.jpg","399.jpg","402.jpg","405.jpg","407.jpg","409.jpg","419.jpg","421.jpg","424.jpg","426.jpg","428.jpg","430.jpg","436.jpg","437.jpg","445.jpg","450.jpg","459.jpg","463.jpg","474.jpg","479.jpg","483.jpg","501.jpg","504.jpg","507.jpg","508.jpg","511.jpg","519.jpg","528.jpg","536.jpg","546.jpg","554.jpg","556.jpg","562.jpg","568.jpg","570.jpg","572.jpg","573.jpg","575.jpg","590.jpg","592.jpg","593.jpg","596.jpg","602.jpg","603.jpg","614.jpg","615.jpg","619.jpg","628.jpg","638.jpg","641.jpg","643.jpg","653.jpg","658.jpg","660.jpg","667.jpg","670.jpg","674.jpg","679.jpg","686.jpg","694.jpg","696.jpg","706.jpg","709.jpg","713.jpg","735.jpg","739.jpg","744.jpg","747.jpg","749.jpg","757.jpg","759.jpg","762.jpg","763.jpg","771.jpg","773.jpg","778.jpg","792.jpg","794.jpg","797.jpg","804.jpg","808.jpg","817.jpg","822.jpg","826.jpg","832.jpg","836.jpg","838.jpg","839.jpg","852.jpg","859.jpg","864.jpg","866.jpg","868.jpg","871.jpg","874.jpg","881.jpg","887.jpg","893.jpg","894.jpg","905.jpg","927.jpg","934.jpg","937.jpg","940.jpg","947.jpg","948.jpg","949.jpg","953.jpg","985.jpg","992.jpg","994.jpg","995.jpg","1003.jpg","1005.jpg","1007.jpg","1009.jpg","1012.jpg","1019.jpg","1022.jpg","1026.jpg","1027.jpg","1037.jpg","1038.jpg","1046.jpg","1049.jpg","1056.jpg","1058.jpg","1065.jpg","1074.jpg","1080.jpg","1083.jpg","1089.jpg","1091.jpg","1094.jpg","1112.jpg","1113.jpg","1119.jpg","1120.jpg","1128.jpg","1135.jpg","1141.jpg","1150.jpg","1151.jpg","1155.jpg","1156.jpg","1160.jpg","1164.jpg","1166.jpg","1170.jpg","1173.jpg","1176.jpg","1185.jpg","1186.jpg","1187.jpg","1189.jpg","1194.jpg","1197.jpg","1198.jpg","1211.jpg","1215.jpg","1216.jpg","1218.jpg","1220.jpg","1221.jpg","1227.jpg","1228.jpg","1229.jpg","1234.jpg","1238.jpg","1242.jpg","1251.jpg","1254.jpg","1256.jpg","1262.jpg","1266.jpg","1267.jpg","1268.jpg","1280.jpg","1286.jpg","1289.jpg","1292.jpg","1295.jpg","1297.jpg","1300.jpg","1310.jpg","1320.jpg","1322.jpg","1330.jpg","1333.jpg","1336.jpg","1339.jpg","1342.jpg","1347.jpg","1354.jpg","1355.jpg","1358.jpg","1359.jpg","1364.jpg","1366.jpg","1370.jpg","1374.jpg","1378.jpg","1379.jpg","1380.jpg","1382.jpg","1383.jpg","1395.jpg","1401.jpg","1402.jpg","1417.jpg","1422.jpg","1425.jpg","1427.jpg","1430.jpg","1436.jpg","1438.jpg","1443.jpg","1447.jpg","1451.jpg","1461.jpg","1464.jpg","1484.jpg","1489.jpg","1495.jpg","1499.jpg"};
    public static final String[]TEST_B={"1.jpg","2.jpg","5.jpg","8.jpg","9.jpg","10.jpg","11.jpg","15.jpg","16.jpg","18.jpg","31.jpg","37.jpg","44.jpg","45.jpg","54.jpg","57.jpg","61.jpg","63.jpg","71.jpg","74.jpg","83.jpg","84.jpg","88.jpg","90.jpg","95.jpg","97.jpg","98.jpg","105.jpg","119.jpg","122.jpg","127.jpg","140.jpg","144.jpg","147.jpg","154.jpg","161.jpg","185.jpg","186.jpg","205.jpg","211.jpg","215.jpg","216.jpg","224.jpg","226.jpg","237.jpg","245.jpg","250.jpg","251.jpg","253.jpg","256.jpg","259.jpg","267.jpg","268.jpg","273.jpg","284.jpg","287.jpg","289.jpg","291.jpg","300.jpg","303.jpg","304.jpg","311.jpg","313.jpg","317.jpg","322.jpg","323.jpg","330.jpg","332.jpg","335.jpg","344.jpg","348.jpg","354.jpg","356.jpg","367.jpg","369.jpg","378.jpg","379.jpg","385.jpg","388.jpg","389.jpg","394.jpg","404.jpg","407.jpg","414.jpg","419.jpg","427.jpg","428.jpg","433.jpg","436.jpg","437.jpg","444.jpg","449.jpg","454.jpg","457.jpg","462.jpg","475.jpg","479.jpg","484.jpg","495.jpg","520.jpg","527.jpg","532.jpg","535.jpg","536.jpg","545.jpg","548.jpg","552.jpg","558.jpg","559.jpg","560.jpg","561.jpg","562.jpg","566.jpg","572.jpg","573.jpg","574.jpg","575.jpg","580.jpg","588.jpg","596.jpg","603.jpg","612.jpg","623.jpg","624.jpg","626.jpg","631.jpg","634.jpg","635.jpg","636.jpg","642.jpg","644.jpg","647.jpg","658.jpg","659.jpg","660.jpg","663.jpg","664.jpg","668.jpg","670.jpg","676.jpg","678.jpg","681.jpg","689.jpg","690.jpg","700.jpg","712.jpg","717.jpg","720.jpg","721.jpg","724.jpg","725.jpg","751.jpg","758.jpg","762.jpg","763.jpg","768.jpg","770.jpg","773.jpg","774.jpg","778.jpg","790.jpg","802.jpg","805.jpg","806.jpg","813.jpg","819.jpg","820.jpg","823.jpg","824.jpg","825.jpg","826.jpg","828.jpg","830.jpg","840.jpg","854.jpg","869.jpg","874.jpg","878.jpg","880.jpg","887.jpg","888.jpg","892.jpg","893.jpg","894.jpg","895.jpg","903.jpg","906.jpg","909.jpg","912.jpg","916.jpg","919.jpg","925.jpg","936.jpg","943.jpg","944.jpg","946.jpg","958.jpg","962.jpg","964.jpg","972.jpg","982.jpg","993.jpg","998.jpg","1003.jpg","1006.jpg","1013.jpg","1018.jpg","1019.jpg","1023.jpg","1025.jpg","1029.jpg","1033.jpg","1034.jpg","1042.jpg","1044.jpg","1055.jpg","1061.jpg","1070.jpg","1074.jpg","1077.jpg","1080.jpg","1087.jpg","1088.jpg","1092.jpg","1110.jpg","1112.jpg","1115.jpg","1118.jpg","1125.jpg","1126.jpg","1130.jpg","1134.jpg","1135.jpg","1139.jpg","1143.jpg","1146.jpg","1151.jpg","1155.jpg","1166.jpg","1170.jpg","1173.jpg","1177.jpg","1180.jpg","1183.jpg","1191.jpg","1200.jpg","1203.jpg","1208.jpg","1214.jpg","1215.jpg","1226.jpg","1231.jpg","1242.jpg","1243.jpg","1244.jpg","1253.jpg","1261.jpg","1265.jpg","1266.jpg","1269.jpg","1273.jpg","1276.jpg","1277.jpg","1278.jpg","1288.jpg","1290.jpg","1297.jpg","1299.jpg","1310.jpg","1319.jpg","1321.jpg","1322.jpg","1334.jpg","1342.jpg","1343.jpg","1350.jpg","1354.jpg","1357.jpg","1359.jpg","1362.jpg","1365.jpg","1367.jpg","1373.jpg","1376.jpg","1377.jpg","1378.jpg","1381.jpg","1393.jpg","1400.jpg","1401.jpg","1409.jpg","1412.jpg","1415.jpg","1441.jpg","1443.jpg","1445.jpg","1448.jpg","1466.jpg","1468.jpg","1471.jpg","1473.jpg","1479.jpg","1481.jpg","1483.jpg","1485.jpg","1488.jpg","1492.jpg","1499.jpg","1502.jpg","1514.jpg","1516.jpg","1519.jpg","1521.jpg","1530.jpg","1537.jpg","1538.jpg","1545.jpg","1559.jpg","1576.jpg","1578.jpg","1579.jpg","1592.jpg","1602.jpg","1607.jpg","1610.jpg","1611.jpg","1628.jpg","1632.jpg","1634.jpg","1637.jpg","1642.jpg","1646.jpg","1648.jpg","1655.jpg","1657.jpg","1672.jpg","1682.jpg","1687.jpg","1696.jpg","1701.jpg","1704.jpg","1710.jpg","1711.jpg","1714.jpg","1719.jpg","1721.jpg","1731.jpg","1737.jpg","1745.jpg","1747.jpg","1748.jpg","1756.jpg","1757.jpg","1758.jpg","1764.jpg","1768.jpg","1773.jpg","1797.jpg","1798.jpg","1800.jpg"};
    public static void main(String[] args) {

  //      preProcessData();
        //81.67 percent jar achieved with NeuralNetwork ne = new NeuralNetwork(new int[]{WIDTH_OF_DATA-2,150,800,200,100,2}); and normalise of 255.0/2.0 and bias of -1.0 and relU
        NeuralNetwork ne = new NeuralNetwork(new int[]{WIDTH_OF_DATA-2,170,800,800,500,100,2});
        //splitBestNetIntoSixTextFiles(ne);
        ne.setAllWeightsAndBiases(getNetNonGeneralForSixFiles());
        //ne.setAllWeightsAndBiases(ne.loadBestNetWithThisArchitecture());
    /*   double learningRate=0.001;int noOfEpochs=30;int batchSize=64;//try 300
        int totalSizeOfTrainingData=NO_OF_TRAIN_IMAGES;int printAfterNoIterations =20;
        ne.trainNew(learningRate, noOfEpochs, batchSize,  totalSizeOfTrainingData,printAfterNoIterations);  */
        MenuUI m=new MenuUI();
        m.setVisible(true);
        m.init(ne);
    }
    public static double[][]getTrainingData(){
       
        int yIndex=0;int xIndex=0;
        int pixel;
        double r,g,b;
        File f;
        BufferedImage img=null;
        double data [][]=new double[2637][WIDTH_OF_DATA];
        for(int index=0;index<TRAIN_M.length;index++){
            try{
                f = new File(TRAIN_PATH_M+TRAIN_M[index]);
                img =ImageIO.read(f);
                xIndex=2;
                data[yIndex][0]=1;data[yIndex][1]=0;//data[yIndex][2]=fileNo;
                for(int y=0;y<img.getHeight();y++){
                    for(int x=0;x<img.getWidth();x++){
                        pixel = img.getRGB(x, y);
                        r = (pixel>>16) & 0xff;
                        g = (pixel>>8) & 0xff;
                        b = pixel & 0xff;
                       // pixel = (a<<24) | (average<<16) | (average<<8) | average;
                       // data[yIndex][xIndex]=Double.parseDouble(String.valueOf(pixel))/NORMALISE_VALUE;
                      data[yIndex][xIndex]=(r/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                    data[yIndex][xIndex]=(b/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                        data[yIndex][xIndex]=(g/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                    }
                }
                yIndex++;
            }catch(IOException e){
                System.out.println("Error "+e);
            }
        }
        for(int index=0;index<TRAIN_B.length;index++){
            try{
                f = new File(TRAIN_PATH_B+TRAIN_B[index]);
                img =ImageIO.read(f);
                xIndex=2;
                data[yIndex][0]=0;data[yIndex][1]=1;//data[yIndex][2]=fileNo;
                for(int y=0;y<img.getHeight();y++){
                    for(int x=0;x<img.getWidth();x++){
                        pixel = img.getRGB(x, y);
                        r = (pixel>>16) & 0xff;
                        g = (pixel>>8) & 0xff;
                        b = pixel & 0xff;
                       // pixel = (a<<24) | (average<<16) | (average<<8) | average;
                       // data[yIndex][xIndex]=Double.parseDouble(String.valueOf(pixel))/NORMALISE_VALUE;
                                          data[yIndex][xIndex]=(r/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                    data[yIndex][xIndex]=(b/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                        data[yIndex][xIndex]=(g/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                    }
                }
                yIndex++;
            }catch(IOException e){
                System.out.println("Error "+e);
            }
        }
        return data;
    }
    public static double[][]getTestData(){
        int yIndex=0;int xIndex=0;
        int pixel;
        double r,g,b;
        File f;
        BufferedImage img=null;   
        double [][]data= new double[660][WIDTH_OF_DATA];
        yIndex=0;xIndex=0;
        for(int index=0;index<TEST_M.length;index++){
            try{
                f = new File(TEST_PATH_M+TEST_M[index]);
                img =ImageIO.read(f);
                xIndex=2;
                data[yIndex][0]=1;data[yIndex][1]=0;//data[yIndex][2]=fileNo;
                for(int y=0;y<img.getHeight();y++){
                    for(int x=0;x<img.getWidth();x++){
                        pixel = img.getRGB(x, y);
                        r = (pixel>>16) & 0xff;
                        g = (pixel>>8) & 0xff;
                        b = pixel & 0xff;
                       // pixel = (a<<24) | (average<<16) | (average<<8) | average;
                       // data[yIndex][xIndex]=Double.parseDouble(String.valueOf(pixel))/NORMALISE_VALUE;
                                          data[yIndex][xIndex]=(r/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                    data[yIndex][xIndex]=(b/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                        data[yIndex][xIndex]=(g/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                    }
                }
                yIndex++;
            }catch(IOException e){
                System.out.println("Error "+e);
            }
        }
        for(int index=0;index<TEST_B.length;index++){
            try{
                f = new File(TEST_PATH_B+TEST_B[index]);
                img =ImageIO.read(f);
                xIndex=2;
                data[yIndex][0]=0;data[yIndex][1]=1;//data[yIndex][2]=fileNo;
                for(int y=0;y<img.getHeight();y++){
                    for(int x=0;x<img.getWidth();x++){
                        pixel = img.getRGB(x, y);
                        r = (pixel>>16) & 0xff;
                        g = (pixel>>8) & 0xff;
                        b = pixel & 0xff;
                       // pixel = (a<<24) | (average<<16) | (average<<8) | average;
                       // data[yIndex][xIndex]=Double.parseDouble(String.valueOf(pixel))/NORMALISE_VALUE;
                                          data[yIndex][xIndex]=(r/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                    data[yIndex][xIndex]=(b/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                        data[yIndex][xIndex]=(g/NORMALISE_VALUE)+BIAS_TO_DATA;
                        xIndex++;
                    }
                }
                yIndex++;
            }catch(IOException e){
                System.out.println("Error "+e);
            }
        }
        return data;
    }
    public static void preProcessData(){
        try{
            FileReader read = new FileReader(HAS_BEEN_PROCESSED);
            BufferedReader buffRead = new BufferedReader(read);
            if( buffRead.readLine().equals(PROCESSED)){
                return;
            }buffRead.close();read.close();
        }catch(IOException e){
            System.out.println("Error with "+HAS_BEEN_PROCESSED+" "+e);
        }
        int yIndex=0;int xIndex=0;
        int pixel;
        int r,g,b;
        int fileNo=0;
        File f;
        int count=0;
        String dat="";
        BufferedImage img=null;
        String trainPathM="archive"+File.separator+"train"+File.separator+"malignant"+File.separator;
        String trainPathB="archive"+File.separator+"train"+File.separator+"benign"+File.separator;
       // String data [][]=new String[2637][WIDTH_OF_DATA];
        fileNo=0;
        String contents="";
        while(fileNo<=MAX_NO_OF_TRAIN_IMAGES_M){
            fileNo++;
            try{
                f = new File(trainPathM+String.valueOf(fileNo)+".jpg");
                img =ImageIO.read(f);
                xIndex=3;
                count++;System.out.println("Count "+count);
                if(yIndex!=0){
                    dat=dat+System.lineSeparator();
                }
                dat=dat+"1"+SEP_CHAR+"0"+SEP_CHAR+String.valueOf(fileNo);
               // data[yIndex][0]="1";data[yIndex][1]="0";data[yIndex][2]=String.valueOf(fileNo);
                for(int y=0;y<img.getHeight();y++){
                    for(int x=0;x<img.getWidth();x++){
                        pixel = img.getRGB(x, y);
                        r = (pixel>>16) & 0xff;
                        g = (pixel>>8) & 0xff;
                        b = pixel & 0xff;
                    /*    data[yIndex][xIndex]=String.valueOf(r/NORMALISE_VALUE);
                        xIndex++;
                        data[yIndex][xIndex]=String.valueOf(b/NORMALISE_VALUE);
                        xIndex++;
                        data[yIndex][xIndex]=String.valueOf(g/NORMALISE_VALUE);
                        xIndex++;*/
                        dat = dat+SEP_CHAR+String.valueOf(r)+SEP_CHAR+String.valueOf(b)+SEP_CHAR+String.valueOf(g);
                    }
                }
                yIndex++;
            }catch(IOException e){
                
            }
        }
        fileNo=0;
        while(fileNo<=MAX_NO_OF_TRAIN_IMAGES_B){
            fileNo++;
            try{
                f = new File(trainPathB+String.valueOf(fileNo)+".jpg");
                img =ImageIO.read(f);
                xIndex=3;
                dat=dat+"0"+SEP_CHAR+"1"+SEP_CHAR+String.valueOf(fileNo);
               // data[yIndex][0]="0";data[yIndex][1]="1";data[yIndex][2]=String.valueOf(fileNo);
                for(int y=0;y<img.getHeight();y++){
                    for(int x=0;x<img.getWidth();x++){
                        pixel = img.getRGB(x, y);
                        r = (pixel>>16) & 0xff;
                        g = (pixel>>8) & 0xff;
                        b = pixel & 0xff;
                        /*                                         data[yIndex][xIndex]=String.valueOf(r/NORMALISE_VALUE);
                        xIndex++;
                    data[yIndex][xIndex]=String.valueOf(b/NORMALISE_VALUE);
                        xIndex++;
                        data[yIndex][xIndex]=String.valueOf(g/NORMALISE_VALUE);
                        xIndex++;*/
                        dat = dat+SEP_CHAR+String.valueOf(r)+SEP_CHAR+String.valueOf(b)+SEP_CHAR+String.valueOf(g);
                    }
                }
                yIndex++;
            }catch(IOException e){
                
            }
        }  
        try{
            contents="";
            FileWriter write = new FileWriter(TRAIN_DATA,false);
            BufferedWriter buffWrite = new BufferedWriter(write);
            String line="";
            System.out.println("Creating contents string");
        /*    for(int y=0;y<data.length;y++){
             //   line = String.valueOf(data[y][0]);
                System.out.println("Y "+y);
                if(y!=0){
                    contents=contents+System.lineSeparator();
                    //buffWrite.newLine();
                }
                contents=contents+data[y][0];
                for(int x=1;x<data[0].length;x++){
                    //line = line +SEP_CHAR+String.valueOf(data[y][x]);
                    contents = contents+SEP_CHAR+data[y][x];
                }
                //contents=contents+line;
                //buffWrite.write(line);
                
            }*/
            System.out.println("Created contents string");
            buffWrite.write(dat);
            System.out.println("Writen to file");
            buffWrite.flush();write.flush();
            buffWrite.close();write.close();
        }catch(IOException e){
            System.out.println("Error with file "+TRAIN_DATA+" : "+e);
        }       
        String testPathM="archive"+File.separator+"test"+File.separator+"malignant"+File.separator;
        String testPathB="archive"+File.separator+"test"+File.separator+"benign"+File.separator;
      //  data= new String[660][WIDTH_OF_DATA];
        yIndex=0;xIndex=0;
        fileNo=0;dat="";
        while(fileNo<=MAX_NO_OF_TEST_IMAGES_M){
            fileNo++;
            try{
                f = new File(testPathM+String.valueOf(fileNo)+".jpg");
                img =ImageIO.read(f);
                xIndex=3;
                if(yIndex!=0){
                    dat =dat+System.lineSeparator();
                }
                dat=dat+"1"+SEP_CHAR+"0"+SEP_CHAR+String.valueOf(fileNo);
                //data[yIndex][0]="1";data[yIndex][1]="0";data[yIndex][2]=String.valueOf(fileNo);
                for(int y=0;y<img.getHeight();y++){
                    for(int x=0;x<img.getWidth();x++){
                        pixel = img.getRGB(x, y);
                        r = (pixel>>16) & 0xff;
                        g = (pixel>>8) & 0xff;
                        b = pixel & 0xff;
                              /*                                     data[yIndex][xIndex]=String.valueOf(r/NORMALISE_VALUE);
                        xIndex++;
                    data[yIndex][xIndex]=String.valueOf(b/NORMALISE_VALUE);
                        xIndex++;
                        data[yIndex][xIndex]=String.valueOf(g/NORMALISE_VALUE);
                        xIndex++;*/
                              dat = dat+SEP_CHAR+String.valueOf(r)+SEP_CHAR+String.valueOf(b)+SEP_CHAR+String.valueOf(g);
                    }
                }
                yIndex++; 
            }catch(IOException e){
                
            }
        }
        fileNo=0;
        while(fileNo<=MAX_NO_OF_TEST_IMAGES_B){
            fileNo++;
            try{
                f = new File(testPathB+String.valueOf(fileNo)+".jpg");
                img =ImageIO.read(f);
                xIndex=3;
                dat=dat+"0"+SEP_CHAR+"1"+SEP_CHAR+String.valueOf(fileNo);
                //data[yIndex][0]="0";data[yIndex][1]="1";data[yIndex][2]=String.valueOf(fileNo);
                for(int y=0;y<img.getHeight();y++){
                    for(int x=0;x<img.getWidth();x++){
                        pixel = img.getRGB(x, y);
                        r = (pixel>>16) & 0xff;
                        g = (pixel>>8) & 0xff;
                        b = pixel & 0xff;
                         /*                                         data[yIndex][xIndex]=String.valueOf(r/NORMALISE_VALUE);
                        xIndex++;
                    data[yIndex][xIndex]=String.valueOf(b/NORMALISE_VALUE);
                        xIndex++;
                        data[yIndex][xIndex]=String.valueOf(g/NORMALISE_VALUE);
                        xIndex++;*/
                        dat = dat+SEP_CHAR+String.valueOf(r)+SEP_CHAR+String.valueOf(b)+SEP_CHAR+String.valueOf(g);
                    }
                }
                yIndex++;
            }catch(IOException e){
                
            }
        }
        try{
            contents="";
            FileWriter write = new FileWriter(TEST_DATA,false);
            BufferedWriter buffWrite = new BufferedWriter(write);
            String line="";
        /*    for(int y=0;y<data.length;y++){
                //line = String.valueOf(data[y][0]);
                if(y!=0){
                    contents=contents+System.lineSeparator();
                    //buffWrite.newLine();
                }
                contents=contents+data[y][0];
                System.out.println("Y "+y);
                for(int x=1;x<data[0].length;x++){
                    //line = line +SEP_CHAR+String.valueOf(data[y][x]);
                    contents=contents+SEP_CHAR+data[y][x];
                }
                //buffWrite.write(line);
            }*/
            buffWrite.write(dat);
            buffWrite.flush();write.flush();
            buffWrite.close();write.close();
        }catch(IOException e){
            System.out.println("Error with file "+TEST_DATA+" : "+e);
        }
        
        try{
            FileWriter w = new FileWriter(HAS_BEEN_PROCESSED,false);
            BufferedWriter bW = new BufferedWriter(w);
            bW.write(PROCESSED);
            bW.flush();w.flush();
            bW.close();w.close();
        }catch(IOException e){
            System.out.println("Error "+e);
            return;
        }
    }
    public static void getBestNetWithArchAndWriteToAFile(NeuralNetwork ne){
        double[]a = ne.loadBestNetWithThisArchitecture();
        try{
            FileWriter w = new FileWriter(BEST_NET,false);
            BufferedWriter buffWrite = new BufferedWriter(w);
            for(int y=0;y<a.length;y++){
                if(y!=0){
                    buffWrite.newLine();
                }
                buffWrite.write(String.valueOf(a[y]));
            }
            buffWrite.flush();w.flush();
            buffWrite.close();w.close();
        }catch(IOException e){
            System.out.println("Error "+e);
        }  
    }
    public static double[] setNetToBest(NeuralNetwork ne){
        double a[]=new double[ne.allLength];
        try{
            FileReader read = new FileReader(BEST_NET);
            BufferedReader buffRead = new BufferedReader(read);
            for(int y=0;y<a.length;y++){
                a[y]=Double.parseDouble(buffRead.readLine());
            }
            buffRead.close();read.close();
        }catch(IOException e){
            System.out.println("Error "+e);
        }
        return a;
    }
    public static void splitBestNetIntoSixTextFiles(NeuralNetwork ne){

        double all[]= setNetToBest(ne);//no is 3813442
        int in=0;int currIn=0;
        try{
            FileWriter one = new FileWriter("one.txt",false);
            BufferedWriter buffOne = new BufferedWriter(one);
            for(currIn=0;currIn<4469722;currIn++){
                if(currIn==0){
                    buffOne.write(String.valueOf(all[in]));
                    in++;
                }else{
                    buffOne.newLine();
                    buffOne.write(String.valueOf(all[in]));
                    in++;
                }
            }
            buffOne.flush();one.flush();
            buffOne.close();one.close();
            FileWriter two = new FileWriter("two.txt",false);
            BufferedWriter buffTwo = new BufferedWriter(two);
            for(currIn=0;currIn<4469722;currIn++){
                if(currIn==0){
                    buffTwo.write(String.valueOf(all[in]));
                    in++;
                }else{
                    buffTwo.newLine();
                    buffTwo.write(String.valueOf(all[in]));
                    in++;
                }
            }
            buffTwo.flush();two.flush();
            buffTwo.close();two.close();
            FileWriter three = new FileWriter("three.txt",false);
            BufferedWriter buffThree = new BufferedWriter(three);
            for(currIn=0;currIn<4469722;currIn++){
                if(currIn==0){
                    buffThree.write(String.valueOf(all[in]));
                    in++;
                }else{
                    buffThree.newLine();
                    buffThree.write(String.valueOf(all[in]));
                    in++;
                }
            }
            buffThree.flush();three.flush();
            buffThree.close();three.close();
            FileWriter four = new FileWriter("four.txt",false);
            BufferedWriter buffFour = new BufferedWriter(four);
            for(currIn=0;currIn<4469722;currIn++){
                if(currIn==0){
                    buffFour.write(String.valueOf(all[in]));
                    in++;
                }else{
                    buffFour.newLine();
                    buffFour.write(String.valueOf(all[in]));
                    in++;
                }
            }
            buffFour.flush();four.flush();
            buffFour.close();four.close();
            FileWriter five = new FileWriter("five.txt",false);
            BufferedWriter buffFive = new BufferedWriter(five);
            for(currIn=0;currIn<4469722;currIn++){
                if(currIn==0){
                    buffFive.write(String.valueOf(all[in]));
                    in++;
                }else{
                    buffFive.newLine();
                    buffFive.write(String.valueOf(all[in]));
                    in++;
                }
            }
            buffFive.flush();five.flush();
            buffFive.close();five.close();
            FileWriter six = new FileWriter("six.txt",false);
            BufferedWriter buffSix = new BufferedWriter(six);
            for(currIn=0;currIn<4469722;currIn++){
                if(currIn==0){
                    buffSix.write(String.valueOf(all[in]));
                    in++;
                }else{
                    buffSix.newLine();
                    buffSix.write(String.valueOf(all[in]));
                    in++;
                }
            }
            buffSix.flush();six.flush();
            buffSix.close();six.close();
        }catch(IOException e){
            System.out.println("Error "+e);
        }

    }
    public static double[] getNetNonGeneralForSixFiles(){
        double all[] = new double[26818332];
        int in=0;int currIn;
        try{
            FileReader one = new FileReader("one.txt");
            BufferedReader buffOne = new BufferedReader(one);
            for(currIn=0;currIn<4469722;currIn++){
                all[in]=Double.parseDouble(buffOne.readLine());
                in++;
            }buffOne.close();one.close();
            FileReader two = new FileReader("two.txt");
            BufferedReader buffTwo = new BufferedReader(two);
            for(currIn=0;currIn<4469722;currIn++){
                all[in]=Double.parseDouble(buffTwo.readLine());
                in++;
            }buffTwo.close();two.close();
            FileReader three = new FileReader("three.txt");
            BufferedReader buffThree = new BufferedReader(three);
            for(currIn=0;currIn<4469722;currIn++){
                all[in]=Double.parseDouble(buffThree.readLine());
                in++;
            }buffThree.close();three.close();
            FileReader four = new FileReader("four.txt");
            BufferedReader buffFour = new BufferedReader(four);
            for(currIn=0;currIn<4469722;currIn++){
                all[in]=Double.parseDouble(buffFour.readLine());
                in++;
            }buffFour.close();four.close();
            FileReader five = new FileReader("five.txt");
            BufferedReader buffFive = new BufferedReader(five);
            for(currIn=0;currIn<4469722;currIn++){
                all[in]=Double.parseDouble(buffFive.readLine());
                in++;
            }buffFive.close();five.close();
            FileReader six = new FileReader("six.txt");
            BufferedReader buffSix = new BufferedReader(six);
            for(currIn=0;currIn<4469722;currIn++){
                all[in]=Double.parseDouble(buffSix.readLine());
                in++;
            }buffSix.close();six.close();
        }catch(IOException e){
            System.out.println("error "+e);
        }
        return all;
    }
}
