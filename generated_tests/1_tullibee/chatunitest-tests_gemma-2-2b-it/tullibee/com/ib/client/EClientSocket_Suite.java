package com.ib.client;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { EClientSocket_faMsgTypeName_0_1_Test.class, EClientSocket_serverVersion_1_1_Test.class, EClientSocket_TwsConnectionTime_2_4_Test.class, EClientSocket_cancelScannerSubscription_7_1_Test.class, EClientSocket_reqScannerSubscription_9_0_Test.class, EClientSocket_reqMktDepth_16_1_Test.class, EClientSocket_reqExecutions_22_1_Test.class, EClientSocket_reqIds_25_1_Test.class, EClientSocket_dataInputStream_51_0_Test.class })
public class EClientSocket_Suite {
}
