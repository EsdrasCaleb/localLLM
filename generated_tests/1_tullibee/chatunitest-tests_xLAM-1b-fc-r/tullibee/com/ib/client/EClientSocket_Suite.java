package com.ib.client;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { EClientSocket_faMsgTypeName_0_1_Test.class, EClientSocket_serverVersion_1_1_Test.class, EClientSocket_eConnect_4_0_Test.class, EClientSocket_eDisconnect_6_1_Test.class, EClientSocket_cancelScannerSubscription_7_4_Test.class, EClientSocket_reqScannerParameters_8_1_Test.class, EClientSocket_faMsgTypeName_0_0_Test.class })
public class EClientSocket_Suite {
}
