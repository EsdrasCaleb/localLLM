package com.densebrain.rif.client;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { RIFManagerFactory_getManager_1_0_Test.class, RIFManagerFactory_getInvoker_2_0_Test.class, RIFManagerFactory_getImpl_3_1_Test.class })
public class RIFManagerFactory_Suite {
}
