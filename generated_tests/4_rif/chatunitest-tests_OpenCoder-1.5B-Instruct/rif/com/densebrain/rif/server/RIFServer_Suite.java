package com.densebrain.rif.server;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { RIFServer_start_0_4_Test.class, RIFServer_stop_1_2_Test.class })
public class RIFServer_Suite {
}
