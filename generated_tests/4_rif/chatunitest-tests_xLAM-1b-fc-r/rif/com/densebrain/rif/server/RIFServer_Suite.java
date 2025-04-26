package com.densebrain.rif.server;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { RIFServer_start_0_1_Test.class, RIFServer_Suite.class, RIFServer_start_0_0_Test.class, RIFServer_stop_1_0_Test.class })
public class RIFServer_Suite {
}
