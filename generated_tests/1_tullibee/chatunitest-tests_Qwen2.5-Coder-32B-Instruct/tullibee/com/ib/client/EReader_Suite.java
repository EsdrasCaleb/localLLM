package com.ib.client;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { EReader_run_1_0_Test.class, EReader_stop_12_0_Test.class })
public class EReader_Suite {
}
