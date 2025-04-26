package org.templateit;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { DynamicTemplate_height_0_2_Test.class, DynamicTemplate_width_1_0_Test.class, DynamicTemplate_Suite.class })
public class DynamicTemplate_Suite {
}
