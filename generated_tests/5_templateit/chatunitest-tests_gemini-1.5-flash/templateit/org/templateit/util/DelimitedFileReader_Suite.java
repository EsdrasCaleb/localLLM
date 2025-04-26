package org.templateit.util;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { DelimitedFileReader_next_1_2_Test.class, DelimitedFileReader_remove_2_2_Test.class })
public class DelimitedFileReader_Suite {
}
