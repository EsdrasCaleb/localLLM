package brain.ga;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { GAUtilities_flipCoin_0_2_Test.class, GAUtilities_nextPos_1_1_Test.class })
public class GAUtilities_Suite {
}
