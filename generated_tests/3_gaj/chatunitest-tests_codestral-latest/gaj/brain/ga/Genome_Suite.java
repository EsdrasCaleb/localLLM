package brain.ga;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Genome_initialize_1_0_Test.class, Genome_compareTo_3_1_Test.class })
public class Genome_Suite {
}
