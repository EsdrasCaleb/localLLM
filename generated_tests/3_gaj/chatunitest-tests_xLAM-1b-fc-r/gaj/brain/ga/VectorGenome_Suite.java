package brain.ga;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { VectorGenome_getGene_2_1_Test.class, VectorGenome_Suite.class, VectorGenome_getGene_2_4_Test.class })
public class VectorGenome_Suite {
}
