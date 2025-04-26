package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.ArgumentMatchers.anyInt;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class UniformCrossover_cross_0_1_Test {

    @Mock
    private Random rnd;

    @InjectMocks
    private UniformCrossover uniformCrossover;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testCross() {
        VectorGenome mom = new VectorGenome(new Vector(), new Evaluator() {

            @Override
            public double evaluate(Genome genome) {
                // Implement the evaluate method for testing purposes
                return 0.0;
            }
        });
        VectorGenome dad = new VectorGenome(new Vector(), new Evaluator() {

            @Override
            public double evaluate(Genome genome) {
                // Implement the evaluate method for testing purposes
                return 0.0;
            }
        });
        when(rnd.nextBoolean()).thenReturn(true);
        Genome result = uniformCrossover.cross(mom, dad);
        VectorGenome vSon = (VectorGenome) result;
        for (int i = 0; i < mom.getGenesCount(); i++) {
            assertEquals(mom.getGene(i), vSon.getGene(i));
        }
    }
}
